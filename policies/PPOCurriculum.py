import time
from functools import partial
from typing import Dict

import tree

from polaris.policies.parametrised import ParametrisedPolicy
from polaris.experience import SampleBatch, get_epochs
from polaris.models.utils import EpsilonCategorical
from polaris.policies import VMPO, PolicyParams

import numpy as np
import tensorflow as tf

from polaris.policies.utils.return_based_scaling import ReturnBasedScaling
from polaris.policies.utils.popart import Popart

from seedsmash2.curriculum import Curriculum

tf.compat.v1.enable_eager_execution()

from polaris.policies.utils.misc import explained_variance
from polaris.policies.utils.vtrace import compute_vtrace


class PPOC(ParametrisedPolicy):
    # TODO: take care of the loss, tophalf adv, not much more i think

    def __init__(
            self,
            *args,
            is_online=False,
            config=None,
            policy_config=None,
            options=None,
            **kwargs
    ):

        self.is_online = is_online

        alpha_range = (1e-6, 1e6)

        def trust_region_constraint(v):
            return tf.clip_by_value(
                v,
                *[tf.math.log(c) / policy_config.kl_coeff_speed for c in alpha_range])

        self.log_kl_coeff = tf.Variable(
            tf.math.log(policy_config.initial_kl_coeff) / policy_config.kl_coeff_speed,
            constraint=trust_region_constraint,
            trainable=True,
            dtype=tf.float32)


        if self.is_online:
            self.curriculum_module = Curriculum(
                config=config,
                policy_config=policy_config,
                bot_config=options
            )

        super().__init__(
            *args,
            config=config,
            policy_config=policy_config,
            options=options,
            **kwargs
        )

    def get_weights(self) -> Dict[str, np.ndarray]:
        w = super().get_weights()
        w["log_kl_coeff"] = self.log_kl_coeff.numpy()
        return w

    def set_weights(self, weights: Dict):

        log_kl_coeff = weights.pop("log_kl_coeff", None)
        if log_kl_coeff is not None:
            self.log_kl_coeff.assign(log_kl_coeff)
        else:
            print("No log_kl_coeff ???")

        super().set_weights(weights)


    def get_params(self, online=False):
        if self.is_online:
            return PolicyParams(
                name=self.name,
                weights=self.get_weights(),
                config=self.curriculum_module.train_config,
                options=self.curriculum_module.current_config,
                version=self.version,
                stats=self.stats,
                policy_type=self.policy_type
            )
        else:
            return super().get_params()


    def train(
            self,
            input_batch: SampleBatch
    ):
        preprocess_start_time = time.time()
        res = super().train(input_batch)

        num_sequences = len(input_batch[SampleBatch.SEQ_LENS])

        # TODO: make this a function, can we do this more efficiently ?
        def make_time_major(v):
            try:
                return np.transpose(np.reshape(v, (num_sequences, -1) + v.shape[1:]), (1, 0) + tuple(range(2, 1+len(v.shape))))
            except Exception as e:
                print(v)
                raise e

        def process_batch(b):
            b = dict(b)
            seq_lens = b.pop(SampleBatch.SEQ_LENS)

            time_major_batch = tree.map_structure(make_time_major, b)

            time_major_batch[SampleBatch.SEQ_LENS] = seq_lens
            time_major_batch[SampleBatch.STATE] = tree.map_structure(lambda v: v[0], time_major_batch[SampleBatch.STATE])

            return time_major_batch

        tm_input_batch = process_batch(input_batch)

        nn_train_time = time.time()

        # TODO: use tf dataset, or see whats happening with the for loop inside the tf function
        # It is super slow otherwise

        # normalise advantages
        adv = tm_input_batch[SampleBatch.ADVANTAGES]
        tm_input_batch[SampleBatch.ADVANTAGES][:] = (adv-np.mean(adv)) / np.maximum(1e-4, np.std(adv))

        for minibatch in get_epochs(tm_input_batch,
                                    n_epochs=self.config.n_epochs,
                                    minibatch_size=self.config.minibatch_size
                                    ):
            metrics = self._train(
                input_batch=minibatch
            )


        if self.version % self.curriculum_module.update_freq == 0:
            self.curriculum_module.update(self.version)

        metrics["curriculum"] = self.curriculum_module.get_metrics()

        #self.return_based_scaling.batch_update(metrics.pop("masked_rewards"), metrics.pop("returns"))
        #metrics["return_based_scaling"] = self.return_based_scaling.get_metrics()

        metrics.update(**res,
                       preprocess_time_ms=(nn_train_time-preprocess_start_time)*1000.,
                       grad_time_ms=(time.time() - nn_train_time) * 1000.,
                       )
        return metrics


    # TODO: https://github.com/google-deepmind/rlax/blob/master/rlax/_src/mpo_ops.py#L441#L522

    @tf.function
    def _train(
            self,
            input_batch
    ):
        #B, T = tf.shape(input_batch[SampleBatch.OBS])
        with tf.GradientTape() as tape:
            with tf.device('/gpu:0'):
                (action_logits, _), vf_preds = self.model(
                    input_batch
                )
                mask = tf.transpose(tf.sequence_mask(input_batch[SampleBatch.SEQ_LENS], maxlen=self.config.max_seq_len), [1, 0])
                action_dist = self.model.action_dist(action_logits)
                action_logp = action_dist.logp(input_batch[SampleBatch.ACTION])

                behavior_logp = input_batch[SampleBatch.ACTION_LOGP]
                behavior_dist = self.model.action_dist(input_batch[SampleBatch.ACTION_LOGITS])
                entropy = tf.boolean_mask(action_dist.entropy(), mask)
                # should be computed by workers
                advantages = input_batch[SampleBatch.ADVANTAGES]
                # TODO: append last value in preprocess ?
                vf_targets = input_batch[SampleBatch.VF_TARGETS]

                logp_ratio = tf.exp(action_logp - behavior_logp)

                surrogate_loss = tf.minimum(
                    advantages * logp_ratio,
                    advantages
                    * tf.clip_by_value(
                        logp_ratio,
                        1 - self.policy_config.ppo_clip,
                        1 + self.policy_config.ppo_clip,
                    ),
                )

                policy_loss = -tf.reduce_mean(tf.boolean_mask(surrogate_loss, mask))

                critic_loss = tf.math.square(
                    vf_preds - vf_targets
                )
                critic_loss_clipped = tf.clip_by_value(
                    critic_loss,
                    0,
                    self.policy_config.vf_clip,
                )
                critic_loss = self.policy_config.baseline_coeff * tf.reduce_mean(tf.boolean_mask(critic_loss_clipped, mask))
                mean_entropy = tf.reduce_mean(entropy)

                if self.policy_config.initial_kl_coeff > 0.0:
                    kl_behavior_to_online = tf.boolean_mask(behavior_dist.kl(action_logits), mask)

                    kl_coeff = tf.exp(self.log_kl_coeff * self.policy_config.kl_coeff_speed)
                    kl_loss = (kl_coeff * (self.policy_config.kl_target - tf.stop_gradient(kl_behavior_to_online)) +
                                   tf.stop_gradient(kl_coeff) * kl_behavior_to_online)
                    kl_loss = tf.reduce_mean(kl_loss)

                else:
                    kl_loss = tf.constant(0.0)

                total_loss = critic_loss + policy_loss - mean_entropy * self.policy_config.entropy_cost + kl_loss

        vars = self.model.trainable_variables + (self.log_kl_coeff,)
        gradients = tape.gradient(total_loss, vars)
        gradients, mean_grad_norm = tf.clip_by_global_norm(gradients, self.policy_config.grad_clip)

        self.model.optimiser.apply(gradients, vars)

        mean_entropy = tf.reduce_mean(entropy)
        explained_vf = explained_variance(
            tf.boolean_mask(vf_targets, mask),
            tf.boolean_mask(vf_preds, mask)
        )

        return {
            "mean_entropy": mean_entropy,
            "vf_loss": critic_loss,
            "pi_loss": policy_loss,
            "mean_grad_norm": mean_grad_norm,
            "explained_vf": explained_vf,
            "kl": tf.reduce_mean(kl_behavior_to_online),
            "kl_loss": kl_loss,
            "kl_coeff": kl_coeff,
        }