import time
from functools import partial
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


class APPOAS(ParametrisedPolicy):
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
        self.popart_module = Popart(
            learning_rate=policy_config.popart_lr,
            std_clip=policy_config.popart_std_clip
        )
        if self.is_online:
            self.curriculum_module = Curriculum(
                config=config,
                policy_config=policy_config,
                bot_config=options
            )
            self.offline_policy = APPOAS(
                *args,
                config=config,
                policy_config=policy_config,
                options=options,
                online=False,
                **kwargs)
        else:
            self.offline_policy = None

        super().__init__(
            *args,
            config=config,
            policy_config=policy_config,
            options=options,
            **kwargs
        )

        # self.return_based_scaling = ReturnBasedScaling(
        #     learning_rate=self.policy_config.popart_lr,
        #     std_clip=self.policy_config.popart_std_clip
        # )



    def init_model(self):
        super().init_model()

        if self.offline_policy is not None:
            self.offline_policy.init_model()
            # set weights to offline policy
            self.update_offline_model()
        else:
            pass
            #self.model.action_dist = partial(EpsilonCategorical, epsilon=self.policy_config.random_action_chance)

    def get_weights(self):
        w = super().get_weights()
        w["popart"] = self.popart_module.get_weights()
        return w


    def set_weights(self, weights):
        w = weights.copy()
        popart_weights = w.pop("popart", None)
        if popart_weights is not None:
            self.popart_module.set_weights(popart_weights)

        super().set_weights(w)


    def get_params(self, online=False):
        if online:
            return PolicyParams(
                name=self.name,
                weights=self.get_weights(),
                config=self.policy_config,
                options=self.curriculum_module.current_config,
                version=self.version,
                stats=self.stats,
                policy_type=self.policy_type
            )
        if not online and self.is_online:
            return self.offline_policy.get_params()
        else:
            return super().get_params()

    def update_offline_model(self):
        self.offline_policy.setup(self.get_params(online=True))

    def setup(self, policy_params: "PolicyParams"):
        super().setup(policy_params)
        if self.is_online:
            self.offline_policy.setup(policy_params)
        return self

    def train(
            self,
            input_batch: SampleBatch,
            action_state_values,
            coaching_model = None,
            coaching_batch = None
    ):
        preprocess_start_time = time.time()
        res = super().train(input_batch)

        num_sequences = len(input_batch[SampleBatch.SEQ_LENS])

        # TODO: make this a function, can we do this more efficiently ?
        def make_time_major(v):
            return np.transpose(np.reshape(v, (num_sequences, -1) + v.shape[1:]), (1, 0) + tuple(range(2, 1+len(v.shape))))

        action_states = np.int32(input_batch[SampleBatch.OBS]["categorical"]["action1"])[:, 0]

        action_state_rewards = action_state_values.get_rewards(
            action_states
        )

        action_state_values.push_samples(
            action_states
        )

        input_batch["action_state_rewards"] = action_state_rewards


        def add_last_timestep(v1, v2):
            return np.concatenate([v1, v2[-1:]], axis=0)

        def process_batch(b):
            b = dict(b)
            seq_lens = b.pop(SampleBatch.SEQ_LENS)
            time_major_batch = tree.map_structure(make_time_major, b)

            # def get_samples(v):
            #     return v[:, :4]
            #
            # print(tree.map_structure(get_samples, time_major_batch[SampleBatch.OBS]))

            time_major_batch[SampleBatch.OBS] = tree.map_structure(
                add_last_timestep,
                time_major_batch[SampleBatch.OBS], time_major_batch[SampleBatch.NEXT_OBS]
            )

            # print(tree.map_structure(get_samples, time_major_batch[SampleBatch.OBS]))
            #
            # print(time_major_batch[SampleBatch.AGENT_ID])


            time_major_batch[SampleBatch.PREV_REWARD] = np.concatenate([time_major_batch[SampleBatch.PREV_REWARD], time_major_batch[SampleBatch.REWARD][-1:]], axis=0)
            time_major_batch[SampleBatch.PREV_ACTION] = np.concatenate([time_major_batch[SampleBatch.PREV_ACTION], time_major_batch[SampleBatch.ACTION][-1:]], axis=0)
            time_major_batch[SampleBatch.SEQ_LENS] = seq_lens
            time_major_batch[SampleBatch.STATE] = tree.map_structure(lambda v: v[0], time_major_batch[SampleBatch.STATE])


            # Sequence is one step longer if we are not done at timestep T
            time_major_batch[SampleBatch.SEQ_LENS][
                np.logical_and(seq_lens == self.config.max_seq_len,
                               np.logical_not(time_major_batch[SampleBatch.DONE][-1])
                               )] += 1

            return time_major_batch

        tm_input_batch = process_batch(input_batch)
        if coaching_batch is not None:
            coaching_batch = process_batch(coaching_batch)


        nn_train_time = time.time()

        # TODO: use tf dataset, or see whats happening with the for loop inside the tf function
        # It is super slow otherwise



        metrics = self._train(
            input_batch=tm_input_batch,
            coaching_model=coaching_model,
            coaching_batch=coaching_batch,
        )

        popart_update_time = time.time()

        #self.return_based_scaling.batch_update(metrics.pop("masked_rewards"), metrics.pop("returns"))
        #metrics["return_based_scaling"] = self.return_based_scaling.get_metrics()

        self.popart_module.batch_update(metrics["vtrace_mean"], metrics["vtrace_std"], value_out=self.model._value_out)
        metrics["popart"] = self.popart_module.get_metrics()
        metrics["version"] = self.version

        metrics["action_state_values"] = action_state_values.get_metrics()

        if self.version % self.policy_config.target_update_freq == 0:
            self.update_offline_model()

        if self.version % self.curriculum_module.update_freq == 0:
            self.curriculum_module.update(self.version)

        metrics["curriculum"] = self.curriculum_module.get_metrics()


        metrics.update(**res,
                       preprocess_time_ms=(nn_train_time-preprocess_start_time)*1000.,
                       grad_time_ms=(popart_update_time - nn_train_time) * 1000.,
                       popart_update_ms=(time.time()-popart_update_time)*1000.,
                       )

        return metrics


    # TODO: https://github.com/google-deepmind/rlax/blob/master/rlax/_src/mpo_ops.py#L441#L522

    @tf.function
    def _train(
            self,
            input_batch,
            coaching_model=None,
            coaching_batch=None
    ):
        #B, T = tf.shape(input_batch[SampleBatch.OBS])
        with (tf.GradientTape() as tape):
            with tf.device('/gpu:0'):
                (action_logits, _), all_vf_preds = self.model(
                    input_batch
                )

                if self.policy_config.target_update_freq == 1:
                    offline_logits = action_logits
                else:
                    (offline_logits, _), _ = self.offline_policy.model(
                        input_batch
                    )

                mask_all = tf.transpose(tf.sequence_mask(input_batch[SampleBatch.SEQ_LENS], maxlen=self.config.max_seq_len+1), [1, 0])
                mask = mask_all[:-1]
                action_logits = action_logits[:-1]
                action_dist = self.model.action_dist(action_logits)
                action_logp = action_dist.logp(input_batch[SampleBatch.ACTION])

                offline_action_dist = self.model.action_dist(tf.stop_gradient(offline_logits[:-1]))
                offline_action_logp = offline_action_dist.logp(input_batch[SampleBatch.ACTION])

                behavior_logp = input_batch[SampleBatch.ACTION_LOGP]
                entropy = tf.boolean_mask(action_dist.entropy(), mask)
                vf_preds = all_vf_preds[:-1]

                unnormalised_all_vf_preds = self.popart_module.unnormalise(all_vf_preds)
                unnormalised_vf_pred = unnormalised_all_vf_preds[:-1]
                unnormalised_next_vf_pred = unnormalised_all_vf_preds[1:]
                unnormalised_bootstrap_v = unnormalised_all_vf_preds[-1]

                rhos = tf.exp(offline_action_logp - behavior_logp)
                # no need to stop the gradient here
                clipped_rhos= tf.stop_gradient(tf.minimum(1.0, rhos, name='cs'))

                rewards = input_batch[SampleBatch.REWARD] + (input_batch["action_state_rewards"] *
                self.curriculum_module.train_config["action_state_reward_scale"])

                with tf.device('/cpu:0'):
                    vtrace_returns = compute_vtrace(
                    rewards=rewards,
                    dones=input_batch[SampleBatch.DONE],
                    values=unnormalised_vf_pred,#vf_preds,
                    next_values=unnormalised_next_vf_pred,
                    discount=self.curriculum_module.train_config["discount"],
                    clipped_rhos=clipped_rhos,
                    bootstrap_v=unnormalised_bootstrap_v,
                    mask=mask,
                    gae_lambda=self.policy_config.gae_lambda,
                    )

                # boolean mask flattens already
                unnormalised_gvs = tf.boolean_mask(vtrace_returns.gvs, mask)
                gvs = self.popart_module.normalise(unnormalised_gvs)
                unnormalised_gpi = tf.boolean_mask(vtrace_returns.gpi, mask)
                gpi = self.popart_module.normalise(unnormalised_gpi)
                clipped_rhos = tf.boolean_mask(clipped_rhos, mask)
                vf_preds = tf.boolean_mask(vf_preds, mask)


                advantages = clipped_rhos * (gpi - tf.stop_gradient(vf_preds))

                is_ratio = tf.clip_by_value(
                    tf.math.exp(behavior_logp - offline_action_logp), 0.0, 2.0
                )

                logp_ratio = tf.boolean_mask(is_ratio * tf.exp(action_logp - behavior_logp), mask)

                surrogate_loss = tf.minimum(
                    advantages * logp_ratio,
                    advantages
                    * tf.clip_by_value(
                        logp_ratio,
                        1 - self.policy_config.ppo_clip,
                        1 + self.policy_config.ppo_clip,
                    ),
                )

                policy_loss = -tf.reduce_mean(surrogate_loss)

                delta = gvs - vf_preds
                critic_loss = 0.5 * tf.reduce_mean(tf.math.square(delta))
                mean_entropy = tf.reduce_mean(entropy)

                kl_offline_to_online = tf.boolean_mask(offline_action_dist.kl(action_logits), mask)

                mean_kl = tf.reduce_mean(kl_offline_to_online)

                total_loss = critic_loss + policy_loss - mean_entropy * self.policy_config.entropy_cost + self.policy_config.ppo_kl_coeff * mean_kl

                if coaching_model is not None:
                    (imitating_logits, _), _ = self.model(
                        coaching_batch
                    )
                    (coaching_logits, _), _ = coaching_model(
                        coaching_batch
                    )
                    coaching_dist = self.model.action_dist(tf.stop_gradient(coaching_logits))
                    kl_coach_online = tf.reduce_mean(tf.boolean_mask(coaching_dist.kl(imitating_logits), mask_all))

                    # todo: imitation_loss_coeff
                    total_loss += kl_coach_online * self.curriculum_module.current_config._coaching_scale * 0.1
                else:
                    kl_coach_online = 0.


        vars = self.model.trainable_variables
        gradients = tape.gradient(total_loss, vars)
        gradients, mean_grad_norm = tf.clip_by_global_norm(gradients, self.policy_config.grad_clip)

        self.model.optimiser.apply(gradients, vars)

        mean_entropy = tf.reduce_mean(entropy)
        explained_vf = explained_variance(
            gvs,
            vf_preds
        )

        return {
            "mean_entropy": mean_entropy,
            "vf_loss": critic_loss,
            "pi_loss": policy_loss,
            "mean_grad_norm": mean_grad_norm,
            "explained_vf": explained_vf,
            "vtrace_mean": tf.reduce_mean(unnormalised_gvs),
            "vtrace_std": tf.math.reduce_std(unnormalised_gvs),
            "rhos": tf.reduce_mean(rhos),
            "batch_abs_rewards": tf.reduce_mean(tf.math.abs(tf.boolean_mask(input_batch[SampleBatch.REWARD], mask))),
            "action_state_rewards":  tf.reduce_mean(tf.boolean_mask(input_batch["action_state_rewards"], mask)),
            "offline_to_online_kl": kl_offline_to_online,
            "kl_coach_online": kl_coach_online
        }