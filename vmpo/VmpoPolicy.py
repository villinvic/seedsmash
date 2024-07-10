import collections
import copy
from pathlib import Path
from time import time
from typing import Union, Type, List, Dict, Tuple
import logging

import numpy as np
import tree
from ray.rllib import SampleBatch, Policy
from ray.rllib.algorithms.impala import vtrace_tf
from ray.rllib.algorithms.impala.impala_tf_policy import _make_time_major
from ray.rllib.algorithms.impala.vtrace_tf import multi_log_probs_from_logits_and_actions, get_log_rhos, \
    VTraceFromLogitsReturns, VTraceReturns
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages, compute_gae_for_sample_batch

from ray.rllib.models import ModelV2, ModelCatalog
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution, Categorical
from ray.rllib.policy.dynamic_tf_policy_v2 import DynamicTFPolicyV2, TFPolicy
from ray.rllib.policy.tf_mixins import GradStatsMixin, EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.utils import override, try_import_tf, force_list
from ray.rllib.utils.tf_utils import explained_variance, make_tf_callable

import ray
import gymnasium as gym
from PIL import Image
from ray.rllib.utils.typing import TensorType, PolicyState, AgentID, PolicyID, LocalOptimizer, ModelGradients

from vmpo.VmpoInterface import VmpoInterface

logger = logging.getLogger(__name__)

tf1, tf, tfv = try_import_tf()


def custom_kl(first, other):
    a0 = first.inputs - tf.reduce_max(first.inputs, axis=1, keepdims=True)
    a1 = other.inputs - tf.reduce_max(other.inputs, axis=1, keepdims=True)
    s0 = tf.nn.softmax(a0) + 1e-8
    s1 = tf.nn.softmax(a1) + 1e-8
    return tf.reduce_sum(s0 * (tf.math.log(s0) - tf.math.log(s1)), axis=1)


class TargetNetworkMixin:
    """Assign the `update_target` method to the policy.

    The function is called every `target_network_update_freq` steps by the
    master learner.
    """

    def __init__(self):
        self._target_variables = ray.experimental.tf_utils.TensorFlowVariables(
            [], self.get_session(), self.target_model.variables()
        )

        model_vars = self.model.variables()
        target_model_vars = self.target_model.variables()

        @make_tf_callable(self.get_session())
        def update_target_fn():
            update_target_expr = []
            assert len(model_vars) == len(target_model_vars), (
                model_vars,
                target_model_vars,
            )
            for var, var_target in zip(model_vars, target_model_vars):
                update_target_expr.append(
                    var_target.assign(var)
                )
            return tf.group(*update_target_expr)

        # Hard initial update.
        self._do_update = update_target_fn
        self.update_target()  # self.config.get("tau", 1.0))

    # Support both hard and soft sync.
    def update_target(self, weights=None) -> None:
        # if weights is None:
        #    weights = self.get_weights() #self._variables.get_weights()
        # print("Update target !", message)
        self._do_update()

    # def set_weights(self, weights):
    #    DynamicTFPolicyV2.set_weights(self, weights)
    #    self.update_target(weights)


class ICMClipGradient:
    """VTrace version of gradient computation logic."""

    def __init__(self):
        """No special initialization required."""
        pass

    @override(DynamicTFPolicyV2)
    def compute_gradients_fn(
        self, optimizer: LocalOptimizer, loss: TensorType
    ) -> ModelGradients:
        # Supporting more than one loss/optimizer.
        if self.config.get("_enable_rl_module_api", False):
            # In order to access the variables for rl modules, we need to
            # use the underlying keras api model.trainable_variables.
            trainable_variables = self.model.trainable_variables
        else:
            trainable_variables = self.model.trainable_variables()
        if self.config["_tf_policy_handles_more_than_one_loss"]:
            optimizers = force_list(optimizer)
            losses = force_list(loss)

            assert len(optimizers) == len(losses), (self.learner_bound, optimizers, losses, self.optimizer())

            clipped_grads_and_vars = []
            for optim, loss_ in zip(optimizers, losses):
                grads_and_vars = optim.compute_gradients(loss_, trainable_variables)
                clipped_g_and_v = []
                for i, (g, v) in enumerate(grads_and_vars):
                    if g is not None:
                        if i == 1:
                            clipped_g_and_v.append((g, v))
                        else:
                            clipped_g, _ = tf.clip_by_global_norm(
                                [g], self.config["grad_clip"]
                            )
                            clipped_g_and_v.append((clipped_g[0], v))
                clipped_grads_and_vars.append(clipped_g_and_v)

            self.grads = [g for g_and_v in clipped_grads_and_vars for (g, v) in g_and_v]
        # Only one optimizer and and loss term.
        else:
            grads_and_vars = optimizer.compute_gradients(
                loss, self.model.trainable_variables()
            )
            grads = [g for (g, v) in grads_and_vars]
            self.grads, _ = tf.clip_by_global_norm(grads, self.config["grad_clip"])
            clipped_grads_and_vars = list(zip(self.grads, trainable_variables))

        return clipped_grads_and_vars

class ICMOptimizer:
    """Optimizer function for VTrace policies."""

    def __init__(self):
        pass

    # TODO: maybe standardize this function, so the choice of optimizers are more
    #  predictable for common algorithms.
    def optimizer(
        self,
    ) -> Union["tf.keras.optimizers.Optimizer", List["tf.keras.optimizers.Optimizer"]]:
        config = self.config
        if config["opt_type"] == "adam":
            if config["framework"] == "tf2":
                optim = tf.keras.optimizers.Adam(self.cur_lr)
                if config["_separate_vf_optimizer"]:
                    return optim, tf.keras.optimizers.Adam(config["_lr_vf"])
            else:
                optim = tf1.train.AdamOptimizer(self.cur_lr)
                if config["_separate_vf_optimizer"]:
                    return optim, tf1.train.AdamOptimizer(config["_lr_vf"])
        else:
            if config["_separate_vf_optimizer"]:
                raise ValueError(
                    "RMSProp optimizer not supported for separate"
                    "vf- and policy losses yet! Set `opt_type=adam`"
                )

            if config["framework"] == "tf2":
                optim = tf.keras.optimizers.RMSprop(
                    self.cur_lr, config["decay"], config["momentum"], config["epsilon"]
                )
                if self.learner_bound:
                    icm_optimizer = tf.keras.optimizers.Adam(1e-4, name="ICM_optim")
                    return optim, icm_optimizer

            else:
                optim = tf1.train.RMSPropOptimizer(
                    self.cur_lr, config["decay"], config["momentum"], config["epsilon"]
                )
                if self.learner_bound:
                    icm_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, name="ICM_optim")
                    return optim, icm_optimizer

        return optim


class VmpoPolicy(

    ICMClipGradient,
    ICMOptimizer,
    LearningRateSchedule,
    EntropyCoeffSchedule,
    TargetNetworkMixin,
    GradStatsMixin,
    DynamicTFPolicyV2
):
    def __init__(
            self,
            observation_space,
            action_space,
            config,
            existing_model=None,
            existing_inputs=None,
            learner_bound=False,
    ):

        self.learner_bound = learner_bound
        self.total_diff = 0

        tf1.disable_eager_execution()

        ICMClipGradient.__init__(self)
        ICMOptimizer.__init__(self)

        # Initialize base class.
        DynamicTFPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            existing_inputs=existing_inputs,
            existing_model=existing_model,
        )

        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        GradStatsMixin.__init__(self)

        # Note: this is a bit ugly, but loss and optimizer initialization must
        # happen after all the MixIns are initialized.

        self.maybe_initialize_optimizer_and_loss()

        if self.learner_bound:
            TargetNetworkMixin.__init__(self)
        else:
            self._do_update = lambda: print("No-op")
            self._target_variables = self._variables

    def variables(self):
        """Return the list of all savable variables for this policy."""
        if self.model is None:
            raise NotImplementedError("No `self.model` to get variables for!")

        return self.model.variables()

    @override(DynamicTFPolicyV2)
    def make_model(self) -> ModelV2:

        model_config= {

            "learner_bound": self.learner_bound,
            **self.config["model"]
        }

        self.model = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=self.action_space.n,
            model_config=model_config,
            framework=self.config["framework"],
            model_interface=VmpoInterface,
            name="model",
            eta=self.config["eta"],
            eps_eta=self.config["eps_eta"],
            alpha=self.config["alpha"],
            eps_alpha=self.config["eps_alpha"],
            statistics_lr=self.config["statistics_lr"]
        )

        # Do not maintain copies on all workers as this takes room in the memory
        if self.learner_bound:

            # Create an exact copy of the model and store it in `policy.target_model`.

            self.target_model = ModelCatalog.get_model_v2(
                obs_space=self.observation_space,
                action_space=self.action_space,
                num_outputs=self.action_space.n,
                model_config=model_config,
                framework=self.config["framework"],
                model_interface=VmpoInterface,
                name="target_model",
                eta=self.config["eta"],
                eps_eta=self.config["eps_eta"],
                alpha=self.config["alpha"],
                eps_alpha=self.config["eps_alpha"],
                statistics_lr=self.config["statistics_lr"]
            )
        else:
            self.target_model = self.model

        return self.model

    @override(DynamicTFPolicyV2)
    def loss(
            self,
            model: Union[ModelV2, "tf.keras.Model"],
            dist_class: Type[TFActionDistribution],
            train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:

        popart_mean = tf.convert_to_tensor(self.model.popart_mean, dtype=tf.float32)
        popart_std = tf.convert_to_tensor(self.model.popart_std, dtype=tf.float32)

        model_out, _ = self.model(train_batch)
        target_model_out, _ = self.target_model(train_batch)

        action_dist = dist_class(model_out, model)

        if isinstance(self.action_space, gym.spaces.Discrete):
            is_multidiscrete = False
            output_hidden_shape = [self.action_space.n]
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            is_multidiscrete = True
            output_hidden_shape = self.action_space.nvec.astype(np.int32)
        else:
            is_multidiscrete = False
            output_hidden_shape = 1

        def make_time_major(*args, **kw):
            return _make_time_major(
                self, train_batch.get(SampleBatch.SEQ_LENS), *args, **kw
            )

        # behaviour_advantage = train_batch[Postprocessing.ADVANTAGES]
        # behaviour_value_targets = train_batch[Postprocessing.VALUE_TARGETS]

        actions = train_batch[SampleBatch.ACTIONS]
        dones = train_batch[SampleBatch.TERMINATEDS]
        rewards = train_batch[SampleBatch.REWARDS]

        # ICM ####################################

        if self.learner_bound:
            # state_prediction_loss = self.model.state_prediction_loss()
            # action_prediction_loss = self.model.action_prediction_loss()
            #
            # self.mean_state_prediction_loss = tf.reduce_mean(state_prediction_loss)
            # self.max_action_prediction_loss = tf.reduce_max(action_prediction_loss)
            # self.min_action_prediction_loss = tf.reduce_min(action_prediction_loss)
            # self.mean_action_prediction_loss = tf.reduce_mean(action_prediction_loss)
            #
            # intrinsic_rewards = state_prediction_loss * self.model.curiosity_reward_scale
            # rewards = rewards + tf.stop_gradient(intrinsic_rewards)
            #
            # self.mean_intrinsic_rewards = tf.reduce_mean(intrinsic_rewards)
            # self.min_intrinsic_rewards = tf.reduce_min(intrinsic_rewards)
            # self.max_intrinsic_rewards = tf.reduce_max(intrinsic_rewards)
            #
            # icm_loss = (self.mean_action_prediction_loss * (1. - self.model.icm_beta)
            #              + self.mean_state_prediction_loss * self.model.icm_beta)
            #
            # self.mean_icm_loss = icm_loss / self.model.icm_lambda

            # Disagreement

            state_prediction_loss = self.model.state_prediction_loss()
            #action_prediction_loss = self.model.action_prediction_loss()
            intrinsic_rewards = tf.stop_gradient(self.model.compute_intrinsic_rewards()) * self.model.intrinsic_reward_scale

            # Make intrinsic rewards of same norm as rewards ?
            intrinsic_rewards = intrinsic_rewards * tf.maximum(tf.reduce_mean(rewards) / tf.reduce_mean(intrinsic_rewards), 1.)

            rewards = rewards * (1. - self.model.intrinsic_reward_ratio) + intrinsic_rewards * self.model.intrinsic_reward_ratio

            self.max_state_prediction_loss = tf.reduce_max(state_prediction_loss)
            self.min_state_prediction_loss = tf.reduce_min(state_prediction_loss)
            self.mean_state_prediction_loss = tf.reduce_mean(state_prediction_loss)

            # self.max_action_prediction_loss = tf.reduce_max(action_prediction_loss)
            # self.min_action_prediction_loss = tf.reduce_min(action_prediction_loss)
            # self.mean_action_prediction_loss = tf.reduce_mean(action_prediction_loss)

            self.mean_intrinsic_rewards = tf.reduce_mean(intrinsic_rewards)
            self.min_intrinsic_rewards = tf.reduce_min(intrinsic_rewards)
            self.max_intrinsic_rewards = tf.reduce_max(intrinsic_rewards)

        else:

            self.mean_intrinsic_rewards = tf.zeros((1,), dtype=tf.float32)
            self.mean_icm_loss = tf.zeros((1,), dtype=tf.float32)
            self.mean_state_prediction_loss = tf.zeros((1,), dtype=tf.float32)
            self.max_state_prediction_loss = tf.zeros((1,), dtype=tf.float32)
            self.min_state_prediction_loss = tf.zeros((1,), dtype=tf.float32)
            self.mean_action_prediction_loss = tf.zeros((1,), dtype=tf.float32)
            self.mean_intrinsic_rewards = tf.zeros((1,), dtype=tf.float32)
            self.min_intrinsic_rewards = tf.zeros((1,), dtype=tf.float32)
            self.max_intrinsic_rewards = tf.zeros((1,), dtype=tf.float32)
            self.max_action_prediction_loss = tf.zeros((1,), dtype=tf.float32)
            self.min_action_prediction_loss = tf.zeros((1,), dtype=tf.float32)

        ##########################################

        self.batch_reward_std = tf.maximum(tf.math.reduce_std(rewards), 1e-2)
        self.batch_reward_mean = tf.math.reduce_mean(rewards)

        behaviour_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]

        old_policy_behaviour_logits = tf.stop_gradient(target_model_out)
        old_policy_action_dist = dist_class(old_policy_behaviour_logits, model)

        behaviour_action_dist = dist_class(behaviour_logits, model)

        unpacked_behaviour_logits = tf.split(
            behaviour_logits, output_hidden_shape, axis=1
        )
        unpacked_old_policy_behaviour_logits = tf.split(
            old_policy_behaviour_logits, output_hidden_shape, axis=1
        )

        values = self.model.value_function()
        values_time_major = make_time_major(values)

        unnormalized_values_time_major = values_time_major * popart_std + popart_mean

        if self.is_recurrent():
            max_seq_len = tf.reduce_max(train_batch[SampleBatch.SEQ_LENS])
            mask = tf.sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
            mask = tf.reshape(mask, [-1])
            mask = make_time_major(mask)
        else:
            mask = tf.ones_like(train_batch[SampleBatch.REWARDS])
            mask = make_time_major(mask)

        # Prepare actions for loss
        loss_actions = (
            actions if is_multidiscrete else tf.expand_dims(actions, axis=1)
        )

        # Inputs are reshaped from [B * T] => [(T|T-1), B] for V-trace calc.
        if self.config["vtrace"]:
            with tf.device("/cpu:0"):
                vtrace_returns = multi_from_logits(
                    behaviour_policy_logits=make_time_major(
                        unpacked_behaviour_logits
                    ),
                    target_policy_logits=make_time_major(
                        unpacked_old_policy_behaviour_logits
                    ),
                    actions=tf.unstack(
                        make_time_major(loss_actions), axis=2
                    ),
                    discounts=tf.cast(
                        ~make_time_major(
                            tf.cast(dones, tf.bool)
                        ),
                        tf.float32,
                    )
                              * self.config["gamma"],
                    rewards=make_time_major(rewards),
                    values=unnormalized_values_time_major,
                    bootstrap_value=unnormalized_values_time_major[-1],
                    normalized_values=values_time_major,
                    dist_class=Categorical if is_multidiscrete else dist_class,
                    model=model,
                    popart_mean=popart_mean,
                    popart_std=popart_std,
                    clip_rho_threshold=tf.cast(
                        self.config["vtrace_clip_rho_threshold"], tf.float32
                    ),
                    clip_pg_rho_threshold=tf.cast(
                        self.config["vtrace_clip_pg_rho_threshold"], tf.float32
                    ),
                )
        else:
            raise NotImplementedError

        self.normalization_scale = tf.math.maximum(tf.math.maximum(self.model.reward_std, self.batch_reward_std), 1e-2)

        delta = (values_time_major - vtrace_returns.vs)  # / self.normalization_scale

        # importance weighting
        rhos = tf.minimum(1., tf.math.exp(vtrace_returns.log_rhos))

        rhos = tf.reshape(rhos, [-1])
        advantages = tf.reshape(vtrace_returns.pg_advantages, [-1])  # / self.normalization_scale

        top_half_advantages, top_half_indices = tf.math.top_k(advantages, k=tf.size(advantages) // 2, sorted=False)

        # top_half_indices = tf.argsort(
        #     vtrace_returns.pg_advantages,
        #     axis=0,
        #     direction='DESCENDING'
        # )[:vtrace_returns.pg_advantages.shape[0]//2]

        top_half_actions_logp = tf.gather(tf.reshape(make_time_major(
            action_dist.logp(actions)
        ), [-1]), top_half_indices)

        top_half_advantages = tf.minimum(tf.math.log(1e32 * tf.stop_gradient(model.eta)), top_half_advantages)
        top_half_rhos = tf.gather(rhos, top_half_indices)
        top_half_mask = tf.gather(tf.reshape(mask, [-1]), top_half_indices)
        # top_half_advantages = tf.minimum(tf.math.log(1e32)*tf.stop_gradient(model.eta), tf.gather(vtrace_returns.pg_advantages, top_half_indices))
        # top_half_mask = tf.gather(mask, top_half_indices)

        # flat_actions_logp = tf.gather(tf.reshape(action_dist.logp(), (flat_shape,)), topk_indices, batch_dims=-1)
        # flat_mask = tf.gather(tf.reshape(mask, (flat_shape,)), topk_indices, batch_dims=-1)
        # flat_rhos = tf.reshape(rhos, (flat_shape,))
        kl = custom_kl(old_policy_action_dist, action_dist)
        self.target_to_behavior_kl = tf.reduce_mean(custom_kl(old_policy_action_dist, behaviour_action_dist))
        self.online_to_behavior_kl = tf.reduce_mean(custom_kl(action_dist, behaviour_action_dist))
        self.kl = tf.reduce_mean(kl)
        self.max_kl = tf.reduce_max(kl)

        # masked_pi_loss = tf.boolean_mask(
        #    actions_logp * self.vtrace_returns.pg_advantages, valid_mask
        # )
        masked_entropy = tf.boolean_mask(make_time_major(action_dist.multi_entropy()), mask)
        target_masked_entropy = tf.boolean_mask(make_time_major(old_policy_action_dist.multi_entropy()), mask)
        behaviour_masked_entropy = tf.boolean_mask(make_time_major(behaviour_action_dist.multi_entropy()), mask)
        self.mean_entropy = tf.reduce_mean(masked_entropy)
        self.target_mean_entropy = tf.reduce_mean(target_masked_entropy)
        self.behaviour_masked_entropy = tf.reduce_mean(behaviour_masked_entropy)

        self._value_targets = vtrace_returns.vs
        self._unormalized_value_targets = vtrace_returns.unormalized_vs

        self.vf_loss = 0.5 * tf.reduce_mean(tf.boolean_mask(tf.math.square(delta), mask))
        self.entropy_loss = self.mean_entropy * self.entropy_coeff
        self.policy_loss = policy_loss(top_half_advantages, tf.stop_gradient(model.eta), top_half_actions_logp,
                                       top_half_rhos, top_half_mask)
        self.temperate_loss = temp_loss(top_half_advantages, model.eta, model.eps_eta, top_half_rhos, top_half_mask)
        self.trust_region_loss = trust_region_loss(model.alpha, model.eps_alpha, self.kl)

        self.total_loss = (
                self.vf_loss + self.policy_loss - self.entropy_loss
                + self.trust_region_loss + self.temperate_loss
        )

        self.new_mean = vtrace_returns.new_mean
        self.new_moment = vtrace_returns.new_moment

        if self.learner_bound:
            return self.total_loss, (
                    self.mean_state_prediction_loss * (1. - self.model.forward_loss_ratio)
                    #+ self.mean_action_prediction_loss * self.model.forward_loss_ratio
            )
        else:
            return self.total_loss

    @override(DynamicTFPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:

        values_batched = _make_time_major(
            self,
            train_batch.get(SampleBatch.SEQ_LENS),
            self.model.value_function(),
        )

        return {
            "cur_lr"               : tf.cast(self.cur_lr, tf.float64),
            "policy_loss"          : self.policy_loss,
            "entropy"              : self.mean_entropy,
            "target_entropy"       : self.target_mean_entropy,
            "behavior_entropy"     : self.behaviour_masked_entropy,
            "entropy_coeff"        : tf.cast(self.entropy_coeff, tf.float64),
            "var_gnorm"            : tf.linalg.global_norm(self.model.trainable_variables()),
            "vf_loss"              : self.vf_loss,
            "vf_explained_var"     : explained_variance(
                tf.reshape(self._value_targets, [-1]),
                tf.reshape(values_batched, [-1]),
            ),
            "mean_predicted_value" : tf.reduce_mean(values_batched),
            "mean_normalized_value": tf.reduce_mean(self._value_targets),
            "eta"                  : self.model.eta,
            "alpha"                : self.model.alpha,
            "temperature_loss"     : self.temperate_loss,
            "trust_region_loss"    : self.trust_region_loss,
            "kl"                   : self.kl,
            "max_kl"               : self.max_kl,
            "target_to_behavior_kl": self.target_to_behavior_kl,
            "online_to_behavior_kl": self.online_to_behavior_kl,
            "eps_eta"              : self.model.eps_eta,
            "eps_alpha"            : self.model.eps_alpha,
            "mean_vs"              : tf.reduce_mean(self._unormalized_value_targets),
            "values_mean"          : self.new_mean,
            "values_moment"        : self.new_moment,
            "batch_reward_std"     : self.batch_reward_std,
            "batch_reward_mean"    : self.batch_reward_mean,
            "baseline_popart_std"  : self.model.popart_std,
            "baseline_popart_mean" : self.model.popart_mean,
            "normalization_scale"  : self.normalization_scale,

            # ICM
            "curiosity/state_prediction_loss_mean": self.mean_state_prediction_loss,
            "curiosity/state_prediction_loss_max": self.max_state_prediction_loss,
            "curiosity/state_prediction_loss_min": self.min_state_prediction_loss,

            # "curiosity/action_prediction_loss": self.mean_action_prediction_loss,
            # "curiosity/action_prediction_loss_min": self.min_action_prediction_loss,
            # "curiosity/action_prediction_loss_max": self.max_action_prediction_loss,
            # "curiosity/total_loss": self.mean_icm_loss,
            "curiosity/intrinsic_rewards_mean": self.mean_intrinsic_rewards,
            "curiosity/intrinsic_rewards_max": self.max_intrinsic_rewards,
            "curiosity/intrinsic_rewards_min": self.min_intrinsic_rewards,

        }

    @override(DynamicTFPolicyV2)
    def get_batch_divisibility_req(self) -> int:
        return self.config["rollout_fragment_length"]

    @override(DynamicTFPolicyV2)
    def set_state(self, state: PolicyState) -> None:

        if not self.learner_bound:
            state = state.copy()
            state.pop("_optimizer_variables")

            weights = state["weights"]
            # We do not care about value related weights
            keys = list(weights.keys())
            for k in keys:
                if "value" in k or "ICM" in k:
                    weights.pop(k)

        # state["weights"] = weights

        super().set_state(state)

    def get_state(self):
        state = super().get_state()
        return state

    def update_statistics(self, policy_id, results):

        if results:

            # Get the weights of the value layer
            w = super().get_weights()
            scope = list(w.keys())[0].split("/")[0]

            kernel_weights = w[f"{scope}/value_out/kernel"]
            biases_weights = w[f"{scope}/value_out/bias"]

            old_mean = w[f"{scope}/popart_mean"]
            old_std = w[f"{scope}/popart_std"]
            old_moment = w[f"{scope}/popart_moment"]
            new_mean = results['learner_stats']["values_mean"]
            new_moment = results['learner_stats']["values_moment"]
            n = w[f"{scope}/count"]

            lr = self.model.popart_lr
            next_mean = old_mean * (1. - lr) + new_mean * lr
            next_moment = (1 - lr) * old_moment + lr * new_moment
            next_std = np.sqrt(next_moment - np.square(next_mean))

            # TODO : aux only exists if we use centralized critic
            super().set_weights({
                f"{scope}/value_out/kernel": kernel_weights * old_std / next_std,
                f"{scope}/value_out/bias"  : (biases_weights * old_std + old_mean - next_mean) / next_std,
                f"{scope}/popart_mean"     : next_mean,
                f"{scope}/popart_std"      : next_std,
                f"{scope}/popart_moment"   : next_moment,
                f"{scope}/count"           : n + 1,

            })

    def get_weights(self, online=False) -> Union[Dict[str, TensorType], List[TensorType]]:
        # TODO : care with how we restore policies
        # TODO : VICTOR : send in online always (already too off policy)

        return super().get_weights()
        # if online:
        #     return super().get_weights()
        # else:
        #     return {
        #         k: v for k, v in zip(super().get_weights().keys(), self._target_variables.get_weights().values())
        #         #if ("value" not in k or "ICM" not in k)
        #     }


    @override(TFPolicy)
    def copy(self, existing_inputs: List[Tuple[str, "tf1.placeholder"]]) -> TFPolicy:
        """Creates a copy of self using existing input placeholders."""

        flat_loss_inputs = tree.flatten(self._loss_input_dict)
        flat_loss_inputs_no_rnn = tree.flatten(self._loss_input_dict_no_rnn)

        # Note that there might be RNN state inputs at the end of the list
        if len(flat_loss_inputs) != len(existing_inputs):
            raise ValueError(
                "Tensor list mismatch",
                self._loss_input_dict,
                self._state_inputs,
                existing_inputs,
            )
        for i, v in enumerate(flat_loss_inputs_no_rnn):
            if v.shape.as_list() != existing_inputs[i].shape.as_list():
                raise ValueError(
                    "Tensor shape mismatch", i, v.shape, existing_inputs[i].shape
                )
        # By convention, the loss inputs are followed by state inputs and then
        # the seq len tensor.
        rnn_inputs = []
        for i in range(len(self._state_inputs)):
            rnn_inputs.append(
                (
                    "state_in_{}".format(i),
                    existing_inputs[len(flat_loss_inputs_no_rnn) + i],
                )
            )
        if rnn_inputs:
            rnn_inputs.append((SampleBatch.SEQ_LENS, existing_inputs[-1]))
        existing_inputs_unflattened = tree.unflatten_as(
            self._loss_input_dict_no_rnn,
            existing_inputs[: len(flat_loss_inputs_no_rnn)],
        )
        input_dict = collections.OrderedDict(
            [("is_exploring", self._is_exploring), ("timestep", self._timestep)]
            + [
                (k, existing_inputs_unflattened[k])
                for i, k in enumerate(self._loss_input_dict_no_rnn.keys())
            ]
            + rnn_inputs
        )

        instance = self.__class__(
            self.observation_space,
            self.action_space,
            self.config,
            existing_inputs=input_dict,
            existing_model=[
                self.model,
                # Deprecated: Target models should all reside under
                # `policy.target_model` now.
                ("target_q_model", getattr(self, "target_q_model", None)),
                ("target_model", getattr(self, "target_model", None)),
            ],
        )
        instance.learner_bound = self.learner_bound

        instance._loss_input_dict = input_dict
        losses = instance._do_loss_init(SampleBatch(input_dict))
        loss_inputs = [
            (k, existing_inputs_unflattened[k])
            for i, k in enumerate(self._loss_input_dict_no_rnn.keys())
        ]

        TFPolicy._initialize_loss(instance, losses, loss_inputs)
        instance._stats_fetches.update(
            instance.grad_stats_fn(input_dict, instance._grads)
        )
        return instance


def temp_loss(advantage, eta, eps_eta, rhos, mask):
    return eta * (eps_eta + tf.math.log(
        1e-8 + tf.reduce_mean(tf.boolean_mask(rhos * tf.math.exp(advantage / eta), mask))
    ))


def trust_region_loss(alpha, eps_alpha, kl):
    return alpha * (eps_alpha - tf.stop_gradient(kl)) + tf.stop_gradient(alpha) * kl


def policy_loss(
        advantage,
        eta,
        logp,
        rho,
        mask
):
    return - tf.reduce_sum(tf.boolean_mask(
        (rho * logp * tf.math.exp(advantage / eta)) / (1e-8 + tf.reduce_sum(rho * tf.math.exp(advantage / eta))), mask))


VTraceFromLogitsReturnsNormalized = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behaviour_action_log_probs",
        "target_action_log_probs",
        "unormalized_vs",
        "new_mean",
        "new_moment"
    ],
)


def multi_from_logits(
        behaviour_policy_logits,
        target_policy_logits,
        actions,
        discounts,
        rewards,
        values,
        bootstrap_value,
        normalized_values,
        dist_class,
        model,
        popart_mean,
        popart_std,
        behaviour_action_log_probs=None,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
        name="vtrace_from_logits",
):
    r"""V-trace for softmax policies.

    Calculates V-trace actor critic targets for softmax polices as described in

    "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures"
    by Espeholt, Soyer, Munos et al.

    Target policy refers to the policy we are interested in improving and
    behaviour policy refers to the policy that generated the given
    rewards and actions.

    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size and
    ACTION_SPACE refers to the list of numbers each representing a number of
    actions.

    Args:
      behaviour_policy_logits: A list with length of ACTION_SPACE of float32
        tensors of shapes
        [T, B, ACTION_SPACE[0]],
        ...,
        [T, B, ACTION_SPACE[-1]]
        with un-normalized log-probabilities parameterizing the softmax behaviour
        policy.
      target_policy_logits: A list with length of ACTION_SPACE of float32
        tensors of shapes
        [T, B, ACTION_SPACE[0]],
        ...,
        [T, B, ACTION_SPACE[-1]]
        with un-normalized log-probabilities parameterizing the softmax target
        policy.
      actions: A list with length of ACTION_SPACE of
        tensors of shapes
        [T, B, ...],
        ...,
        [T, B, ...]
        with actions sampled from the behaviour policy.
      discounts: A float32 tensor of shape [T, B] with the discount encountered
        when following the behaviour policy.
      rewards: A float32 tensor of shape [T, B] with the rewards generated by
        following the behaviour policy.
      values: A float32 tensor of shape [T, B] with the value function estimates
        wrt. the target policy.
      bootstrap_value: A float32 of shape [B] with the value function estimate at
        time T.
      dist_class: action distribution class for the logits.
      model: backing ModelV2 instance
      behaviour_action_log_probs: precalculated values of the behaviour actions
      clip_rho_threshold: A scalar float32 tensor with the clipping threshold for
        importance weights (rho) when calculating the baseline targets (vs).
        rho^bar in the paper.
      clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold
        on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)).
      name: The name scope that all V-trace operations will be created in.

    Returns:
      A `VTraceFromLogitsReturns` namedtuple with the following fields:
        vs: A float32 tensor of shape [T, B]. Can be used as target to train a
            baseline (V(x_t) - vs_t)^2.
        pg_advantages: A float 32 tensor of shape [T, B]. Can be used as an
          estimate of the advantage in the calculation of policy gradients.
        log_rhos: A float32 tensor of shape [T, B] containing the log importance
          sampling weights (log rhos).
        behaviour_action_log_probs: A float32 tensor of shape [T, B] containing
          behaviour policy action log probabilities (log \mu(a_t)).
        target_action_log_probs: A float32 tensor of shape [T, B] containing
          target policy action probabilities (log \pi(a_t)).
    """

    for i in range(len(behaviour_policy_logits)):
        behaviour_policy_logits[i] = tf.convert_to_tensor(
            behaviour_policy_logits[i], dtype=tf.float32
        )
        target_policy_logits[i] = tf.convert_to_tensor(
            target_policy_logits[i], dtype=tf.float32
        )

        # Make sure tensor ranks are as expected.
        # The rest will be checked by from_action_log_probs.
        behaviour_policy_logits[i].shape.assert_has_rank(3)
        target_policy_logits[i].shape.assert_has_rank(3)

    target_action_log_probs = multi_log_probs_from_logits_and_actions(
        target_policy_logits, actions, dist_class, model
    )

    if len(behaviour_policy_logits) > 1 or behaviour_action_log_probs is None:
        # can't use precalculated values, recompute them. Note that
        # recomputing won't work well for autoregressive action dists
        # which may have variables not captured by 'logits'
        behaviour_action_log_probs = multi_log_probs_from_logits_and_actions(
            behaviour_policy_logits, actions, dist_class, model
        )

    log_rhos = get_log_rhos(target_action_log_probs, behaviour_action_log_probs)

    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        normalized_values=normalized_values,
        model=model,
        popart_mean=popart_mean,
        popart_std=popart_std,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )

    return VTraceFromLogitsReturnsNormalized(
        log_rhos=log_rhos,
        behaviour_action_log_probs=behaviour_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict()
    )


VTraceReturnsNormalized = collections.namedtuple("VTraceReturnsNormalized",
                                                 "vs pg_advantages unormalized_vs new_mean new_moment")


def from_importance_weights(
        log_rhos,
        discounts,
        rewards,
        values,
        bootstrap_value,
        normalized_values,
        model,
        popart_mean,
        popart_std,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,

        name="vtrace_from_importance_weights",
):
    r"""V-trace from log importance weights.

    Calculates V-trace actor critic targets as described in

    "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures"
    by Espeholt, Soyer, Munos et al.

    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size. This code
    also supports the case where all tensors have the same number of additional
    dimensions, e.g., `rewards` is [T, B, C], `values` is [T, B, C],
    `bootstrap_value` is [B, C].

    Args:
      log_rhos: A float32 tensor of shape [T, B] representing the
        log importance sampling weights, i.e.
        log(target_policy(a) / behaviour_policy(a)). V-trace performs operations
        on rhos in log-space for numerical stability.
      discounts: A float32 tensor of shape [T, B] with discounts encountered when
        following the behaviour policy.
      rewards: A float32 tensor of shape [T, B] containing rewards generated by
        following the behaviour policy.
      values: A float32 tensor of shape [T, B] with the value function estimates
        wrt. the target policy.
      bootstrap_value: A float32 of shape [B] with the value function estimate at
        time T.
      clip_rho_threshold: A scalar float32 tensor with the clipping threshold for
        importance weights (rho) when calculating the baseline targets (vs).
        rho^bar in the paper. If None, no clipping is applied.
      clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold
        on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)). If
        None, no clipping is applied.
      name: The name scope that all V-trace operations will be created in.

    Returns:
      A VTraceReturns namedtuple (vs, pg_advantages) where:
        vs: A float32 tensor of shape [T, B]. Can be used as target to
          train a baseline (V(x_t) - vs_t)^2.
        pg_advantages: A float32 tensor of shape [T, B]. Can be used as the
          advantage in the calculation of policy gradients.
    """
    log_rhos = tf.convert_to_tensor(log_rhos, dtype=tf.float32)
    discounts = tf.convert_to_tensor(discounts, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    values = tf.convert_to_tensor(values, dtype=tf.float32)
    bootstrap_value = tf.convert_to_tensor(bootstrap_value, dtype=tf.float32)
    if clip_rho_threshold is not None:
        clip_rho_threshold = tf.convert_to_tensor(clip_rho_threshold, dtype=tf.float32)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold = tf.convert_to_tensor(
            clip_pg_rho_threshold, dtype=tf.float32
        )

    # Make sure tensor ranks are consistent.
    rho_rank = log_rhos.shape.ndims  # Usually 2.
    values.shape.assert_has_rank(rho_rank)
    bootstrap_value.shape.assert_has_rank(rho_rank - 1)
    discounts.shape.assert_has_rank(rho_rank)
    rewards.shape.assert_has_rank(rho_rank)
    if clip_rho_threshold is not None:
        clip_rho_threshold.shape.assert_has_rank(0)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold.shape.assert_has_rank(0)

    rhos = tf.math.exp(log_rhos)
    if clip_rho_threshold is not None:
        clipped_rhos = tf.minimum(clip_rho_threshold, rhos, name="clipped_rhos")
    else:
        clipped_rhos = rhos

    cs = tf.minimum(1.0, rhos, name="cs")
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = tf.concat(
        [values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0
    )
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    # All sequences are reversed, computation starts from the back.
    sequences = (
        tf.reverse(discounts, axis=[0]),
        tf.reverse(cs, axis=[0]),
        tf.reverse(deltas, axis=[0]),
    )

    # V-trace vs are calculated through a scan from the back to the
    # beginning of the given trajectory.
    def scanfunc(acc, sequence_item):
        discount_t, c_t, delta_t = sequence_item
        return delta_t + discount_t * c_t * acc

    initial_values = tf.zeros_like(bootstrap_value)
    vs_minus_v_xs = tf.nest.map_structure(
        tf.stop_gradient,
        tf.scan(
            fn=scanfunc,
            elems=sequences,
            initializer=initial_values,
            parallel_iterations=1,
            name="scan",
        ),
    )
    # Reverse the results back to original order.
    vs_minus_v_xs = tf.reverse(vs_minus_v_xs, [0], name="vs_minus_v_xs")

    # Add V(x_s) to get v_s.
    vs = tf.add(vs_minus_v_xs, values, name="vs")

    vs_t_plus_1 = tf.concat([vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)

    # mean_vs = tf.reduce_mean(vs_t_plus_1)

    # TODO : too slow
    # def update_stats_step(mean_and_second_moment, gvt):
    #     _mean, _moment = mean_and_second_moment
    #     _new_mean = (1. - model.popart_lr) * _mean + model.popart_lr * tf.reduce_mean(gvt)
    #     _new_moment = (1. - model.popart_lr) * _moment + model.popart_lr * tf.math.square(tf.reduce_mean(gvt))
    #
    #     return _new_mean, _new_moment
    #
    # def update_stats_batch(mm, gvt):
    #     return tf.foldl(update_stats_step, gvt, initializer=mm)

    # new_mean, new_moment = tf.foldl(update_stats_batch, vs, initializer=(popart_mean, popart_moment))

    new_mean = tf.reduce_mean(vs)
    # new_std = tf.clip_by_value(tf.math.reduce_std(vs), 1e-2, 1e6)
    new_moment = tf.reduce_mean(tf.math.square(vs))

    # Advantage for policy gradient.
    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = tf.minimum(
            clip_pg_rho_threshold, rhos, name="clipped_pg_rhos"
        )
    else:
        clipped_pg_rhos = rhos

    # std = tf.clip_by_value(tf.math.sqrt(popart_moment - tf.math.square(popart_mean)), 1e-2, 1e6)

    # pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1  - values)
    normalized_vs = (vs - popart_mean) / popart_std
    pg_advantages = clipped_pg_rhos * (
                (rewards + discounts * vs_t_plus_1 - popart_mean) / popart_std - normalized_values)

    # Make sure no gradients backpropagated through the returned values.
    return VTraceReturnsNormalized(
        vs=tf.stop_gradient(normalized_vs), pg_advantages=tf.stop_gradient(pg_advantages),
        unormalized_vs=tf.stop_gradient(vs),
        new_mean=new_mean, new_moment=new_moment,  # new_std=new_std
    )