import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.models.tf import TFModelV2
import tensorflow as tf
from ray.rllib.policy.view_requirement import ViewRequirement


class VmpoInterface(TFModelV2):
    """Extension of the standard TFModelV2 for SAC.

    To customize, do one of the following:
    - sub-class SACTFModel and override one or more of its methods.
    - Use SAC's `q_model_config` and `policy_model` keys to tweak the default model
      behaviors (e.g. fcnet_hiddens, conv_filters, etc..).
    - Use SAC's `q_model_config->custom_model` and `policy_model->custom_model` keys
      to specify your own custom Q-model(s) and policy-models, which will be
      created within this SACTFModel (see `build_policy_model` and
      `build_q_model`.

    Note: It is not recommended to override the `forward` method for SAC. This
    would lead to shared weights (between policy and Q-nets), which will then
    not be optimized by either of the critic- or actor-optimizers!

    Data flow:
        `obs` -> forward() (should stay a noop method!) -> `model_out`
        `model_out` -> get_policy_output() -> pi(actions|obs)
        `model_out`, `actions` -> get_q_values() -> Q(s, a)
        `model_out`, `actions` -> get_twin_q_values() -> Q_twin(s, a)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        config,
        existing_model=None,
        existing_inputs=None,
        eta: float = None,
        eps_eta: float = None,
        alpha: float = None,
        eps_alpha: float = None,
        statistics_lr: float = None,
        **kwargs,
    ):

        def relup(x):
            return tf.maximum(x, 1e-8)

        super().__init__(observation_space, action_space, config, existing_model, existing_inputs)

        self.eta = tf.Variable(eta, dtype=tf.float32, constraint=relup, name="eta")
        self.eps_eta = tf.Variable(eps_eta, dtype=tf.float32, trainable=False, name="eps_eta")
        self.alpha = tf.Variable(alpha, dtype=tf.float32, constraint=relup, name="alpha")
        self.eps_alpha = tf.Variable(eps_alpha, dtype=tf.float32, trainable=False, name="eps_alpha")

        self.popart_mean = tf.Variable(0., dtype=tf.float32, name="popart_mean", trainable=False)
        self.popart_moment = tf.Variable(1., dtype=tf.float32, name="popart_moment", trainable=False)
        self.popart_std = tf.Variable(0.5, dtype=tf.float32, name="popart_std", trainable=False)

        self.reward_std = tf.Variable(1., dtype=tf.float32, name="reward_std", trainable=False)
        self.reward_mean = tf.Variable(0., dtype=tf.float32, name="reward_mean", trainable=False)
        self.count = tf.Variable(0, dtype=tf.int32, name="count", trainable=False)

        self.popart_lr = statistics_lr #tf.Variable(statistics_lr, dtype=tf.float32, name="popart_lr", trainable=False)
