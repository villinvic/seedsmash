import sonnet as snt
import tensorflow as tf
import tree
from gymnasium.spaces import Discrete
from polaris.experience import SampleBatch
from polaris.models import BaseModel

tf.compat.v1.enable_eager_execution()

import numpy as np
from polaris.models.utils import CategoricalDistribution


class SmallFCCatVf(BaseModel):

    is_recurrent = False

    def initialise(self):
        T = 5
        B = 3
        x = self.observation_space.sample()
        dummy_obs = tree.map_structure(
            lambda v: np.zeros_like(v, shape=(T, B) + v.shape),
            x
        )
        dummy_reward = np.zeros((T, B), dtype=np.float32)
        dummy_actions = np.zeros((T, B), dtype=np.int32)

        dummy_state = self.get_initial_state()
        #states = [np.zeros_like(dummy_state, shape=(B,) + d.shape) for d in dummy_state]
        seq_lens = np.ones((B,), dtype=np.int32) * T

        @tf.function
        def run(d):
            self(
                d
            )

        run({
            SampleBatch.OBS        : dummy_obs,
            SampleBatch.PREV_ACTION: dummy_actions,
            SampleBatch.PREV_REWARD: dummy_reward,
            #SampleBatch.STATE      : states,
            SampleBatch.SEQ_LENS   : seq_lens,
        })

    def __init__(
            self,
            observation_space,
            action_space: Discrete,
            config,
    ):
        super(SmallFCCatVf, self).__init__(
            name="SmallFCCatVf",
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )

        self.num_outputs = action_space.n
        self.size = 128+32
        self.character_embedding_size = 5
        self.action_state_embedding_size = 32
        self.joint_embedding_size = 64
        self.last_action_embedding_size = 16

        self.num_bins = 64
        self.v_min, self.v_max = (-16, 16)
        self.bin_width = (self.v_max - self.v_min) / self.num_bins
        self.support = tf.cast(tf.expand_dims(tf.linspace(self.v_min, self.v_max, self.num_bins + 1), axis=0), tf.float32)
        self.centers = (self.support[:, :-1] + self.support[:, 1:]) / 2.
        smoothing_ratio = 0.75
        sigma = smoothing_ratio * self.bin_width
        self.sqrt_two_sigma = tf.math.sqrt(2.) * sigma

        self.optimiser = snt.optimizers.RMSProp(
            learning_rate=config.lr,
            decay=config.rms_prop_rho,
            momentum=0.0,
            epsilon=config.rms_prop_epsilon,
            centered=False,
            name="rmsprop"
        )
        self.action_dist = CategoricalDistribution

        self._mlp = snt.nets.MLP(
            output_sizes=self.config.fc_dims,
            activate_final=True,
            name="MLP"
        )

        self._pi_out = snt.Linear(
            output_size=self.num_outputs,
            name="action_logits"
        )
        self._value_out = snt.nets.MLP(
            output_sizes=[self.num_bins],
            activate_final=False,
            name="values"
        )

        self.n_chars = int(observation_space["categorical"]["character1"].high) + 1
        self.n_action_states = int(observation_space["categorical"]["action1"].high) + 1

        self.last_action_embed = snt.Embed(
            vocab_size=self.num_outputs,
            embed_dim=self.last_action_embedding_size,
            densify_gradients=True,
            name="last_action_embed"
        )
        self.action_state_embed = snt.Embed(
            vocab_size=self.n_action_states,
            embed_dim=self.action_state_embedding_size,
            densify_gradients=True,
            name="action_state_embed"
        )
        self.opp_char_embed = snt.Embed(
            vocab_size=self.n_chars,
            embed_dim=self.character_embedding_size,
            densify_gradients=True,
            name = "opp_char_embed"
        )
        self.opp_char_state_joint_embed = snt.Embed(
            vocab_size=self.n_chars * self.n_action_states,
            embed_dim=self.joint_embedding_size,
            densify_gradients=True,
            name="opp_char_state_joint_embed"
        )

        self.post_embedding_concat = tf.keras.layers.Concatenate(axis=-1, name="post_embedding_concat")

    def forward(self,
            *,
            obs,
            prev_action,
            prev_reward,
            #state,
            seq_lens,
            single_obs=False,
            **kwargs
    ):

        if single_obs:

            continuous_inputs = [tf.expand_dims(v, axis=0) for k, v in obs["continuous"].items()]

            binary_inputs = [tf.cast(tf.expand_dims(v, axis=0), dtype=tf.float32, name=k) for k, v in obs["binary"].items()]

            categorical_inputs = obs["categorical"]

            categorical_one_hots = [
                tf.one_hot(tf.cast(tensor, tf.int32), depth=tf.cast(space.high[0], tf.int32) + 1, dtype=tf.float32,
                           name=name)
                for tensor, (name, space) in
                zip(categorical_inputs.values(), self.observation_space["categorical"].items())
                if name not in ["character1", "action1", "action2", "character2"]]  # "action1", "action2", "character2"

            #last_action_one_hot = tf.one_hot(tf.cast(tf.expand_dims(prev_action, axis=0), tf.int32),
            #                                 depth=self.num_outputs, dtype=tf.float32, name="prev_action_one_hot")

            action_state_self = categorical_inputs["action1"]
            action_state_opp = categorical_inputs["action2"]
            opp_char_input = categorical_inputs["character2"]

            # last_action_one_hot = tf.one_hot(tf.cast(prev_action, tf.int32), depth=self.num_outputs, dtype=tf.float32, name="prev_action_one_hot")

            last_action_embedding = self.last_action_embed(tf.expand_dims(prev_action, axis=0))
            action_state_embedding = self.action_state_embed(tf.cast(action_state_self, tf.int32))
            opp_char_state_joint_embedding = self.opp_char_state_joint_embed(
                tf.cast(action_state_opp + self.n_action_states * opp_char_input, tf.int32))
            opp_char_embedding = self.opp_char_embed(tf.cast(opp_char_input, tf.int32))
            prev_reward = tf.expand_dims(prev_reward, axis=0)


        else:

            continuous_inputs = [v for k, v in obs["continuous"].items()]

            binary_inputs = [tf.cast(v, dtype=tf.float32, name=k) for k, v in obs["binary"].items()]

            categorical_inputs = obs["categorical"]

            categorical_one_hots = [tf.one_hot(tf.cast(tensor, tf.int32), depth=tf.cast(space.high[0], tf.int32) + 1, dtype=tf.float32, name=name)[:, :, 0]
                                    for tensor, (name, space) in
                                    zip(categorical_inputs.values(), self.observation_space["categorical"].items())
                                    if name not in ["character1", "action1", "action2", "character2"]] # "action1", "action2", "character2"
            #last_action_one_hot = tf.one_hot(tf.cast(prev_action, tf.int32), depth=self.num_outputs, dtype=tf.float32, name="prev_action_one_hot")

            action_state_self = categorical_inputs["action1"]
            action_state_opp = categorical_inputs["action2"]
            opp_char_input = categorical_inputs["character2"]

            #last_action_one_hot = tf.one_hot(tf.cast(prev_action, tf.int32), depth=self.num_outputs, dtype=tf.float32, name="prev_action_one_hot")

            last_action_embedding = self.last_action_embed(prev_action)
            action_state_embedding = self.action_state_embed(tf.cast(action_state_self, tf.int32))[:, :, 0]
            opp_char_state_joint_embedding = self.opp_char_state_joint_embed(tf.cast(action_state_opp + self.n_action_states * opp_char_input, tf.int32))[:, :, 0]
            opp_char_embedding = self.opp_char_embed(tf.cast(opp_char_input, tf.int32))[:, :, 0]

        obs_input_post_embedding = self.post_embedding_concat(
            continuous_inputs + binary_inputs + categorical_one_hots
            +
            [
                last_action_embedding, action_state_embedding, opp_char_state_joint_embedding, opp_char_embedding,
            tf.tanh(tf.expand_dims(tf.cast(prev_reward, dtype=tf.float32), axis=-1) / 5.), #last_action_one_hot
            ]

        )

        x = self._mlp(obs_input_post_embedding)


        pi_out = self._pi_out(x)
        self._values = self._value_out(x)

        return (pi_out, None), self._values


    def compute_predicted_values(self):
        return tf.reduce_sum(self.centers * tf.nn.softmax(self._values),
                              axis=-1)

    def targets_to_probs(self, targets):
        cdf_evals = tf.math.erf(
            (self.support - tf.expand_dims(targets, axis=1))
            / self.sqrt_two_sigma
        )
        z = cdf_evals[:, -1:] - cdf_evals[:, :1]
        bin_probs = cdf_evals[:, 1:] - cdf_evals[:, :-1]

        return bin_probs / z


















