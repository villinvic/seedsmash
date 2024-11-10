from polaris.models import BaseModel

import sonnet as snt
import tree
from gymnasium.spaces import Discrete
import tensorflow as tf

from polaris.experience import SampleBatch

from models.modules import ResLSTMBlock

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import RMSprop
import numpy as np
from polaris.models.utils import CategoricalDistribution


class ResNetCatVF(BaseModel):
    is_recurrent = True

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
        states = tree.map_structure(
            lambda v: np.repeat(v, B, axis=0), dummy_state
        )
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
            SampleBatch.STATE      : states,
            SampleBatch.SEQ_LENS   : seq_lens,
        })

    def __init__(
            self,
            observation_space,
            action_space: Discrete,
            config,
    ):
        super(ResNetCatVF, self).__init__(
            name="ResNetCatVF",
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )
        self.action_dist = CategoricalDistribution

        self.num_outputs = action_space.n

        # self.optimiser = snt.optimizers.Adam(
        #     learning_rate=config.lr,
        #     name="adam"
        # )

        self.optimiser = snt.optimizers.RMSProp(
            learning_rate=config.lr,
            epsilon=1e-5,
            decay=0.99,
            momentum=0.,
        )

        self.encoder = snt.nets.MLP([256, 256], activate_final=True, name="encoder_mlp")
        self.deep_rnn = snt.DeepRNN([snt.LSTM(256) for _ in range(1)])
        self._pi_out = snt.Linear(self.num_outputs, name="pi_out")

        # Categorical value function

        self.num_bins = 50
        self.v_min, self.v_max = (-5., 5.)
        self.bin_width = (self.v_max - self.v_min) / self.num_bins
        self.support = tf.cast(tf.expand_dims(tf.expand_dims(tf.linspace(self.v_min, self.v_max, self.num_bins + 1), axis=0), axis=0),
                               tf.float32)
        self.centers = (self.support[0, :, :-1] + self.support[0, :, 1:]) / 2.
        smoothing_ratio = 0.75
        sigma = smoothing_ratio * self.bin_width
        self.sqrt_two_sigma = tf.math.sqrt(2.) * sigma
        self._value_out = snt.Linear(self.num_bins)


        self.lstm_input_concat = tf.keras.layers.Concatenate(axis=-1, name="lstm_input_concat")
        self.post_embedding_concat = tf.keras.layers.Concatenate(axis=-1, name="post_embedding_concat")
        self.post_true_embedding_concat = tf.keras.layers.Concatenate(axis=-1, name="post_true_embedding_concat")


    def forward(self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens,
            single_obs=False,
            **kwargs

    ):
        categorical_inputs = obs["categorical"]
        continuous_inputs = obs["continuous"]
        binary_inputs = obs["binary"]

        # true_categorical_inputs = obs["ground_truth"]["categorical"]
        # true_continuous_inputs = obs["ground_truth"]["continuous"]
        # true_binary_inputs = obs["ground_truth"]["binary"]

        if single_obs:

            continuous_inputs = [tf.expand_dims(continuous_inputs[k], axis=0, name=k) for k in self.observation_space["continuous"]]

            binary_inputs = [tf.cast(tf.expand_dims(binary_inputs[k], axis=0), dtype=tf.float32, name=k) for k in
                             self.observation_space["binary"]]

            categorical_one_hots = [
                tf.one_hot(tf.cast(categorical_inputs[k], tf.int32), depth=tf.cast(space.high[0], tf.int32) + 1, dtype=tf.float32,
                           name=k)
                for k, space in self.observation_space["categorical"].items()
                if k not in ["character1"]]  # "action1", "action2", "character2"

            last_action_one_hot = tf.one_hot(tf.cast(tf.expand_dims(prev_action, axis=0), tf.int32),
                                             depth=self.num_outputs, dtype=tf.float32, name="prev_action_one_hot")

            obs_input_post_embedding = tf.expand_dims(self.post_embedding_concat(
                continuous_inputs + binary_inputs + categorical_one_hots + [last_action_one_hot]), axis=0
            )
            ###############

            # true_continuous_inputs = [tf.expand_dims(true_continuous_inputs[k], axis=0, name=k) for k in self.observation_space["continuous"]]
            #
            # true_binary_inputs = [tf.cast(tf.expand_dims(true_binary_inputs[k], axis=0), dtype=tf.float32, name=k) for k in
            #                  self.observation_space["binary"]]
            #
            # true_categorical_one_hots = [
            #     tf.one_hot(tf.cast(true_categorical_inputs[k], tf.int32), depth=tf.cast(space.high[0], tf.int32) + 1, dtype=tf.float32,
            #                name=k)
            #     for k, space in self.observation_space["categorical"].items()
            #     if k not in ["character1"]]  # "action1", "action2", "character2"
            #
            # obs_input_post_true_embedding = tf.expand_dims(self.post_true_embedding_concat(
            #     true_continuous_inputs + true_binary_inputs + true_categorical_one_hots + [last_action_one_hot]), axis=0
            # )


        else:
            continuous_inputs = [continuous_inputs[k] for k in self.observation_space["continuous"]]

            binary_inputs = [tf.cast(binary_inputs[k], dtype=tf.float32, name=k) for k in self.observation_space["binary"]]

            categorical_one_hots = [
                tf.one_hot(tf.cast(categorical_inputs[k], tf.int32), depth=tf.cast(space.high[0], tf.int32) + 1, dtype=tf.float32,
                           name=k)[:, :, 0]
                for k, space in self.observation_space["categorical"].items()
                if k not in ["character1"]]  # "action1", "action2", "character2"

            last_action_one_hot = tf.one_hot(tf.cast(prev_action, tf.int32), depth=self.num_outputs, dtype=tf.float32, name="prev_action_one_hot")

            obs_input_post_embedding = self.post_embedding_concat(
                continuous_inputs + binary_inputs + categorical_one_hots
                +[last_action_one_hot]
                )

            ######################

            # true_continuous_inputs = [true_continuous_inputs[k] for k in self.observation_space["continuous"]]
            #
            # true_binary_inputs = [tf.cast(true_binary_inputs[k], dtype=tf.float32, name=k) for k in self.observation_space["binary"]]
            #
            # true_categorical_one_hots = [
            #     tf.one_hot(tf.cast(true_categorical_inputs[k], tf.int32), depth=tf.cast(space.high[0], tf.int32) + 1, dtype=tf.float32,
            #                name=k)[:, :, 0]
            #     for k, space in self.observation_space["categorical"].items()
            #     if k not in ["character1"]]  # "action1", "action2", "character2"
            #
            # obs_input_post_true_embedding = self.post_true_embedding_concat(
            #     true_continuous_inputs + true_binary_inputs + true_categorical_one_hots
            #     +[last_action_one_hot]
            #     )


        lstm_out, states_out = snt.static_unroll(
            self.deep_rnn,
            input_sequence=obs_input_post_embedding,
            initial_state=state,
            sequence_length=seq_lens
        )

        f = self.encoder(lstm_out)

        action_logits = self._pi_out(f)
        self._value_logits = self._value_out(f)

        return (action_logits, states_out), tf.squeeze(self.compute_predicted_values()), {}


    def get_initial_state(self):
        return tuple(snt.LSTMState(
                hidden=np.zeros((1, 256,), dtype=np.float32),
                cell=np.zeros((1, 256,), dtype=np.float32),
        ) for _ in range(1))

    def compute_predicted_values(self):
        return tf.reduce_sum(self.centers * tf.nn.softmax(self._value_logits),
                              axis=-1)

    def targets_to_probs(self, targets):
        cdf_evals = tf.math.erf(
            (self.support - tf.expand_dims(targets, axis=2))
            / self.sqrt_two_sigma
        )
        z = cdf_evals[:, :, -1:] - cdf_evals[:, :,  :1]
        bin_probs = cdf_evals[:, :, 1:] - cdf_evals[:, :, :-1]

        return bin_probs / z


    def critic_loss(self, targets):

        # HL-Gauss classification loss
        return tf.losses.categorical_crossentropy(
            y_true=self.targets_to_probs(targets),
            y_pred=self._value_logits,
            from_logits=True,
        )


    def aux_loss(self, **k):
        return 0.

    def get_metrics(self):
        return {}

