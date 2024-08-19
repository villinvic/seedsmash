from polaris.models import BaseModel

import sonnet as snt
import tree
from gymnasium.spaces import Discrete
import tensorflow as tf

from polaris.experience import SampleBatch

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import RMSprop
import numpy as np
from polaris.models.utils import CategoricalDistribution


class RNN(BaseModel):
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
        super(RNN, self).__init__(
            name="RNN",
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )

        self.num_outputs = action_space.n
        self.character_embedding_size = 5
        self.action_state_embedding_size = 32
        self.joint_embedding_size = 64
        self.last_action_embedding_size = 16

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

        self._lstm = snt.LSTM(self.config.lstm_dim, name="lstm")

        self._pi_out = snt.Linear(
            output_size=self.num_outputs,
            name="action_logits"
        )
        self._value_out = snt.Linear(
            output_size=1,
            name="values"
        )

        self.n_chars = int(observation_space["categorical"]["character1"].high) + 1
        self.n_action_states = int(observation_space["categorical"]["action1"].high) + 1


        # Have a large common action_state embedding
        # have a small self action_state embedding
        # have a small joint char/action_state embedding

        self.last_action_embed = snt.Embed(
            vocab_size=self.num_outputs,
            embed_dim=self.last_action_embedding_size,
            densify_gradients=True,
            name="last_action_embed"
        )
        self.common_action_state_embed = snt.Embed(
            vocab_size=self.n_action_states,
            embed_dim=128,
            densify_gradients=True,
            name="common_action_state_embed"
        )

        self.self_action_state_embed = snt.Embed(
            vocab_size=self.n_action_states,
            embed_dim=16,
            densify_gradients=True,
            name="self_action_state_embed"
        )

        self.opp_joint_char_action_state_embed = snt.Embed(
            vocab_size=self.n_action_states*self.n_chars,
            embed_dim=16,
            densify_gradients=True,
            name="opp_joint_char_action_state_embed"
        )

        self.opp_char_embed = snt.Embed(
            vocab_size=self.n_chars,
            embed_dim=8,
            densify_gradients=True,
            name="opp_char_embed"
        )

        # self.opp_char_state_joint_embed = snt.Embed(
        #     vocab_size=self.n_chars * self.n_action_states,
        #     embed_dim=self.joint_embedding_size,
        #     densify_gradients=True,
        #     name="opp_char_state_joint_embed"
        # )
        self.lstm_input_concat = tf.keras.layers.Concatenate(axis=-1, name="lstm_input_concat")
        self.post_embedding_concat = tf.keras.layers.Concatenate(axis=-1, name="post_embedding_concat")

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

            # action_state_self = categorical_inputs["action1"]
            # action_state_opp = categorical_inputs["action2"]
            # opp_char_input = categorical_inputs["character2"]


            # last_action_embedding = self.last_action_embed(tf.expand_dims(prev_action, axis=0))
            # action_state_self_common_embed = self.common_action_state_embed(tf.cast(action_state_self, tf.int32))
            # action_state_opp_common_embed = self.common_action_state_embed(tf.cast(action_state_opp, tf.int32))
            #
            # action_state_self_embed = self.self_action_state_embed(tf.cast(action_state_self, tf.int32))
            # joint_char_action_state_opp_embed = self.opp_joint_char_action_state_embed(
            #     tf.cast(action_state_opp + self.n_action_states * opp_char_input, tf.int32))
            # char_embedding_opp = self.opp_char_embed(tf.cast(opp_char_input, tf.int32))

            prev_reward = tf.expand_dims(prev_reward, axis=0)

        else:
            continuous_inputs = [continuous_inputs[k] for k in self.observation_space["continuous"]]

            binary_inputs = [tf.cast(binary_inputs[k], dtype=tf.float32, name=k) for k in self.observation_space["binary"]]

            categorical_one_hots = [
                tf.one_hot(tf.cast(categorical_inputs[k], tf.int32), depth=tf.cast(space.high[0], tf.int32) + 1, dtype=tf.float32,
                           name=k)[:, :, 0]
                for k, space in self.observation_space["categorical"].items()
                if k not in ["character1"]]  # "action1", "action2", "character2"

            # action_state_self = categorical_inputs["action1"]
            # action_state_opp = categorical_inputs["action2"]
            # opp_char_input = categorical_inputs["character2"]

            last_action_one_hot = tf.one_hot(tf.cast(prev_action, tf.int32), depth=self.num_outputs, dtype=tf.float32, name="prev_action_one_hot")

            # last_action_embedding = self.last_action_embed(prev_action)
            #
            # action_state_self_common_embed = self.common_action_state_embed(tf.cast(action_state_self, tf.int32))[:, :, 0]
            # action_state_opp_common_embed = self.common_action_state_embed(tf.cast(action_state_opp, tf.int32))[:, :, 0]
            #
            # action_state_self_embed = self.self_action_state_embed(tf.cast(action_state_self, tf.int32))[:, :, 0]
            # joint_char_action_state_opp_embed = self.opp_joint_char_action_state_embed(
            #     tf.cast(action_state_opp + self.n_action_states * opp_char_input, tf.int32))[:, :, 0]
            # char_embedding_opp = self.opp_char_embed(tf.cast(opp_char_input, tf.int32))[:, :, 0]


        obs_input_post_embedding = self.post_embedding_concat(
            continuous_inputs + binary_inputs + categorical_one_hots
            #+ [action_state_self_common_embed, action_state_opp_common_embed,
            #   action_state_self_embed, joint_char_action_state_opp_embed, char_embedding_opp]
        )

        # print(single_obs, [t.shape for t in continuous_inputs + binary_inputs + categorical_one_hots],
        #       obs_input_post_embedding.shape)

        x = self._mlp(obs_input_post_embedding)

        lstm_input = self.lstm_input_concat(
            [x, tf.tanh(tf.expand_dims(tf.cast(prev_reward, dtype=tf.float32), axis=-1) / 5.),  last_action_one_hot]
        )

        if single_obs:
            lstm_input = (tf.expand_dims(lstm_input, axis=0))

        lstm_out, states_out = snt.static_unroll(
            self._lstm,
            input_sequence=lstm_input,
            initial_state=state,
            sequence_length=seq_lens,
        )

        pi_out = self._pi_out(lstm_out)
        self._values = tf.squeeze(self._value_out(lstm_out))

        if single_obs:
            return (pi_out, states_out), self._values

        return (pi_out, states_out), self._values


    def get_initial_state(self):
        return snt.LSTMState(
                hidden=np.zeros((1, self.config.lstm_dim,), dtype=np.float32),
                cell=np.zeros((1, self.config.lstm_dim,), dtype=np.float32),
            )




















