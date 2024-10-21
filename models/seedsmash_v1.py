from typing import Optional

from polaris.models import BaseModel

import sonnet as snt
import tree
from gymnasium.spaces import Discrete
import tensorflow as tf

from polaris.experience import SampleBatch
from tensorflow.python.ops.gen_data_flow_ops import stage

from models.modules import LayerNormLSTM, ResLSTMBlock, ResGRUBlock

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import RMSprop
import numpy as np
from polaris.models.utils import CategoricalDistribution



class SS1(BaseModel):
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
        super(SS1, self).__init__(
            name="SS1",
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )
        self.action_dist = CategoricalDistribution

        self.num_outputs = action_space.n

        self.optimiser = snt.optimizers.Adam(
            learning_rate=config.lr,
            name="adam"
        )

        self.player_embeddings = snt.nets.MLP([64], activate_final=False)


        # binary
        # facing
        # on_ground
        # invulnerable
        # buttons
        #self.binary_embeddings = snt.nets.MLP([8], activate_final=True)


        # prev_action
        #self.prev_action_embeddings = snt.Embed(self.num_outputs, 16, densify_gradients=True)

        # undelay LSTM
        self.undelay_encoder = snt.nets.MLP([64], activate_final=False,
                                            name="encoder")
        self.undelay_rnn = snt.DeepRNN([ResGRUBlock(64) for _ in range(1)])

        # full game
        self.game_embeddings = snt.nets.MLP([128, 128], activate_final=True,
                                            name="game_embeddings")


        # partial obs LSTM
        self.partial_obs_lstm = snt.DeepRNN([ResLSTMBlock(128) for _ in range(1)])
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
        self._value_out = snt.Linear(self.num_bins, name="value_out")

        self.post_embedding_concat = tf.keras.layers.Concatenate(axis=-1, name="post_embedding_concat")

    def get_player_embedding(self, obs, stage_oh, aid, single_obs):
        categorical_inputs = obs["categorical"]
        continuous_inputs = obs["continuous"]
        binary_inputs = obs["binary"]

        continuous_inputs = [continuous_inputs[k] for k in
                                  self.observation_space["continuous"]
                                  if aid in k]
        binary_inputs = [tf.cast(binary_inputs[k], dtype=tf.float32, name=k) for k in
                              self.observation_space["binary"] if aid in k]

        jumps_oh = tf.one_hot(tf.cast(categorical_inputs[f"jumps_left{aid}"], tf.int32),
                                depth=tf.cast(self.observation_space["categorical"][f"jumps_left{aid}"].high[0],
                                              tf.int32) + 1)
        stocks_oh = tf.one_hot(tf.cast(categorical_inputs[f"stock{aid}"], tf.int32),
                                 depth=tf.cast(self.observation_space["categorical"][f"stock{aid}"].high[0], tf.int32) + 1)
        action_state_oh = tf.one_hot(tf.cast(categorical_inputs[f"action{aid}"], tf.int32),
                                       depth=tf.cast(self.observation_space["categorical"][f"action{aid}"].high[0],
                                                     tf.int32) + 1)
        char_oh = tf.one_hot(tf.cast(categorical_inputs[f"character{aid}"], tf.int32),
                               depth=tf.cast(self.observation_space["categorical"][f"character{aid}"].high[0], tf.int32) + 1)
        if not single_obs:
            jumps_oh = jumps_oh[:, :, 0]
            stocks_oh = stocks_oh[:, :, 0]
            action_state_oh = action_state_oh[:, :, 0]
            char_oh = char_oh[:, :, 0]
        else:
            jumps_oh = jumps_oh[0]
            stocks_oh = stocks_oh[0]
            action_state_oh = action_state_oh[0]
            char_oh = char_oh[0]

        all_concat = tf.concat(
            continuous_inputs + binary_inputs + [
                jumps_oh, stocks_oh, action_state_oh, char_oh, stage_oh
                               ], axis=-1)

        if single_obs:
            all_concat =  tf.expand_dims(tf.expand_dims(all_concat, axis=0), axis=0)

        embed_player = self.player_embeddings(all_concat)


        return embed_player

    def get_undelayed_player_embedding(self, self_embedded, opp_embedded, rnn_state, seq_lens):

        undelay_input = self.undelay_encoder(
            tf.concat([
                self_embedded, opp_embedded
            ], axis=-1)
        )

        undelayed_opp_embedded, next_rnn_state = snt.static_unroll(
            self.undelay_rnn,
            input_sequence=undelay_input,
            initial_state=rnn_state,
            sequence_length=seq_lens
        )
        return undelayed_opp_embedded, next_rnn_state

    def get_game_embed(self, self_embedded, undelayed_op_embedded, prev_action, rnn_state, seq_lens, single_obs):
        if single_obs:
            prev_action = tf.expand_dims(tf.expand_dims(prev_action, axis=0))
        game_embedded = self.game_embeddings(
            tf.concat(
                [self_embedded, undelayed_op_embedded, prev_action]
            , axis=-1)
        )

        lstm_out, next_rnn_state = snt.static_unroll(
            self.partial_obs_lstm,
            input_sequence=game_embedded,
            initial_state=rnn_state,
            sequence_length=seq_lens
        )
        return lstm_out, next_rnn_state


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

        stage = obs["ground_truth"]["categorical"]["stage"]

        stage_oh = tf.one_hot(tf.cast(stage, tf.int32),
                                   depth=tf.cast(self.observation_space["categorical"]["stage"].high[0],
                                                 tf.int32) + 1, dtype=tf.float32, name="stage_one_hot")
        prev_action = tf.one_hot(tf.cast(prev_action, tf.int32), depth=self.num_outputs)

        if not single_obs:
            stage_oh = stage_oh[:, :, 0]
        else:
            stage_oh = stage_oh[0]

        self_embedded = self.get_player_embedding(
            obs["ground_truth"],
            stage_oh,
            "1",
            single_obs
        )

        self_delayed_embedded = self.get_player_embedding(
            obs,
            stage_oh,
            "1",
            single_obs
        )

        opp_delayed_embedded = self.get_player_embedding(
            obs,
            stage_oh,
            "2",
            single_obs
        )

        opp_embedded, next_undelay_state = self.get_undelayed_player_embedding(
            self_delayed_embedded, opp_delayed_embedded,
            state[0], seq_lens
        )
        self._undelayed_opp_embedded = opp_embedded

        lstm_out, next_lstm_state = self.get_game_embed(
            self_embedded, opp_delayed_embedded, prev_action,
            state[1], seq_lens, single_obs

        )

        action_logits = self._pi_out(lstm_out)
        self._value_logits = self._value_out(lstm_out)
        self.stage_oh = stage_oh

        return (action_logits, (next_undelay_state, next_lstm_state)), tf.squeeze(self.compute_predicted_values())


    def get_initial_state(self):
        return (tuple(np.zeros((1, 64), dtype=np.float32) for _ in range(1)), tuple(snt.LSTMState(
                hidden=np.zeros((1, 128,), dtype=np.float32),
                cell=np.zeros((1, 128,), dtype=np.float32),
        ) for _ in range(1)))


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
        ret = bin_probs / z

        return ret


    def critic_loss(self, targets):

        # HL-Gauss classification loss
        return tf.losses.categorical_crossentropy(
            y_true=self.targets_to_probs(targets),
            y_pred=self._value_logits,
            from_logits=True,
        )

    def aux_loss(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens,
            single_obs=False,
            **kwargs
    ):


        opp_embedded = self.get_player_embedding(
            obs,
            self.stage_oh,
            "2",
            single_obs
        )

        return tf.reduce_mean(tf.math.square(tf.stop_gradient(opp_embedded) - self._undelayed_opp_embedded))



