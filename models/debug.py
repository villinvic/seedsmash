import time
from typing import Optional

from polaris.models import BaseModel

import sonnet as snt
import tree
from gymnasium.spaces import Discrete
import tensorflow as tf

from polaris.experience import SampleBatch
from tensorflow.python.ops.gen_data_flow_ops import stage
from tensorflow.python.ops.random_ops import categorical

from models.modules import LayerNormLSTM, ResLSTMBlock, ResGRUBlock

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import RMSprop
import tensorflow_probability as tfp
import numpy as np
from polaris.models.utils import CategoricalDistribution, GaussianDistribution



class Debug(BaseModel):
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
            self.aux_loss()

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
        super(Debug, self).__init__(
            name="Debug",
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )
        self.action_dist = CategoricalDistribution

        self.num_outputs = action_space.n

        self.optimiser = snt.optimizers.RMSProp(
            learning_rate=config.lr,
            epsilon=1e-5,
            decay=0.99,
            momentum=0.,
        )

        #self.player_embeddings = snt.nets.MLP([64], activate_final=False)


        # binary
        # facing
        # on_ground
        # invulnerable
        # buttons
        #self.binary_embeddings = snt.nets.MLP([8], activate_final=True)


        # prev_action
        #self.prev_action_embeddings = snt.Embed(self.num_outputs, 16, densify_gradients=True)

        # undelay LSTM
        self.embed_binary_size = sum([obs.shape[-1] for k, obs in self.observation_space["binary"].items() if "1" in k])
        self.embed_categorical_sizes = [int(obs.high[0])+1 for k, obs in self.observation_space["categorical"].items() if "1" in k]
        self.embed_categorical_total_size = sum(self.embed_categorical_sizes)
        self.embed_continuous_size = sum([obs.shape[-1] for k, obs in self.observation_space["continuous"].items() if "1" in k])
        self.embedding_size = (
            self.embed_binary_size+self.embed_categorical_total_size+self.embed_continuous_size
        )
        self.embedding_size_with_stds = (
                self.embed_binary_size + self.embed_categorical_total_size + self.embed_continuous_size * 2
        )
        self.continuous_high = tf.expand_dims(tf.expand_dims(tf.concat(
            [v.high for k, v in self.observation_space["continuous"].items() if "1" in k],
            axis=0
        ), axis=0), axis=0)
        self.continuous_low = tf.expand_dims(tf.expand_dims(tf.concat(
            [v.low for k, v in self.observation_space["continuous"].items() if "1" in k],
            axis=0
        ), axis=0), axis=0)

        self.pred_encoder = snt.Linear(self.embedding_size)

        self.state_size = 512
        self.rnn = snt.LSTM(self.state_size)
        self._pi_out = snt.nets.MLP([128, self.num_outputs], name="pi_out")


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
        self._value_out = snt.nets.MLP([128, self.num_bins], name="value_out")

        self.post_embedding_concat = tf.keras.layers.Concatenate(axis=-1, name="post_embedding_concat")

    def get_player_embedding(self, obs, aid, single_obs):
        categorical_inputs = obs["categorical"]
        continuous_inputs = obs["continuous"]
        binary_inputs = obs["binary"]

        continuous_inputs = [continuous_inputs[k] for k in
                                  self.observation_space["continuous"]
                                  if aid in k]
        binary_inputs = [tf.cast(binary_inputs[k], dtype=tf.float32, name=k) for k in
                              self.observation_space["binary"] if aid in k]


        # jumps_oh = tf.one_hot(tf.cast(categorical_inputs[f"jumps_left{aid}"], tf.int32),
        #                         depth=tf.cast(self.observation_space["categorical"][f"jumps_left{aid}"].high[0],
        #                                       tf.int32) + 1)
        # stocks_oh = tf.one_hot(tf.cast(categorical_inputs[f"stock{aid}"], tf.int32),
        #                          depth=tf.cast(self.observation_space["categorical"][f"stock{aid}"].high[0], tf.int32) + 1)
        # action_state_oh = tf.one_hot(tf.cast(categorical_inputs[f"action{aid}"], tf.int32),
        #                                depth=tf.cast(self.observation_space["categorical"][f"action{aid}"].high[0],
        #                                              tf.int32) + 1)
        # char_oh = tf.one_hot(tf.cast(categorical_inputs[f"character{aid}"], tf.int32),
        #                        depth=tf.cast(self.observation_space["categorical"][f"character{aid}"].high[0], tf.int32) + 1)
        if not single_obs:
            one_hots = [
                tf.one_hot(tf.cast(categorical_inputs[k], tf.int32),
                           depth=tf.cast(self.observation_space["categorical"][k].high[0],
                                         tf.int32) + 1)[:, :, 0]
                for k in self.observation_space["categorical"] if aid in k
            ]
            # jumps_oh = jumps_oh[:, :, 0]
            # stocks_oh = stocks_oh[:, :, 0]
            # action_state_oh = action_state_oh[:, :, 0]
            # char_oh = char_oh[:, :, 0]
        else:
            one_hots = [
                tf.one_hot(tf.cast(categorical_inputs[k], tf.int32),
                           depth=tf.cast(self.observation_space["categorical"][k].high[0],
                                         tf.int32) + 1)[0]
                for k in self.observation_space["categorical"] if aid in k
            ]
            # jumps_oh = jumps_oh[0]
            # stocks_oh = stocks_oh[0]
            # action_state_oh = action_state_oh[0]
            # char_oh = char_oh[0]

        embed_player = tf.concat(
            continuous_inputs + binary_inputs + one_hots, axis=-1)

        if single_obs:
            embed_player =  tf.expand_dims(tf.expand_dims(embed_player, axis=0), axis=0)


        return embed_player

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
            prev_action = tf.expand_dims(tf.expand_dims(prev_action, axis=0), axis=0)
            stage_oh = tf.expand_dims(stage_oh, axis=0)

        self_embedded = self.get_player_embedding(
            obs["ground_truth"],
            "1",
            single_obs
        )

        self.opp_embedded = self.get_player_embedding(
            obs["ground_truth"],
            "2",
            single_obs
        )

        opp_delayed_embedded = self.get_player_embedding(
            obs,
            "2",
            single_obs
        )

        rnn_input = tf.concat(
            [self_embedded, opp_delayed_embedded, stage_oh, prev_action],
            axis = -1
        )

        #rnn_input = self.rnn_encoder(pi_input)
        lstm_out, next_state = snt.static_unroll(
            self.rnn,
            input_sequence=rnn_input,
            initial_state=state,
            sequence_length=seq_lens
        )

        self.lstm_out = lstm_out

        v_input = tf.concat(
            [self.opp_embedded, lstm_out],
            axis = -1
        )

        action_logits = self._pi_out(lstm_out)

        self._value_logits = self._value_out(v_input)

        return (action_logits, next_state), tf.squeeze(self.compute_predicted_values()), {}


    def get_initial_state(self):
        return snt.LSTMState(
                hidden=np.zeros((1, self.state_size,), dtype=np.float32),
                cell=np.zeros((1, self.state_size,), dtype=np.float32),
        )

    def compute_predicted_values(self):
        return tf.reduce_sum(self.centers * tf.nn.softmax(self._value_logits),
                              axis=-1)

    def targets_to_probs(self, targets):

        targets = tf.clip_by_value(targets, self.v_min, self.v_max)
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

    def predict_opp_ground_truth(self):
        _, aux_out = tf.split(self.lstm_out, 2, axis=-1)
        pred = self.pred_encoder(self.lstm_out)

        continuous, binary, categoricals = tf.split(pred, [self.embed_continuous_size,
                                                                             self.embed_binary_size,
                                                                             self.embed_categorical_total_size], axis=-1)
        categoricals = tf.split(categoricals, self.embed_categorical_sizes, axis=-1)

        return continuous, binary, categoricals


    def aux_loss(
            self,
            **kwargs
    ):
        # continuous, binary, categoricals = self.predict_opp_ground_truth()
        #
        # true_continuous, true_binary, true_categoricals = tf.split(self.opp_embedded, [self.embed_continuous_size,
        #                                                   self.embed_binary_size,
        #                                                   self.embed_categorical_total_size], axis=-1)
        # true_categoricals = tf.split(true_categoricals, self.embed_categorical_sizes, axis=-1)
        #
        #
        # self.continuous_loss = tf.reduce_mean(tf.keras.losses.huber(
        #     true_continuous, continuous, delta=0.3
        # ))
        #
        # self.binary_loss = tf.reduce_mean(
        #     # advantage_weights*
        #     tf.keras.losses.binary_crossentropy(
        #         true_binary, binary,
        #         from_logits=True,
        #     ))
        #
        # self.categorical_loss = tf.reduce_mean([
        #     # tf.reduce_mean(advantage_weights *
        #     tf.keras.losses.categorical_crossentropy(
        #         t, p, from_logits=True
        #     )
        #     # )
        #     for t, p in zip(true_categoricals, categoricals)
        # ])
        #
        # self.tmp1 = true_continuous[0, 0]
        # self.tmp2 = continuous[0, 0]
        #
        # return self.continuous_loss + self.binary_loss + self.categorical_loss
        return 0.


    def get_metrics(self):
        return {
            # "continuous_loss": self.continuous_loss,
            # "categorical_loss": self.categorical_loss,
            # "binary_loss": self.binary_loss,
            # "tmp1": self.tmp1,
            # "tmp2": self.tmp2,
        }



