import time
from typing import Optional

from polaris.models import BaseModel

import sonnet as snt
import tree
from gymnasium.spaces import Discrete
import tensorflow as tf

from polaris.experience import SampleBatch
from models.modules import LayerNormLSTM, ResLSTMBlock, ResGRUBlock, ResMLP

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import RMSprop
import tensorflow_probability as tfp
import numpy as np
from polaris.models.utils import CategoricalDistribution, GaussianDistribution



class Debug6(BaseModel):
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
        super(Debug6, self).__init__(
            name="Debug5",
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

        self.opp_predictor = snt.Linear(self.embedding_size)
        self.encoder = snt.nets.MLP([256, 256], activate_final=True, name="encoder")
        self.state_size = 256
        self.rnn = snt.LSTM(self.state_size)
        self._pi_out = snt.nets.MLP([128, self.num_outputs], name="pi_out")
        self._value_out = snt.nets.MLP([128, 1], name="value_out")

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

        if not single_obs:
            one_hots = [
                tf.one_hot(tf.cast(categorical_inputs[k], tf.int32),
                           depth=tf.cast(self.observation_space["categorical"][k].high[0],
                                         tf.int32) + 1)[:, :, 0]
                for k in self.observation_space["categorical"] if aid in k
            ]
        else:
            one_hots = [
                tf.one_hot(tf.cast(categorical_inputs[k], tf.int32),
                           depth=tf.cast(self.observation_space["categorical"][k].high[0],
                                         tf.int32) + 1)[0]
                for k in self.observation_space["categorical"] if aid in k
            ]

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
            compute_value=True,
            **kwargs

    ):

        # global stuff
        stage = obs["ground_truth"]["categorical"]["stage"]
        stage_width = obs["ground_truth"]["continuous"]["stage_width"]

        stage_oh = tf.one_hot(tf.cast(stage, tf.int32),
                                   depth=tf.cast(self.observation_space["categorical"]["stage"].high[0],
                                                 tf.int32) + 1, dtype=tf.float32, name="stage_one_hot")
        prev_action = tf.one_hot(tf.cast(prev_action, tf.int32), depth=self.num_outputs)
        if not single_obs:
            stage_oh = stage_oh[:, :, 0]
        else:
            prev_action = tf.expand_dims(tf.expand_dims(prev_action, axis=0), axis=0)
            stage_oh = tf.expand_dims(stage_oh, axis=0)
            stage_width = tf.expand_dims(tf.expand_dims(stage_width, axis=0), axis=0)

        self_embedded = self.get_player_embedding(
            obs["ground_truth"],
            "1",
            single_obs
        )

        opp_delayed_embedded = self.get_player_embedding(
            obs,
            "2",
            single_obs
        )

        core_embed = self.encoder(tf.concat(
            [self_embedded, opp_delayed_embedded, stage_oh, stage_width, prev_action],
            axis=-1
        ))

        lstm_out, next_state = snt.static_unroll(
            self.rnn,
            input_sequence=core_embed,
            initial_state=state,
            sequence_length=seq_lens
        )
        self.out = tf.concat([lstm_out, core_embed], axis=-1)

        action_logits = self._pi_out(self.out)

        if compute_value:
            self.opp_ground_truth = self.get_player_embedding(
                obs["ground_truth"],
                "2",
                single_obs=single_obs
            )
            value_input = tf.concat([self.out, self.opp_ground_truth], axis=-1)
            self._value_logits = self._value_out(value_input)
            return (action_logits, next_state), tf.squeeze(self._value_logits), {}
        else:
            return action_logits, next_state


    def get_initial_state(self):
        return snt.LSTMState(
                hidden=np.zeros((1, self.state_size,), dtype=np.float32),
                cell=np.zeros((1, self.state_size,), dtype=np.float32),
        )

    def critic_loss(self, targets):

        return tf.math.square(tf.squeeze(self._value_logits) - targets)

    def predict_opp_ground_truth(self):
        return self.opp_predictor(self.out)

    def split_to_predicted_types(self, pred):

        continuous, binary, categoricals = tf.split(pred, [self.embed_continuous_size,
                                                                             self.embed_binary_size,
                                                                             self.embed_categorical_total_size], axis=-1)
        categoricals = tf.split(categoricals, self.embed_categorical_sizes, axis=-1)

        return continuous, binary, categoricals


    def aux_loss(
            self,
            *
            obs,
            **kwargs
    ):

        # todo: dont concat and split...
        predicted = self.predict_opp_ground_truth()
        continuous, binary, categoricals = self.split_to_predicted_types(predicted)

        true_continuous, true_binary, true_categoricals = tf.split(self.opp_ground_truth, [self.embed_continuous_size,
                                                          self.embed_binary_size,
                                                          self.embed_categorical_total_size], axis=-1)
        true_categoricals = tf.split(true_categoricals, self.embed_categorical_sizes, axis=-1)


        self.continuous_loss = tf.reduce_mean(tf.keras.losses.huber(
            true_continuous, continuous, delta=0.3
        ))

        self.binary_loss = tf.reduce_mean(
            # advantage_weights*
            tf.keras.losses.binary_crossentropy(
                true_binary, binary,
                from_logits=True,
            ))

        self.categorical_loss = tf.reduce_mean([
            # tf.reduce_mean(advantage_weights *
            tf.keras.losses.categorical_crossentropy(
                t, p, from_logits=True
            )
            # )
            for t, p in zip(true_categoricals, categoricals)
        ])

        self.tmp1 = true_continuous[0, 0]
        self.tmp2 = continuous[0, 0]

        return self.continuous_loss + self.binary_loss + self.categorical_loss


    def get_metrics(self):
        return {
            "continuous_loss": self.continuous_loss,
            "categorical_loss": self.categorical_loss,
            "binary_loss": self.binary_loss,
            "tmp1": self.tmp1,
            "tmp2": self.tmp2,
        }



