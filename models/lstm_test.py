import time
from typing import Optional

from polaris.models import BaseModel

import sonnet as snt
import tree
from gymnasium.spaces import Discrete
import tensorflow as tf

from polaris.experience import SampleBatch
from tensorflow.python.ops.gen_data_flow_ops import stage

from models.modules import LayerNormLSTM, ResLSTMBlock, ResGRUBlock
from models.seedsmash_v2 import add_batch_time_dimensions

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import RMSprop
import tensorflow_probability as tfp
import numpy as np
from polaris.models.utils import CategoricalDistribution, GaussianDistribution


class CategoricalValueModel(snt.Module):

    def __init__(
            self,
            num_bins=50,
            value_bounds=(-5., 5.),
            smoothing_ratio=0.75
    ):
        super().__init__(name='CategoricalValueModel')
        self.num_bins = num_bins
        self.value_bounds = value_bounds
        self.bin_width = (self.value_bounds[1] - self.value_bounds[0]) / self.num_bins
        self.support = tf.cast(tf.expand_dims(tf.expand_dims(tf.linspace(*self.value_bounds, self.num_bins + 1), axis=0), axis=0),
                               tf.float32)
        self.centers = (self.support[0, :, :-1] + self.support[0, :, 1:]) / 2.
        sigma = smoothing_ratio * self.bin_width
        self.sqrt_two_sigma = tf.math.sqrt(2.) * sigma

        self.head = snt.Linear(self.num_bins, name="value_head")
        self._logits = None

    def __call__(
            self,
            input_
    ):
        self._logits = self.head(input_)
        return tf.reduce_sum(self.centers * tf.nn.softmax(self._logits),
                              axis=-1)

    def targets_to_probs(self, targets):

        # this may occur on rare occasion that targets are outside of the set interval.
        targets = tf.clip_by_value(targets, *self.value_bounds)

        cdf_evals = tf.math.erf(
            (self.support - tf.expand_dims(targets, axis=2))
            / self.sqrt_two_sigma
        )
        z = cdf_evals[:, :, -1:] - cdf_evals[:, :,  :1]
        bin_probs = cdf_evals[:, :, 1:] - cdf_evals[:, :, :-1]
        ret = bin_probs / z

        return ret

    def loss(self, targets):

        # HL-Gauss classification loss
        return tf.losses.categorical_crossentropy(
            y_true=self.targets_to_probs(targets),
            y_pred=self._logits,
            from_logits=True,
        )


class LSTMT(BaseModel):
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
        super(LSTMT, self).__init__(
            name="LSTMT",
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )

        self.observation_keys = {
            aid: {
                obs_type: [k for k in self.observation_space[obs_type] if aid in k]
                for obs_type in ["categorical", "binary", "continuous"]
            }
            for aid in ["1", "2"]
        }

        self.action_dist = CategoricalDistribution
        self.num_outputs = action_space.n

        self.optimiser = snt.optimizers.RMSProp(
            learning_rate=config.lr,
            epsilon=1e-5,
            decay=0.99,
            momentum=0.,
        )

        self.state_dim = 256
        self.action_embedding = snt.nets.MLP([16], activate_final=True)

        #self.delay_transformer = snt.nets.
        self.policy_core = snt.nets.MLP([256, 256-16], activate_final=True)
        self.policy_rnn = snt.LSTM(self.state_dim)
        self.policy_head = snt.Linear(self.num_outputs, name="policy_head")
        self.value_core = snt.nets.MLP([256, 256], activate_final=True)
        self.value_head = CategoricalValueModel()

    def get_flat_player_obs(self, obs, player_id, single_obs=False):
        categorical_inputs = obs["categorical"]
        continuous_inputs = obs["continuous"]
        binary_inputs = obs["binary"]

        if single_obs:
            continuous_inputs = [add_batch_time_dimensions(continuous_inputs[k]) for k in
                                 self.observation_keys[player_id]["continuous"]]
            binary_inputs = [add_batch_time_dimensions(tf.cast(binary_inputs[k], dtype=tf.float32, name=k)) for k in
                             self.observation_keys[player_id]["binary"]]
            categorical_inputs = [
                tf.one_hot(tf.cast(categorical_inputs[k], dtype=tf.int32, name=k),
                           depth=tf.cast(self.observation_space["categorical"][k].high[0], tf.int32)+1, dtype=tf.float32) for k in
                             self.observation_keys[player_id]["categorical"]
            ]
        else:
            continuous_inputs = [continuous_inputs[k] for k in
                                 self.observation_keys[player_id]["continuous"]]
            binary_inputs = [tf.cast(binary_inputs[k], dtype=tf.float32, name=k) for k in
                             self.observation_keys[player_id]["binary"]]
            categorical_inputs = [tf.one_hot(tf.cast(tf.squeeze(categorical_inputs[k]), dtype=tf.int32, name=k),
                           depth=tf.cast(self.observation_space["categorical"][k].high[0], tf.int32)+1, dtype=tf.float32) for k in
                             self.observation_keys[player_id]["categorical"]]

        return categorical_inputs + binary_inputs + continuous_inputs

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
        if single_obs:
            stage_oh = tf.expand_dims(stage_oh, axis=0)
            state = tf.expand_dims(state, axis=0)
            prev_action = add_batch_time_dimensions(prev_action)
        else:
            stage_oh = stage_oh[:, :, 0]

        last_action_one_hot = tf.one_hot(tf.cast(prev_action, tf.int32), depth=self.num_outputs, dtype=tf.float32,
                                         name="prev_action_one_hot")

        embed_prev_action = self.action_embedding(last_action_one_hot)


        self_true = self.get_flat_player_obs(obs["ground_truth"], "1")
        opp_true = self.get_flat_player_obs(obs["ground_truth"], "2")

        #self_delayed = self.get_flat_player_obs(obs, "1")
        opp_delayed = self.get_flat_player_obs(obs, "2")

        player_obs = self_true + opp_delayed
        true_player_obs = self_true + opp_true

        if single_obs:
            player_obs = [add_batch_time_dimensions(tf.concat(player_obs, axis=0))]
            true_player_obs = [add_batch_time_dimensions(tf.concat(true_player_obs, axis=0))]

        pi_input = tf.concat(
            player_obs + [stage_oh],
            axis = -1
        )
        pi_core_out = self.policy_core(pi_input)
        rnn_input = tf.concat(
            [pi_core_out, embed_prev_action]
            , axis=-1)

        pi_lstm_out, next_state = snt.static_unroll(
            self.policy_rnn,
            input_sequence=rnn_input,
            initial_state=state,
            sequence_length=seq_lens
        )

        action_logits = self.policy_head(pi_input)

        v_input = tf.concat(
            true_player_obs + [stage_oh, embed_prev_action],
            axis = -1
        )

        v_core_out = self.value_core(v_input)
        value = self.value_head(v_core_out)

        return ((action_logits, next_state), tf.squeeze(value)), {}


    def get_initial_state(self):
        return snt.LSTMState(
                hidden=np.zeros((1, self.state_dim,), dtype=np.float32),
                cell=np.zeros((1, self.state_dim,), dtype=np.float32),
        )

    def critic_loss(self, targets):
        return self.value_head.loss(targets)

    def get_metrics(self):
        return {}

    def aux_loss(self, **k):
        return 0.



