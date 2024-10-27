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
from wandb import keras

from models.modules import LayerNormLSTM, ResLSTMBlock, ResGRUBlock, ResItem

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import RMSprop
import tensorflow_probability as tfp
import numpy as np
from polaris.models.utils import CategoricalDistribution, GaussianDistribution


def add_batch_time_dimensions(v):
    return tf.expand_dims(tf.expand_dims(v, axis=0), axis=0)


class SS2(BaseModel):
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
        super(SS2, self).__init__(
            name="SS2",
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
        self.aux_optimiser = snt.optimizers.Adam(
            learning_rate=config.aux_lr,
            name="adam"
        )

        # undelay LSTM
        self.observation_keys = {
            aid: {
                obs_type: [k for k in self.observation_space[obs_type] if aid in k]
                for obs_type in ["categorical", "binary", "continuous"]
            }
            for aid in ["1", "2"]
        }

        self.continuous_high = add_batch_time_dimensions(tf.concat(
            [self.observation_space["continuous"][k].high for k in self.observation_keys["1"]["continuous"]],
            axis=0
        ))

        self.continuous_high =add_batch_time_dimensions(tf.concat(
            [self.observation_space["continuous"][k].low for k in self.observation_keys["1"]["continuous"]],
            axis=0
        ))

        self.undelay_encoder = snt.nets.MLP([128, 128], activate_final=True, name="predict_encoder")
        self.undelay_rnn = snt.DeepRNN([ResGRUBlock(128) for _ in range(1)], name="predict_rnn")
        self.to_residual = snt.Linear(32, name="to_residual")

        self.res_items = [ResItem(
            embedder=lambda x: tf.one_hot(tf.cast(x, dtype=tf.int32), depth=v.high[0]+1, dtype=tf.float32),
            sampler=lambda logits: CategoricalDistribution(logits).sample(),
            loss_func=lambda true, pred: tf.keras.losses.categorical_crossentropy(true, pred, from_logits=True),
            embedding_size=v.high[0]+1,
            residual_size=32,
        ) for k, v in self.observation_space["categorical"].items() if "1" in k] + [ResItem(
            embedder=lambda x: x,
            sampler=lambda logits: tfp.distributions.Bernoulli(logits=logits).sample(),
            loss_func=lambda true, pred:  tf.keras.losses.binary_crossentropy(true, pred, from_logits=True),
            embedding_size=v.shape[0],
            residual_size=32,
        ) for k, v in self.observation_space["binary"].items() if "1" in k] + [ResItem(
            embedder=lambda x: tf.clip_by_value(x, v.low, v.high),
            sampler=lambda logits: GaussianDistribution(logits).sample(),
            loss_func=lambda true, pred: -GaussianDistribution(pred).logp(true),
            embedding_size=v.shape[0] * 2,
            residual_size=32,
        ) for k, v in self.observation_space["continuous"].items() if "1" in k]

        # full game
        self.game_embeddings = snt.nets.MLP([128, 128], activate_final=True,
                                            name="game_embeddings")

        # partial obs LSTM
        self.partial_obs_lstm = snt.DeepRNN([ResLSTMBlock(128+self.num_outputs) for _ in range(1)])
        self._pi_out = snt.Linear(self.num_outputs, name="pi_out")

        # Categorical value function

        self.num_bins = 50
        self.v_min, self.v_max = (-5., 5.)
        self.bin_width = (self.v_max - self.v_min) / self.num_bins
        self.support = tf.cast(add_batch_time_dimensions(tf.linspace(self.v_min, self.v_max, self.num_bins + 1)),
                               tf.float32)
        self.centers = (self.support[0, :, :-1] + self.support[0, :, 1:]) / 2.
        smoothing_ratio = 0.75
        sigma = smoothing_ratio * self.bin_width
        self.sqrt_two_sigma = tf.math.sqrt(2.) * sigma
        self._value_out = snt.Linear(self.num_bins, name="value_out")

    def compute_action(
            self,
            input_dict: SampleBatch
    ):

        tx = []
        t = time.time()
        # batch_input_dict = tree.map_structure(expand_values, input_dict)
        self.prepare_single_input(input_dict)

        tx.append(time.time() - t)
        t = time.time()

        (action_logits, state, predictions), value, action, logp = self._compute_action_dist(
            input_dict
        )

        tx.append(time.time() - t)
        t = time.time()

        out = (action.numpy(), tree.map_structure(lambda v: v.numpy(), state), logp.numpy(),
               action_logits.numpy(), value.numpy())

        tx.append(time.time() - t)

        input_dict["obs"]["sampled_prediction"] = predictions.numpy()

        return out + (tx,)

    def compute_value(self, input_dict: SampleBatch):
        self.prepare_single_input(input_dict)
        return self._compute_value(input_dict).numpy()

    @tf.function
    def _compute_value(self, input_dict):
        _, value = self(input_dict)
        return value

    @tf.function
    def _compute_action_dist(self, input_dict):
        (action_logits, state, predictions), value = self(input_dict)
        action_logits = tf.squeeze(action_logits)
        action_dist = self.action_dist(action_logits)
        action = action_dist.sample()
        logp = action_dist.logp(action)
        return (action_logits, state, tf.squeeze(predictions)), value, action, logp

    def get_flat_player_obs(self, obs, player_id, single_obs=False):
        categorical_inputs = obs["categorical"]
        continuous_inputs = obs["continuous"]
        binary_inputs = obs["binary"]

        if single_obs:
            continuous_inputs = [add_batch_time_dimensions(continuous_inputs[k]) for k in
                                 self.observation_keys[player_id]["continuous"]]
            binary_inputs = [add_batch_time_dimensions(tf.cast(binary_inputs[k], dtype=tf.float32, name=k)) for k in
                             self.observation_keys[player_id]["binary"]]
            categorical_inputs = [tf.cast(tf.expand_dims(categorical_inputs[k], axis=0), dtype=tf.int32, name=k) for k in
                             self.observation_keys[player_id]["categorical"]]
        else:

            continuous_inputs = [continuous_inputs[k] for k in
                                 self.observation_keys[player_id]["continuous"]]
            binary_inputs = [tf.cast(binary_inputs[k], dtype=tf.float32, name=k) for k in
                             self.observation_keys[player_id]["binary"]]
            categorical_inputs = [tf.cast(tf.squeeze(categorical_inputs[k]), dtype=tf.int32, name=k) for k in
                             self.observation_keys[player_id]["categorical"]]

        return categorical_inputs + binary_inputs + continuous_inputs

    def embed_flat_player_obs(self, flat):
        return [
           res_item.embedder(v) for res_item, v in zip(self.res_items, flat)
        ]

    def get_undelayed_player_embedding(self, self_delayed_flat, opp_delayed_flat, stage_oh, rnn_state, seq_lens):

        opp_embedded = self.embed_flat_player_obs(opp_delayed_flat)

        undelay_input = tf.concat(
                self.embed_flat_player_obs(self_delayed_flat)
                + opp_embedded
                + [
                 stage_oh #, time ?
             ], axis=-1)

        undelay_post_mlp = self.undelay_encoder(
            undelay_input
        )

        undelayed_opp_embedded, next_rnn_state = snt.static_unroll(
            self.undelay_rnn,
            input_sequence=undelay_post_mlp,
            initial_state=rnn_state,
            sequence_length=seq_lens
        )

        residual = self.to_residual(undelayed_opp_embedded)
        samples = []
        embedded_samples = []
        logits = []

        for res_item, prev in zip(self.res_items, opp_embedded):
            residual, logits, sample, embedded_sample = res_item.predict(residual, prev)
            samples.append(sample)
            embedded_samples.append(embedded_sample)
            logits.append(logits)

        return samples, embedded_samples, logits, next_rnn_state

    def get_game_embed(self, self_embedded, opp_undelayed_embedded, stage_oh, prev_action, rnn_state, seq_lens):

        game_embedded = self.game_embeddings(
            tf.concat(
                self_embedded + opp_undelayed_embedded + [stage_oh]
            , axis=-1)
        )

        game_embedded_with_action = tf.concat(
            [game_embedded, prev_action],
            axis=-1
        )

        lstm_out, next_rnn_state = snt.static_unroll(
            self.partial_obs_lstm,
            input_sequence=game_embedded_with_action,
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

        if single_obs:
            stage_oh = tf.expand_dims(stage_oh, axis=0)
        else:
            stage_oh = stage_oh[:, :, 0]

        prev_action = tf.one_hot(tf.cast(prev_action, tf.int32), depth=self.num_outputs)
        if single_obs:
            prev_action = add_batch_time_dimensions(prev_action)

        # TODO: handle cleanly single-step and batch
        self_flat = self.get_flat_player_obs(
            obs["ground_truth"],
            "1",
            single_obs
        )

        self_flat_delayed = self.get_flat_player_obs(
            obs,
            "1",
            single_obs
        )

        opp_flat_delayed = self.get_flat_player_obs(
            obs,
            "2",
            single_obs
        )

        samples, embedded_samples, logits, next_undelay_state = self.get_undelayed_player_embedding(
            self_flat_delayed, opp_flat_delayed, stage_oh,
            state[0], seq_lens,
        )

        if "sampled_prediction" in obs:
            embedded_samples = self.embed_flat_player_obs(obs["sampled_prediction"])

        self._undelayed_opp_logits = logits

        # do we want value and policy gradients backpropagate to the opp state prediction ?
        lstm_out, next_lstm_state = self.get_game_embed(
            self.embed_flat_player_obs(self_flat), embedded_samples, stage_oh, prev_action,
            state[1], seq_lens
        )

        action_logits = self._pi_out(lstm_out)
        self._value_logits = self._value_out(lstm_out)
        self.stage_oh = stage_oh

        return ((action_logits, (next_undelay_state, next_lstm_state), samples),
                tf.squeeze(self.compute_predicted_values()))


    def get_initial_state(self):
        return (tuple(np.zeros((1, 128), dtype=np.float32) for _ in range(1)), tuple(snt.LSTMState(
                hidden=np.zeros((1, 128+self.num_outputs,), dtype=np.float32),
                cell=np.zeros((1, 128+self.num_outputs,), dtype=np.float32),
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
            advantages,
            single_obs=False,
            **kwargs
    ):

        opp_embedded = self.get_flat_player_obs(
            obs["ground_truth"],
            "2",
            single_obs,
        )

        # TODO: test this again
        # should be normalised, therefore this should be ok.
        # advantage_weights = tf.keras.activations.softmax(
        #     tf.math.abs(advantages), axis=[0,1]
        # )

        loss = tf.reduce_mean(tf.concat(
            [res_item.compute_loss(true, pred) for res_item, true, pred in zip(self.res_items, opp_embedded, self._undelayed_opp_logits)],
            axis=-1
        ))
        return loss


    def get_metrics(self):
        return {
        }



