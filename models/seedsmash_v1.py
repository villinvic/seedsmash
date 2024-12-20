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
        self.aux_optimiser = snt.optimizers.Adam(
            learning_rate=config.aux_lr,
            name="adam"
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

        self.undelay_encoder = snt.nets.MLP([128, 128], activate_final=True, name="predict_encoder")
        self.delta_gate = snt.Linear(self.embedding_size_with_stds, w_init=tf.zeros_initializer(), name="predict_delta")
        self.new_gate = snt.Linear(self.embedding_size_with_stds, name="predict_new")
        self.forget_gate = snt.nets.MLP([self.embedding_size_with_stds], activation=tf.sigmoid, activate_final=True,
                                        w_init=tf.zeros_initializer(),
                                        b_init=tf.constant_initializer(-10.), name="predict_forget")

        self.undelay_rnn = snt.DeepRNN([ResGRUBlock(128) for _ in range(1)], name="predict_rnn")

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
        self.support = tf.cast(tf.expand_dims(tf.expand_dims(tf.linspace(self.v_min, self.v_max, self.num_bins + 1), axis=0), axis=0),
                               tf.float32)
        self.centers = (self.support[0, :, :-1] + self.support[0, :, 1:]) / 2.
        smoothing_ratio = 0.75
        sigma = smoothing_ratio * self.bin_width
        self.sqrt_two_sigma = tf.math.sqrt(2.) * sigma
        self._value_out = snt.Linear(self.num_bins, name="value_out")

        self.post_embedding_concat = tf.keras.layers.Concatenate(axis=-1, name="post_embedding_concat")

    def split_player_embedding(self, embedding, stds=False):
        continuous_size = self.embed_continuous_size
        if stds:
            continuous_size = 2 * continuous_size
        continuous, binary, categorical = tf.split(embedding, [continuous_size, self.embed_binary_size, self.embed_categorical_total_size], axis=2)
        categoricals  = tf.split(categorical, self.embed_categorical_sizes, axis=2)

        return continuous, binary, categoricals


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

    def get_undelayed_player_embedding(self, self_embedded, opp_embedded, stage_oh, rnn_state, seq_lens):

        undelay_input = self.undelay_encoder(
            tf.concat([
                self_embedded, opp_embedded, stage_oh #, time ?
            ], axis=-1)
        )

        undelayed_opp_embedded, next_rnn_state = snt.static_unroll(
            self.undelay_rnn,
            input_sequence=undelay_input,
            initial_state=rnn_state,
            sequence_length=seq_lens
        )

        # put back to logspace binary and categorical
        continuous, binary, categorical = self.split_player_embedding(opp_embedded)
        binary_logits = tf.math.log(binary+1e-8)
        categorical_logits = [
            tf.math.log(sm+1e-8)
            for sm in categorical
        ]

        continuous_stds = tf.fill(continuous.shape, 1e-8)

        opp_embedded_logits = tf.concat(
            [continuous, continuous_stds, binary_logits] + categorical_logits,
            axis=-1
        )

        delta = self.delta_gate(undelayed_opp_embedded)
        new = self.new_gate(undelayed_opp_embedded)
        forget = self.forget_gate(undelayed_opp_embedded)

        predicted_opp_embedded = (1. - forget) * (opp_embedded_logits + delta) + forget * new

        return predicted_opp_embedded, next_rnn_state

    def get_game_embed(self, self_embedded, undelayed_op_embedded, stage_oh, prev_action, rnn_state, seq_lens, single_obs):
        if single_obs:
            prev_action = tf.expand_dims(tf.expand_dims(prev_action, axis=0), axis=0)
        game_embedded = self.game_embeddings(
            tf.concat(
                [self_embedded, undelayed_op_embedded, stage_oh]
            , axis=-1)
        )

        game_embedded = tf.concat(
            [game_embedded, prev_action],
            axis=-1
        )

        lstm_out, next_rnn_state = snt.static_unroll(
            self.partial_obs_lstm,
            input_sequence=game_embedded,
            initial_state=rnn_state,
            sequence_length=seq_lens
        )
        return lstm_out, next_rnn_state

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
            stage_oh = tf.expand_dims(stage_oh, axis=0)

        self_embedded = self.get_player_embedding(
            obs["ground_truth"],
            "1",
            single_obs
        )

        self_delayed_embedded = self.get_player_embedding(
            obs,
            "1",
            single_obs
        )

        opp_delayed_embedded = self.get_player_embedding(
            obs,
            "2",
            single_obs
        )


        predicted_opp_embedded, next_undelay_state = self.get_undelayed_player_embedding(
            self_delayed_embedded, opp_delayed_embedded, stage_oh,
            state[0], seq_lens
        )
        continuous, binary, categoricals = self.split_player_embedding(predicted_opp_embedded, stds=True)
        continuous_dist = GaussianDistribution(continuous)
        self._undelayed_opp_embedded = (continuous_dist, binary, categoricals)

        if "sampled_prediction" in obs:
            # TODO: causes retracing ?
            predicted_opp_embedded = obs["sampled_prediction"]
        else:

            sampled_continuous = continuous_dist.sample()
            clipped_sampled_continuous = tf.clip_by_value(sampled_continuous, self.continuous_low, self.continuous_high)

            sampled_binary = tf.cast(tfp.distributions.Bernoulli(logits=binary).sample(), tf.float32)

            #binary_probs = tf.cast(binary >= 0., dtype=tf.float32) # tf.nn.sigmoid(binary)
            #categorical_probs = [
            #    tf.one_hot(tf.argmax(c, axis=-1), depth=c.shape[-1], dtype=tf.float32)
            #    #tf.nn.softmax(c)
            #    for c in categoricals
            #]
            sampled_categorical = [
               tf.one_hot(CategoricalDistribution(c).sample(), depth=c.shape[-1], dtype=tf.float32)
               #tf.nn.softmax(c)
               for c in categoricals
            ]

            #continuous, _, _ = self.split_player_embedding(opp_delayed_embedded)

            predicted_opp_embedded = tf.stop_gradient(tf.concat([clipped_sampled_continuous, sampled_binary]
                                                                + sampled_categorical, axis=-1))

        # do we want value and policy gradients backpropagate to the opp state prediction ?
        lstm_out, next_lstm_state = self.get_game_embed(
            self_embedded, predicted_opp_embedded, stage_oh, prev_action,
            state[1], seq_lens, single_obs
        )

        action_logits = self._pi_out(lstm_out)
        self._value_logits = self._value_out(lstm_out)
        self.stage_oh = stage_oh

        return ((action_logits, (next_undelay_state, next_lstm_state), predicted_opp_embedded),
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

        opp_embedded = self.get_player_embedding(
            obs["ground_truth"],
            "2",
            single_obs,
        )

        # TODO: test this again
        # should be normalised, therefore this should be ok.
        # advantage_weights = tf.keras.activations.softmax(
        #     tf.math.abs(advantages), axis=[0,1]
        # )

        continuous_true, binary_true, categorical_true = self.split_player_embedding(opp_embedded)
        continuous_predicted_dist, binary_predicted, categorical_predicted = self._undelayed_opp_embedded

        # self.continuous_loss = tf.reduce_sum(advantage_weights*tf.reduce_mean(tf.math.square(continous_predicted - continuous_true),
        #                                                                       axis=-1))
        self.continuous_loss = - tf.reduce_mean(
            #advantage_weights *
            continuous_predicted_dist.logp(continuous_true)
        )

        self.binary_loss = tf.reduce_mean(
            #advantage_weights*
            tf.keras.losses.binary_crossentropy(
            binary_true, binary_predicted,
            from_logits=True,
        ))

        self.tmp1 = continuous_true[0, 0]
        self.tmp2 = continuous_predicted_dist.means[0, 0]

        self.categorical_loss = tf.reduce_mean([
            #tf.reduce_mean(advantage_weights *
                           tf.keras.losses.categorical_crossentropy(
                t, p, from_logits=True
            )
                           #)
            for t, p in zip(categorical_true, categorical_predicted)
        ])

        return self.continuous_loss + self.categorical_loss + self.binary_loss


    def get_metrics(self):
        return {
            "continuous_loss": self.continuous_loss,
            "categorical_loss": self.categorical_loss,
            "binary_loss": self.binary_loss,
            "tmp1": self.tmp1,
            "tmp2": self.tmp2,

        }



