from typing import Optional

from polaris.models import BaseModel

import sonnet as snt
import tree
from gymnasium.spaces import Discrete
import tensorflow as tf

from polaris.experience import SampleBatch

from models.modules import LayerNormLSTM, ResLSTMBlock, ResGRUBlock

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import RMSprop
import numpy as np
from polaris.models.utils import CategoricalDistribution



class SS0(BaseModel):
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
        super(SS0, self).__init__(
            name="SS0",
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

        # # categorical
        # # jumps
        self.jumps_embeddings = snt.Embed(tf.cast(self.observation_space["categorical"]["jumps_left1"].high[0], dtype=tf.int32)+1, 4,
                                          densify_gradients=True)
        # # stocks
        self.stocks_embeddings = snt.Embed(tf.cast(self.observation_space["categorical"]["stock1"].high[0], dtype=tf.int32)+1, 2,
                                          densify_gradients=True)
        # action_state
        self.action_state_embeddings = snt.Embed(tf.cast(self.observation_space["categorical"]["action1"].high[0], dtype=tf.int32)+1, 32,
                                          densify_gradients=True)
        # # char
        self.char_embeddings = snt.Embed(tf.cast(self.observation_space["categorical"]["character1"].high[0], dtype=tf.int32)+1, 8,
                                                  densify_gradients=True)
        #
        # # global info (necessary at player level)
        #self.stage_embeddings =  snt.Embed(self.observation_space["categorical"]["stage"].high+1, 4,
        #                                          densify_gradients=True)
        #
        # # continuous
        self.continuous_embeddings = snt.nets.MLP([32, 16], activate_final=True)

        self.player_embeddings = snt.nets.MLP([128], activate_final=True)

        # binary
        # facing
        # on_ground
        # invulnerable
        # buttons
        self.binary_embeddings = snt.nets.MLP([8], activate_final=True)


        # prev_action
        self.prev_action_embeddings = snt.Embed(self.num_outputs, 16, densify_gradients=True)

        # undelay LSTM
        self.undelay_lstm = snt.DeepRNN([ResGRUBlock(128) for _ in range(1)])

        # full game
        self.game_embeddings = snt.nets.MLP([256], activate_final=True,
                                            name="encoder")


        # partial obs LSTM
        self.partial_obs_lstm = snt.DeepRNN([ResLSTMBlock(256) for _ in range(1)])
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

            # continuous_inputs = [tf.expand_dims(continuous_inputs[k], axis=0, name=k) for k in
            #                      self.observation_space["continuous"]]
            #
            # binary_inputs = [tf.cast(tf.expand_dims(binary_inputs[k], axis=0), dtype=tf.float32, name=k) for k in
            #                  self.observation_space["binary"]]
            #
            # categorical_one_hots = [
            #     tf.one_hot(tf.cast(categorical_inputs[k], tf.int32), depth=tf.cast(space.high[0], tf.int32) + 1,
            #                dtype=tf.float32,
            #                name=k)
            #     for k, space in self.observation_space["categorical"].items()
            #     if k not in ["character1"]]  # "action1", "action2", "character2"

            self_continuous_inputs = [tf.expand_dims(continuous_inputs[k], axis=0, name=k) for k in self.observation_space["continuous"]
                                      if "1" in k]
            opp_continuous_inputs = [tf.expand_dims(continuous_inputs[k], axis=0, name=k) for k in self.observation_space["continuous"]
                                     if "2" in k]

            self_binary_inputs = [tf.cast(tf.expand_dims(binary_inputs[k], axis=0), dtype=tf.float32, name=k) for k in
                             self.observation_space["binary"] if "1" in k]
            opp_binary_inputs = [tf.cast(tf.expand_dims(binary_inputs[k], axis=0), dtype=tf.float32, name=k) for k in
                                  self.observation_space["binary"] if "2" in k]


            self_jumps_one_hot = tf.one_hot(tf.cast(categorical_inputs["jumps_left1"], tf.int32),
                                            depth=tf.cast(self.observation_space["categorical"]["jumps_left1"].high[0],
                                                           tf.int32) + 1, dtype=tf.int32)
            opp_jumps_one_hot = tf.one_hot(tf.cast(categorical_inputs["jumps_left2"], tf.int32),
                                            depth=tf.cast(self.observation_space["categorical"]["jumps_left2"].high[0],
                                                           tf.int32) + 1, dtype=tf.int32)
            self_stocks_one_hot = tf.one_hot(tf.cast(categorical_inputs["stock1"], tf.int32),
                                            depth=tf.cast(self.observation_space["categorical"]["stock1"].high[0],
                                                           tf.int32) + 1, dtype=tf.int32)
            opp_stocks_one_hot = tf.one_hot(tf.cast(categorical_inputs["stock2"], tf.int32),
                                            depth=tf.cast(self.observation_space["categorical"]["stock2"].high[0],
                                                           tf.int32) + 1, dtype=tf.int32)
            self_action_state_one_hot = tf.one_hot(tf.cast(categorical_inputs["action1"], tf.int32),
                                            depth=tf.cast(self.observation_space["categorical"]["action1"].high[0],
                                                           tf.int32) + 1, dtype=tf.int32)
            opp_action_state_one_hot = tf.one_hot(tf.cast(categorical_inputs["action2"], tf.int32),
                                            depth=tf.cast(self.observation_space["categorical"]["action2"].high[0],
                                                           tf.int32) + 1, dtype=tf.int32)
            self_char_one_hot = tf.one_hot(tf.cast(categorical_inputs["character1"], tf.int32),
                                                  depth=tf.cast(
                                                      self.observation_space["categorical"]["character1"].high[0],
                                                      tf.int32) + 1, dtype=tf.int32)
            opp_char_one_hot = tf.one_hot(tf.cast(categorical_inputs["character2"], tf.int32),
                                                  depth=tf.cast(
                                                      self.observation_space["categorical"]["character2"].high[0],
                                                      tf.int32) + 1, dtype=tf.int32)

            last_action_one_hot = tf.one_hot(tf.cast(tf.expand_dims(prev_action, axis=0), tf.int32),
                                             depth=self.num_outputs, dtype=tf.int32, name="prev_action_one_hot")

            # obs_input_post_embedding = tf.expand_dims(self.post_embedding_concat(
            #     continuous_inputs + binary_inputs + categorical_one_hots + [last_action_one_hot]), axis=0
            # )
            #
            # # global info
            # # we do not use time for now
            stage_one_hot = tf.one_hot(tf.cast(categorical_inputs["stage"], tf.int32),
                                             depth=tf.cast(self.observation_space["categorical"]["stage"].high[0],
                                                           tf.int32) + 1, dtype=tf.float32, name="stage_one_hot")

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
            # continuous_inputs = [continuous_inputs[k] for k in self.observation_space["continuous"]]
            #
            # binary_inputs = [tf.cast(binary_inputs[k], dtype=tf.float32, name=k) for k in
            #                  self.observation_space["binary"]]
            #
            # categorical_one_hots = [
            #     tf.one_hot(tf.cast(categorical_inputs[k], tf.int32), depth=tf.cast(space.high[0], tf.int32) + 1,
            #                dtype=tf.float32,
            #                name=k)[:, :, 0]
            #     for k, space in self.observation_space["categorical"].items()
            #     if k not in ["character1"]]  # "action1", "action2", "character2"
            self_continuous_inputs = [continuous_inputs[k] for k in self.observation_space["continuous"] if "1" in k]
            opp_continuous_inputs = [continuous_inputs[k] for k in self.observation_space["continuous"] if "2" in k]

            self_binary_inputs = [tf.cast(binary_inputs[k], dtype=tf.float32, name=k) for k in self.observation_space["binary"] if "1" in k]
            opp_binary_inputs = [tf.cast(binary_inputs[k], dtype=tf.float32, name=k) for k in self.observation_space["binary"] if "2" in k]

            self_jumps_one_hot = tf.one_hot(tf.cast(categorical_inputs["jumps_left1"], tf.int32),
                                            depth=tf.cast(self.observation_space["categorical"]["jumps_left1"].high[0],
                                                          tf.int32) + 1, dtype=tf.int32)[:, :, 0]
            opp_jumps_one_hot = tf.one_hot(tf.cast(categorical_inputs["jumps_left2"], tf.int32),
                                           depth=tf.cast(self.observation_space["categorical"]["jumps_left2"].high[0],
                                                         tf.int32) + 1, dtype=tf.int32)[:, :, 0]
            self_stocks_one_hot = tf.one_hot(tf.cast(categorical_inputs["stock1"], tf.int32),
                                             depth=tf.cast(self.observation_space["categorical"]["stock1"].high[0],
                                                           tf.int32) + 1, dtype=tf.int32)[:, :, 0]
            opp_stocks_one_hot = tf.one_hot(tf.cast(categorical_inputs["stock2"], tf.int32),
                                            depth=tf.cast(self.observation_space["categorical"]["stock2"].high[0],
                                                          tf.int32) + 1, dtype=tf.int32)[:, :, 0]
            self_action_state_one_hot = tf.one_hot(tf.cast(categorical_inputs["action1"], tf.int32),
                                                   depth=tf.cast(
                                                       self.observation_space["categorical"]["action1"].high[0],
                                                       tf.int32) + 1, dtype=tf.int32)[:, :, 0]
            opp_action_state_one_hot = tf.one_hot(tf.cast(categorical_inputs["action2"], tf.int32),
                                                  depth=tf.cast(
                                                      self.observation_space["categorical"]["action2"].high[0],
                                                      tf.int32) + 1, dtype=tf.int32)[:, :, 0]
            self_char_one_hot = tf.one_hot(tf.cast(categorical_inputs["character1"], tf.int32),
                                          depth=tf.cast(
                                              self.observation_space["categorical"]["character1"].high[0],
                                              tf.int32) + 1, dtype=tf.int32)[:, :, 0]
            opp_char_one_hot = tf.one_hot(tf.cast(categorical_inputs["character2"], tf.int32),
                                          depth=tf.cast(
                                              self.observation_space["categorical"]["character2"].high[0],
                                              tf.int32) + 1, dtype=tf.int32)[:, :, 0]

            last_action_one_hot = tf.one_hot(tf.cast(prev_action, tf.int32), depth=self.num_outputs, dtype=tf.int32, name="prev_action_one_hot")
            # obs_input_post_embedding = self.post_embedding_concat(
            #     continuous_inputs + binary_inputs + categorical_one_hots
            #     + [last_action_one_hot]
            # )
            #
            stage_one_hot = tf.one_hot(tf.cast(categorical_inputs["stage"], tf.int32),
                                       depth=tf.cast(self.observation_space["categorical"]["stage"].high[0],
                                                     tf.int32) + 1, dtype=tf.float32, name="stage_one_hot")[:, :, 0]


        # stage_embedded = self.stage_embeddings(stage_one_hot)
        #
        self_binary_embedded = self.binary_embeddings(tf.concat(self_binary_inputs, axis=-1))
        opp_binary_embedded = self.binary_embeddings(tf.concat(opp_binary_inputs, axis=-1))
        self_continuous_embedded = self.continuous_embeddings(tf.concat(self_continuous_inputs, axis=-1))
        opp_continuous_embedded = self.continuous_embeddings(tf.concat(opp_continuous_inputs, axis=-1))
        last_action_embedded = self.prev_action_embeddings(last_action_one_hot)

        opp_embedded = self.player_embeddings(tf.concat(
            [
                opp_binary_embedded,
                opp_continuous_embedded,
                stage_one_hot,
                self.jumps_embeddings(opp_jumps_one_hot),
                self.stocks_embeddings(opp_stocks_one_hot),
                self.action_state_embeddings(opp_action_state_one_hot),
                self.char_embeddings(opp_char_one_hot)
             ], axis=-1
        ))

        self_embedded = self.player_embeddings(tf.concat(
            [
                self_binary_embedded,
                self_continuous_embedded,
                stage_one_hot,
                self.jumps_embeddings(self_jumps_one_hot),
                self.stocks_embeddings(self_stocks_one_hot),
                self.action_state_embeddings(self_action_state_one_hot),
                self.char_embeddings(self_char_one_hot)
            ], axis=-1
        ))


        if single_obs:
            opp_embedded = tf.expand_dims(opp_embedded, axis=0)
            self_embedded = tf.expand_dims(self_embedded, axis=0)
            last_action_embedded = tf.expand_dims(last_action_embedded, axis=0)

        undelayed_opp_embedded, undelayed_opp_state = snt.static_unroll(
            self.undelay_lstm,
            input_sequence=opp_embedded,
            initial_state=state[0],
            sequence_length=seq_lens
        )

        self._undelayed_opp_embedded = undelayed_opp_embedded

        obs_input_post_embedding = self.post_embedding_concat(
            [self_embedded, last_action_embedded, undelayed_opp_embedded]
        )

        game_embedded = self.game_embeddings(obs_input_post_embedding)

        # game_embedded = self.game_embeddings(tf.concat(
        #     [
        #         self_embedded, opp_undelayed_embedded, last_action_one_hot
        #     ], axis=-1
        # ))


        lstm_out, states_out = snt.static_unroll(
            self.partial_obs_lstm,
            input_sequence=game_embedded,
            initial_state=state[1],
            sequence_length=seq_lens
        )

        action_logits = self._pi_out(lstm_out)
        self._value_logits = self._value_out(lstm_out)

        return (action_logits, (undelayed_opp_state, states_out)), tf.squeeze(self.compute_predicted_values())


    def get_initial_state(self):
        return (tuple(np.zeros((1, 128), dtype=np.float32) for _ in range(1)), tuple(snt.LSTMState(
                hidden=np.zeros((1, 256,), dtype=np.float32),
                cell=np.zeros((1, 256,), dtype=np.float32),
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
        continuous_inputs = obs["ground_truth"]["continuous"]
        binary_inputs = obs["ground_truth"]["binary"]
        categorical_inputs = obs["ground_truth"]["categorical"]


        opp_continuous_inputs = [continuous_inputs[k] for k in self.observation_space["continuous"] if "2" in k]
        opp_binary_inputs = [tf.cast(binary_inputs[k], dtype=tf.float32, name=k) for k in
                             self.observation_space["binary"] if "2" in k]
        opp_jumps_one_hot = tf.one_hot(tf.cast(categorical_inputs["jumps_left2"], tf.int32),
                                       depth=tf.cast(self.observation_space["categorical"]["jumps_left2"].high[0],
                                                     tf.int32) + 1, dtype=tf.float32)[:, :, 0]
        opp_stocks_one_hot = tf.one_hot(tf.cast(categorical_inputs["stock2"], tf.int32),
                                        depth=tf.cast(self.observation_space["categorical"]["stock2"].high[0],
                                                      tf.int32) + 1, dtype=tf.float32)[:, :, 0]
        opp_action_state_one_hot = tf.one_hot(tf.cast(categorical_inputs["action2"], tf.int32),
                                              depth=tf.cast(
                                                  self.observation_space["categorical"]["action2"].high[0],
                                                  tf.int32) + 1, dtype=tf.float32)[:, :, 0]
        opp_char_one_hot = tf.one_hot(tf.cast(categorical_inputs["character2"], tf.int32),
                                      depth=tf.cast(
                                          self.observation_space["categorical"]["character2"].high[0],
                                          tf.int32) + 1, dtype=tf.float32)[:, :, 0]
        stage_one_hot = tf.one_hot(tf.cast(categorical_inputs["stage"], tf.int32),
                                   depth=tf.cast(self.observation_space["categorical"]["stage"].high[0],
                                                 tf.int32) + 1, dtype=tf.float32, name="stage_one_hot")[:, :, 0]

        opp_embedded = self.player_embeddings(tf.concat(
            opp_binary_inputs +
            opp_continuous_inputs +
            [
                stage_one_hot,
                opp_jumps_one_hot,
                opp_stocks_one_hot,
                opp_action_state_one_hot,
                opp_char_one_hot
            ], axis=-1
        ))

        return tf.reduce_mean(tf.math.square(tf.stop_gradient(opp_embedded) - self._undelayed_opp_embedded))



