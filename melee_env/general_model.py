from typing import List, Optional, Dict, Union

import gymnasium as gym
import numpy as np
import tree
from ray.rllib import Policy, SampleBatch
from ray.rllib.models import ModelV2, ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
from ray.rllib.models.tf import RecurrentNetwork
from ray.rllib.models.tf.tf_action_dist import Categorical, DiagGaussian, ActionDistribution, TFActionDistribution

from gymnasium.spaces import Discrete, Tuple, MultiDiscrete

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.tf_utils import flatten_inputs_to_1d_tensor, one_hot
from ray.rllib.utils.typing import AlgorithmConfigDict, TensorType

from melee_env.action_space import ActionSpaceStick

tf1, tf, tfv = try_import_tf()


class EpsilonCategorical(Categorical):

    def __init__(self, inputs, model=None, temperature=1., epsilon=0.02):
        self.epsilon = epsilon
        # weird
        super().__init__(inputs, model, temperature)

        self.probs = tf.math.exp(self.inputs - tf.reduce_max(self.inputs, axis=-1, keepdims=True))
        self.epsilon_probs = (1. - self.epsilon) * self.probs + self.epsilon / tf.cast(self.inputs.shape[-1],
                                                                                       tf.float32)
        self.inputs = tf.math.log(self.epsilon_probs)


class SSBMModelAux(TFModelV2):
    """Implements the `.action_model` branch required above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, delay=None):
        self.num_outputs = action_space.n
        self.size = 128
        self.character_embedding_size = 10
        self.action_state_embedding_size = 70
        self.action_embedding_size = 32
        self.joint_embedding_size = 128
        self.prediction_loss = 0.
        self.continuous_prediction_loss = 0.
        self.action_state_prediction_loss_self = 0.
        self.action_state_prediction_loss_opp = 0.

        self.opp_prediction_range = delay + 1
        self.self_prediction_range = 1

        self.binary_prediction_loss = 0.
        self.seq_lens = None
        self.state = None
        self.unflattened_input_self = None
        self.unflattened_input_opp = None
        self.unflattened_input = None

        self.continuous_prediction_loss_self = None
        self.continuous_prediction_loss_opp = None
        self.binary_prediction_loss_self = None
        self.binary_prediction_loss_opp = None
        self.categorical_losses_self = {}
        self.categorical_losses_opp = {}

        super(SSBMModelAux, self).__init__(
            obs_space, action_space, self.num_outputs, model_config, name
        )

        self.cell_size = 256

        self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
            SampleBatch.ACTIONS, space=self.action_space, shift=-1
        )
        self.view_requirements["OBSERVATION_SELF_FUTURE"] = ViewRequirement(
            SampleBatch.OBS, space=obs_space.original_space, shift=self.self_prediction_range
        )
        self.view_requirements["OBSERVATION_OPP_FUTURE"] = ViewRequirement(
            SampleBatch.OBS, space=obs_space.original_space, shift=self.opp_prediction_range
        )
        # observation = prep.transform(raw_observation)

        previous_action_input = tf.keras.layers.Input(shape=(1,), name="prev_actions", dtype=tf.int32)
        curr_action_input = tf.keras.layers.Input(shape=(1,), name="curr_actions", dtype=tf.int32)

        continuous_inputs = [tf.keras.layers.Input(shape=v.shape, name=k)
                             for k, v in obs_space.original_space["continuous"].items()]

        binary_inputs = [tf.keras.layers.Input(shape=v.shape, name=k)
                         for k, v in obs_space.original_space["binary"].items()]

        categorical_inputs = [tf.keras.layers.Input(shape=v.shape, name=k, dtype=tf.int32)
                              for k, v in obs_space.original_space["categorical"].items()]

        # action_state_embeddings = tf.one_hot(action_state_input, depth=tf.cast(obs_space.original_space[2].high[0], tf.int32)+1, dtype=tf.float32)
        #         # #
        #         # action_state_embeddings = tf.reshape(action_state_embeddings, (-1,
        #         #                                      n_players*self.action_state_embedding_size)
        #         #                                      )
        #         #

        categorical_one_hots = [tf.one_hot(tensor, depth=tf.cast(space.high[0], tf.int32) + 1, dtype=tf.float32)[:, 0]
                                for tensor, space in zip(categorical_inputs, obs_space.original_space["categorical"].values())]

        one_hot_last_actions = tf.one_hot(previous_action_input, depth=action_space.n, dtype=tf.float32)[:, 0]

        obs_input_post_embedding = tf.keras.layers.Concatenate(axis=-1)(
            continuous_inputs + binary_inputs + categorical_one_hots + [one_hot_last_actions]
        )

        state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        fe_0 = tf.keras.layers.Dense(
            self.size,
            name="fe_0",
            activation=tf.nn.silu,
        )(obs_input_post_embedding)

        z = tf.keras.layers.Dense(
            self.size,
            name="z",
            activation=tf.nn.silu,
        )(fe_0)

        timed_input = add_time_dimension(
            padded_inputs=z, seq_lens=seq_in, framework="tf"
        )

        # Preprocess observation with a hidden layer and send to LSTM cell
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            self.cell_size, return_sequences=True, return_state=True, name="lstm"
        )(
            inputs=timed_input,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c],
        )

        lstm_out_no_time = tf.reshape(lstm_out, [-1, self.cell_size])

        one_hot_actions = tf.one_hot(curr_action_input, depth=action_space.n, dtype=tf.float32)[:, 0]

        fe_1_input = tf.keras.layers.Concatenate(axis=-1)([lstm_out_no_time, one_hot_actions])

        za = tf.keras.layers.Dense(
            self.size,
            name="za",
            activation=tf.nn.silu,
        )(fe_1_input)

        # continuous_input, binary_input, character_input, action_state_input,

        total_continuous_self = sum([v.shape[0]  for k, v in obs_space.original_space["continuous"].items()
                                     if k[-1] != "2"] )
        total_continuous_opp = sum([v.shape[0] for k, v in obs_space.original_space["continuous"].items()
                                     if k[-1] == "2"])
        total_binary_self = sum([v.shape[0] for k, v in obs_space.original_space["binary"].items()
                                     if k[-1] != "2"])
        total_binary_opp = sum([v.shape[0] for k, v in obs_space.original_space["binary"].items()
                                     if k[-1] == "2"])

        predicted_continuous_self = tf.keras.layers.Dense(
                total_continuous_self,
                name="predicted_continuous_self",
                activation=None,
            )(za)

        predicted_continuous_opp = tf.keras.layers.Dense(
            total_continuous_opp,
            name="predicted_continuous_opp",
            activation=None,
        )(za)

        predicted_binary_logits_self = tf.keras.layers.Dense(
            total_binary_self,
            name="predicted_binary_self",
            activation=None,
        )(za)

        predicted_binary_logits_opp = tf.keras.layers.Dense(
            total_binary_opp,
            name="predicted_binary_opp",
            activation=None,
        )(za)

        predicted_categorical_logits_self = [
            tf.keras.layers.Dense(
                int(v.high[0]) + 1,
                name="predicted_" + k,
                activation=None,
            )(za)
            for k, v in obs_space.original_space["categorical"].items() if k[-1] != "2"]

        predicted_categorical_logits_opp = [
            tf.keras.layers.Dense(
                int(v.high[0]) + 1,
                name="predicted_" + k,
                activation=None,
            )(za)
            for k, v in obs_space.original_space["categorical"].items() if k[-1] == "2"]


        a1_logits = tf.keras.layers.Dense(
            self.num_outputs,
            name="a1_logits",
            activation=None,
        )(lstm_out_no_time)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
        )(lstm_out_no_time)

        # Base layers
        self.base_model = tf.keras.Model(
            [previous_action_input]
            + binary_inputs + categorical_inputs + continuous_inputs
            + [seq_in, state_in_h, state_in_c],
            [a1_logits, value_out, state_h, state_c])

        self.feature_extractor = tf.keras.Model(
            [previous_action_input]
            + binary_inputs + categorical_inputs + continuous_inputs
            + [seq_in, state_in_h, state_in_c, curr_action_input],
            [predicted_continuous_self, predicted_continuous_opp,
             predicted_binary_logits_self, predicted_binary_logits_opp,
             predicted_categorical_logits_self, predicted_categorical_logits_opp])

    def forward(self, input_dict, state, seq_lens):

        context, self._value_out, h, c = self.base_model(
            [input_dict[SampleBatch.PREV_ACTIONS]] + list(input_dict["obs"].values()) + [seq_lens] + state
        )
        self.seq_lens = seq_lens
        self.state = state

        return tf.reshape(context, [-1, self.num_outputs]), [h, c]
        # context, self._value_out = self.base_model(input_dict["obs"])

        # return tf.reshape(context, [-1, self.num_outputs]), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    @override(ModelV2)
    def get_initial_state(self) -> List[np.ndarray]:
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32)
        ]

    def custom_loss(
            self, policy_loss: TensorType, loss_inputs: Dict[str, TensorType]
    ) -> Union[List[TensorType], TensorType]:
        obs_delay_self = restore_original_dimensions(loss_inputs["OBSERVATION_SELF_FUTURE"], self.obs_space)
        obs_delay_opp = restore_original_dimensions(loss_inputs["OBSERVATION_OPP_FUTURE"], self.obs_space)
        obs = restore_original_dimensions(loss_inputs["obs"], self.obs_space)

        (predicted_continuous_self, predicted_continuous_opp,
         predicted_binary_logits_self, predicted_binary_logits_opp,
         predicted_categorical_logits_self, predicted_categorical_logits_opp
         ) = (
            self.feature_extractor([loss_inputs[SampleBatch.PREV_ACTIONS]] + list(obs.values()) +
                                   [self.seq_lens] + self.state + [loss_inputs[SampleBatch.ACTIONS]])
        )

        continuous_inputs_self = tf.concat(
            [v for k, v in obs_delay_self["continuous"].items() if k[-1] != "2"], axis=-1
        )
        continuous_inputs_opp = tf.concat(
            [v for k, v in obs_delay_opp["continuous"].items() if k[-1] == "2"], axis=-1
        )

        binary_inputs_self = tf.concat(
            [v for k, v in obs_delay_self["binary"].items() if k[-1] != "2"], axis=-1
        )
        binary_inputs_opp = tf.concat(
            [v for k, v in obs_delay_opp["binary"].items() if k[-1] == "2"], axis=-1
        )

        categorical_inputs_self = [(k, v) for k, v in obs_delay_self["categorical"].items() if k[-1] != "2"]

        categorical_inputs_opp = [(k, v) for k, v in obs_delay_opp["categorical"].items() if k[-1] == "2"]

        self.continuous_prediction_loss_self = tf.reduce_mean(
            tf.math.square(predicted_continuous_self - continuous_inputs_self)
        )
        self.continuous_prediction_loss_opp = tf.reduce_mean(
            tf.math.square(predicted_continuous_opp - continuous_inputs_opp)
        )

        self.binary_prediction_loss_self = tf.keras.losses.BinaryCrossentropy(from_logits=True)(binary_inputs_self,
                                                                                                predicted_binary_logits_self)

        self.binary_prediction_loss_opp = tf.keras.losses.BinaryCrossentropy(from_logits=True)(binary_inputs_opp,
                                                                                            predicted_binary_logits_opp)

        for predicted, (name, true) in zip(predicted_categorical_logits_self, categorical_inputs_self):
            self.categorical_losses_self["prediction_loss_"+name+"_self"] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)\
                (true, predicted)

        for predicted, (name, true) in zip(predicted_categorical_logits_opp, categorical_inputs_opp):

            self.categorical_losses_opp["prediction_loss_"+name+"_opp"] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)\
                (true, predicted)

        self.prediction_loss = (self.continuous_prediction_loss_self
                                + self.continuous_prediction_loss_opp
                                + self.binary_prediction_loss_self
                                + self.binary_prediction_loss_opp
                                + sum([loss for loss in self.categorical_losses_self.values()])
                                + sum([loss for loss in self.categorical_losses_opp.values()])
                                )

        return policy_loss #+ self.prediction_loss

    def metrics(self):
        d = super().metrics()
        d.update(
            prediction_loss=self.prediction_loss,
            binary_prediction_loss_self=self.binary_prediction_loss_self,
            binary_prediction_loss_opp=self.binary_prediction_loss_opp,
            continuous_prediction_loss_self=self.continuous_prediction_loss_self,
            continuous_prediction_loss_opp=self.continuous_prediction_loss_opp,
            **self.categorical_losses_self,
            **self.categorical_losses_opp
        )
        return d
