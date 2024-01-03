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


class SeedSmashModel(TFModelV2):
    """Implements the `.action_model` branch required above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 parent_name=None, distillation_weight=1.):

        # learns to play as one character, against many characters

        self.parent_name = parent_name
        self.distillation_weight = distillation_weight

        self.num_outputs = action_space.n
        self.size = 120
        self.character_embedding_size = 5
        self.action_state_embedding_size = 32
        self.joint_embedding_size = 42
        self.last_action_embedding_size = 10

        self.distillation_loss = None

        super(SeedSmashModel, self).__init__(
            obs_space, action_space, self.num_outputs, model_config, name
        )

        self.cell_size = 250

        self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
            SampleBatch.ACTIONS, space=self.action_space, shift=-1
        )

        previous_action_input = tf.keras.layers.Input(shape=(1,), name="prev_actions", dtype=tf.int32)

        continuous_inputs = [tf.keras.layers.Input(shape=v.shape, name=k)
                             for k, v in obs_space.original_space["continuous"].items()]

        binary_inputs = [tf.keras.layers.Input(shape=v.shape, name=k)
                         for k, v in obs_space.original_space["binary"].items()]

        categorical_inputs = [tf.keras.layers.Input(shape=v.shape, name=k, dtype=tf.int32)
                              for k, v in obs_space.original_space["categorical"].items()]

        named_categorical_inputs = {k: c for k, c in zip(obs_space.original_space["categorical"], categorical_inputs)}

        categorical_one_hots = [tf.one_hot(tensor, depth=tf.cast(space.high[0], tf.int32) + 1, dtype=tf.float32)[:, 0]
                                for tensor, (name, space) in zip(categorical_inputs, obs_space.original_space["categorical"].items())
                                if name not in ["action", "character"]]

        # TODO :
        #   This is only for 1v1
        # We do not observe our own char
        action_state_input_self = named_categorical_inputs["action1"]
        action_state_input_opp = named_categorical_inputs["action2"]
        char_input = named_categorical_inputs["character2"]

        n_chars = int(obs_space.original_space["categorical"]["character1"].high)+1
        n_action_states = int(obs_space.original_space["categorical"]["action1"].high)+1

        #one_hot_last_actions = tf.one_hot(previous_action_input, depth=action_space.n, dtype=tf.float32)[:, 0]

        last_action_embedding = tf.keras.layers.Embedding(
            action_space.n,
            self.last_action_embedding_size,
        )(previous_action_input)[:, 0]

        action_state_char_joint_embedding = tf.keras.layers.Embedding(
            n_chars * n_action_states,
            self.joint_embedding_size,
            )(action_state_input_opp + n_action_states * char_input)[:, 0]

        self_action_state_embedding = tf.keras.layers.Embedding(
            n_action_states,
            self.action_state_embedding_size
            )(action_state_input_self)[:, 0]

        char_embedding = tf.keras.layers.Embedding(
            n_chars,
            self.character_embedding_size
            )(char_input)[:, 0]


        obs_input_post_embedding = tf.keras.layers.Concatenate(axis=-1)(
            continuous_inputs + binary_inputs + categorical_one_hots +
            [last_action_embedding, self_action_state_embedding, action_state_char_joint_embedding, char_embedding]
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
            self.cell_size, return_sequences=True, return_state=True, name="lstm",
        )(
            inputs=timed_input,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c],
        )

        lstm_out_no_time = tf.reshape(lstm_out, [-1, self.cell_size])

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

    def forward(self, input_dict, state, seq_lens):

        context, self._value_out, h, c = self.base_model(
            [input_dict[SampleBatch.PREV_ACTIONS]] + list(input_dict["obs"].values()) + [seq_lens] + state
        )

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
        # Todo : Add some distillation loss if we have a parent
        #   In this example, MyPolicy is a custom policy class that takes an additional trainer argument in
        #   its constructor. When you create a MyPolicy instance, you pass the Trainer to it, and the policy stores a
        #   reference to the trainer in its self.trainer attribute.

        # if self.distillation_weight > 1e-3:
        #     other_policy = self.trainer.get_(self.parent_name)
        #
        #     self.distillation_loss = self.compute_distillation_loss(other_policy)
        #
        #     self.distillation_weight *= 0.99
        #
        #     policy_loss += self.distillation_loss * self.distillation_weight

        return policy_loss

    def metrics(self):
        d = super().metrics()
        if self.distillation_loss is not None:
            d.update(
                distillation_loss=self.distillation_loss,
                distillation_loss_weighted=self.distillation_weight*self.distillation_loss
            )
        return d
