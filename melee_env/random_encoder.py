from typing import List, Optional, Dict

import gymnasium as gym
import numpy as np
import tree
from ray.rllib import Policy, SampleBatch
from ray.rllib.models import ModelV2, ModelCatalog
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


class RandomEncoder(TFModelV2):


    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        self.num_outputs = 64
        self.size = 256

        super(RandomEncoder, self).__init__(
            obs_space, action_space, self.num_outputs, model_config, name
        )

        character_input = tf.keras.layers.Input(shape=obs_space.original_space[2].shape[0], name="character_input", dtype=tf.int32)
        action_state_input = tf.keras.layers.Input(shape=obs_space.original_space[3].shape[0],
                                                        name="action_state_input", dtype=tf.int32)
        binary_input = tf.keras.layers.Input(shape=obs_space.original_space[1].n, name="binary_input")
        continuous_input = tf.keras.layers.Input(shape=obs_space.original_space[0].shape[0], name="continuous_input")

        n_action_states = int(obs_space.original_space[3].high[0]+1)
        n_players = obs_space.original_space[3].shape[0]
        n_chars = int(obs_space.original_space[2].high[0]+1)

        char_embeddings = tf.one_hot(character_input, depth=n_chars, dtype=tf.float32)
        char_embeddings = tf.reshape(char_embeddings, (-1, n_players*n_chars))

        action_state_embeddings = tf.one_hot(action_state_input, depth=n_action_states, dtype=tf.float32)
        action_state_embeddings = tf.reshape(action_state_embeddings, (-1,
                                             n_players*n_action_states)
                                             )

        obs_input_post_embedding = tf.keras.layers.Concatenate(axis=-1)(
            [continuous_input, binary_input,
             char_embeddings,
             action_state_embeddings
             ]
        )

        hidden_out_0 = tf.keras.layers.Dense(
            self.size,
            name="hidden_out_0",
            activation=tf.nn.tanh,
        )(obs_input_post_embedding)

        # hidden_out_1 = tf.keras.layers.Dense(
        #     self.size,
        #     name="hidden_out_1",
        #     activation=tf.nn.tanh,
        # )(hidden_out_0)

        out = tf.keras.layers.Dense(
            self.num_outputs,
            name="out",
            activation=None,
        )(hidden_out_0)

        value = tf.keras.layers.Dense(
            1,
            name="value",
            activation=None,
        )(hidden_out_0)

        self.base_model = tf.keras.Model([continuous_input, binary_input, character_input, action_state_input],
                                         [out, value])

    def forward(self, input_dict, state, seq_lens):

        embedding, self._value_out = self.base_model(input_dict["obs"])

        return tf.reshape(embedding, [-1, self.num_outputs]), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])