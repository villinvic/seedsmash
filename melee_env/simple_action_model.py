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


class EpsilonCategorical(Categorical):

    def __init__(self, inputs, model=None, temperature=1., epsilon=0.02):
        self.epsilon = epsilon
        # weird
        super().__init__(inputs, model, temperature)

        self.probs = tf.math.exp(self.inputs - tf.reduce_max(self.inputs, axis=-1, keepdims=True))
        self.epsilon_probs = (1.-self.epsilon) * self.probs + self.epsilon / tf.cast(self.inputs.shape[-1], tf.float32)
        self.inputs = tf.math.log(self.epsilon_probs)


class SSBMModel(TFModelV2):
    """Implements the `.action_model` branch required above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        self.num_outputs = action_space.n
        self.size = 200
        self.character_embedding_size = 10
        self.action_state_embedding_size = 70
        self.action_embedding_size = 32
        self.joint_embedding_size = 64
        super(SSBMModel, self).__init__(
            obs_space, action_space, self.num_outputs, model_config, name
        )

        self.cell_size = 256
        # self.use_prev_action = model_config["lstm_use_prev_action"]
        # self.use_prev_reward = model_config["lstm_use_prev_reward"]

        # Shift rewards for delay (2 frames) TODO: TO TEST : actually to do in the env
        # self.view_requirements[SampleBatch.REWARDS] = \
        #     ViewRequirement(shift=-model_config["delay"])
        self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
            SampleBatch.ACTIONS, space=self.action_space, shift=-1
        )

        #print(obs_space.original_space[0])

        # Inputs
        #print(obs_space)
        # action_state_input = tf.keras.layers.Input(shape=(len(obs_space[2].nvec),), name="action_state_input", dtype=tf.int32)
        # binary_input = tf.keras.layers.Input(shape=(obs_space[1].n,), name="binary_input", dtype=tf.float32)
        # continuous_input = tf.keras.layers.Input(shape=obs_space[0].shape, name="continuous_input", dtype=tf.float32)
        # print(obs_space.original_space[2].nvec)

        previous_action_input = tf.keras.layers.Input(shape=(1,), name="prev_actions", dtype=tf.int32)
        character_input = tf.keras.layers.Input(shape=obs_space.original_space[2].shape[0], name="character_input", dtype=tf.int32)
        action_state_input = tf.keras.layers.Input(shape=obs_space.original_space[3].shape[0],
                                                        name="action_state_input", dtype=tf.int32)
        binary_input = tf.keras.layers.Input(shape=obs_space.original_space[1].n, name="binary_input")
        continuous_input = tf.keras.layers.Input(shape=obs_space.original_space[0].shape[0], name="continuous_input")
        #ctx_input = tf.keras.layers.Input(shape=(self.size,), name="ctx_input")

        n_action_states = int(obs_space.original_space[3].high[0]+1)
        n_players = obs_space.original_space[3].shape[0]
        n_chars = int(obs_space.original_space[2].high[0]+1)

        # We do cross embedding now !
        # TODO : is this fine when trying to learn difficult to reach action states ?
        # such as wall tech, etc
        joint_embedding = tf.keras.layers.Embedding(
            n_chars * n_action_states,
            self.joint_embedding_size, input_length=n_players
            )(action_state_input + n_action_states * character_input)
        joint_embedding = tf.reshape(joint_embedding, (-1, joint_embedding.shape[1]*joint_embedding.shape[2]))

        # and char alone too, because the char also impacts other stuff
        char_embeddings = tf.keras.layers.Embedding(
            n_chars, self.character_embedding_size, input_length=n_players
        )(character_input)
        char_embeddings = tf.reshape(char_embeddings, (-1, char_embeddings.shape[1]*char_embeddings.shape[2]))

        # Also provide the last action we did
        action_embeddings = tf.keras.layers.Embedding(
            self.action_space.n, self.action_embedding_size
        )(previous_action_input)[:, 0]
        #
        # action_state_embeddings = tf.keras.layers.Embedding(
        #     n_action_states, self.action_state_embedding_size, input_length=n_players
        # )(action_state_input)
        # # action_state_embeddings = tf.one_hot(action_state_input, depth=tf.cast(obs_space.original_space[2].high[0], tf.int32)+1, dtype=tf.float32)
        # #
        # action_state_embeddings = tf.reshape(action_state_embeddings, (-1,
        #                                      n_players*self.action_state_embedding_size)
        #                                      )
        #

        # # action_state_embeddings = tf.one_hot(action_state_input, depth=tf.cast(obs_space.original_space[2].high[0], tf.int32)+1, dtype=tf.float32)
        # #

        obs_input_post_embedding = tf.keras.layers.Concatenate(axis=-1)(
            [continuous_input, binary_input,
             char_embeddings,
             joint_embedding,
             action_embeddings
             ]
        )

        hidden_out_0 = tf.keras.layers.Dense(
            self.size,
            name="hidden_out_0",
            activation=tf.nn.tanh,
        )(obs_input_post_embedding)

        hidden_out_1 = tf.keras.layers.Dense(
            self.size,
            name="hidden_out_1",
            activation=tf.nn.tanh,
        )(hidden_out_0)

        # dummy_0 = tf.keras.layers.Dense(
        #     self.num_outputs,
        #     name="hidden_out_2",
        #     activation=None,
        # )(hidden_out_1)
        #
        state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        timed_input = add_time_dimension(
            padded_inputs=hidden_out_1, seq_lens=seq_in, framework="tf"
        )

        # Preprocess observation with a hidden layer and send to LSTM cell
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            self.cell_size, return_sequences=True, return_state=True, name="lstm"
        )(
            inputs=timed_input,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c],
        )

        # context = tf.keras.layers.Dense(
        #     self.num_outputs,
        #     name="hidden0",
        #     activation=tf.nn.tanh,
        # )(lstm_out)

        a1_logits = tf.keras.layers.Dense(
            self.num_outputs,
            name="a1_logits",
            activation=None,
        )(lstm_out)

        # v0 = tf.keras.layers.Dense(
        #     self.size,
        #     name="v0",
        #     activation=tf.nn.tanh,
        # )(lstm_out)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
        )(lstm_out)

        # Base layers
        self.base_model = tf.keras.Model([previous_action_input, continuous_input, binary_input,
                                          character_input,
                                          action_state_input,
                                          seq_in, state_in_h, state_in_c],
                                         [a1_logits, value_out, state_h, state_c])
        # self.base_model = tf.keras.Model([continuous_input, binary_input, action_state_input], [a1_logits, value_out])
        #self.base_model = tf.keras.Model([inputs], [a1_logits, value_out])

    def forward(self, input_dict, state, seq_lens):

        context, self._value_out, h, c = self.base_model(
            [input_dict[SampleBatch.PREV_ACTIONS]] + list(input_dict["obs"]) + [seq_lens] + state
        )

        return tf.reshape(context, [-1, self.num_outputs]), [h, c]
        # context, self._value_out = self.base_model(input_dict["obs"])

        #return tf.reshape(context, [-1, self.num_outputs]), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    @override(ModelV2)
    def get_initial_state(self) -> List[np.ndarray]:
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32)
        ]


# # LSTM wrapper from RLLIB
# class RecurrentSSBMModel(RecurrentNetwork):
#     """An LSTM wrapper serving as an interface for ModelV2s that set use_lstm."""
#
#     def __init__(
#         self,
#         obs_space: gym.spaces.Space,
#         action_space: gym.spaces.Space,
#         num_outputs: int,
#         model_config,
#         name: str,
#     ):
#
#         super(RecurrentSSBMModel, self).__init__(
#             obs_space, action_space, num_outputs, model_config, name
#         )
#
#         # At this point, self.num_outputs is the number of nodes coming
#         # from the wrapped (underlying) model. In other words, self.num_outputs
#         # is the input size for the LSTM layer.
#         # If None, set it to the observation space.
#         print(self.num_outputs, self.size)
#         if self.num_outputs is None:
#             self.num_outputs = int(np.product(self.obs_space.shape))
#
#         self.cell_size = model_config["lstm_cell_size"]
#         self.use_prev_action = model_config["lstm_use_prev_action"]
#         self.use_prev_reward = model_config["lstm_use_prev_reward"]
#
#         self.action_space_struct = get_base_struct_from_space(self.action_space)
#         self.action_dim = 0
#
#         for space in tree.flatten(self.action_space_struct):
#             if isinstance(space, Discrete):
#                 self.action_dim += space.n
#             elif isinstance(space, MultiDiscrete):
#                 self.action_dim += np.sum(space.nvec)
#             elif space.shape is not None:
#                 self.action_dim += int(np.product(space.shape))
#             else:
#                 self.action_dim += int(len(space))
#
#         # Add prev-action/reward nodes to input to LSTM.
#         if self.use_prev_action:
#             self.num_outputs += self.action_dim
#         if self.use_prev_reward:
#             self.num_outputs += 1
#
#         # Define input layers.
#         input_layer = tf.keras.layers.Input(
#             shape=(None, self.num_outputs), name="inputs"
#         )
#
#         # Set self.num_outputs to the number of output nodes desired by the
#         # caller of this constructor.
#         self.num_outputs = num_outputs
#
#         state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name="h")
#         state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name="c")
#         seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)
#
#         # Preprocess observation with a hidden layer and send to LSTM cell
#         lstm_out, state_h, state_c = tf.keras.layers.LSTM(
#             self.cell_size, return_sequences=True, return_state=True, name="lstm"
#         )(
#             inputs=input_layer,
#             mask=tf.sequence_mask(seq_in),
#             initial_state=[state_in_h, state_in_c],
#         )
#
#         # Postprocess LSTM output with another hidden layer and compute values
#         logits = tf.keras.layers.Dense(
#             self.num_outputs, activation=tf.keras.activations.linear, name="logits"
#         )(lstm_out)
#         values = tf.keras.layers.Dense(1, activation=None, name="values")(lstm_out)
#
#         # Create the RNN model
#         self._rnn_model = tf.keras.Model(
#             inputs=[input_layer, seq_in, state_in_h, state_in_c],
#             outputs=[logits, values, state_h, state_c],
#         )
#
#         # Add prev-a/r to this model's view, if required.
#         if model_config["lstm_use_prev_action"]:
#             self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
#                 SampleBatch.ACTIONS, space=self.action_space, shift=-1
#             )
#         if model_config["lstm_use_prev_reward"]:
#             self.view_requirements[SampleBatch.PREV_REWARDS] = ViewRequirement(
#                 SampleBatch.REWARDS, shift=-1
#             )
#
#     @override(RecurrentNetwork)
#     def forward(
#         self,
#         input_dict: Dict[str, TensorType],
#         state: List[TensorType],
#         seq_lens: TensorType,
#     ):
#         assert seq_lens is not None
#         # Push obs through "unwrapped" net's `forward()` first.
#         wrapped_out, _ = self.wrapped_forward(input_dict, [], None)
#
#         # Concat. prev-action/reward if required.
#         prev_a_r = []
#
#         # Prev actions.
#         if self.model_config["lstm_use_prev_action"]:
#             prev_a = input_dict[SampleBatch.PREV_ACTIONS]
#             # If actions are not processed yet (in their original form as
#             # have been sent to environment):
#             # Flatten/one-hot into 1D array.
#             if self.model_config["_disable_action_flattening"]:
#                 prev_a_r.append(
#                     flatten_inputs_to_1d_tensor(
#                         prev_a,
#                         spaces_struct=self.action_space_struct,
#                         time_axis=False,
#                     )
#                 )
#             # If actions are already flattened (but not one-hot'd yet!),
#             # one-hot discrete/multi-discrete actions here.
#             else:
#                 if isinstance(self.action_space, (Discrete, MultiDiscrete)):
#                     prev_a = one_hot(prev_a, self.action_space)
#                 prev_a_r.append(
#                     tf.reshape(tf.cast(prev_a, tf.float32), [-1, self.action_dim])
#                 )
#         # Prev rewards.
#         if self.model_config["lstm_use_prev_reward"]:
#             prev_a_r.append(
#                 tf.reshape(
#                     tf.cast(input_dict[SampleBatch.PREV_REWARDS], tf.float32), [-1, 1]
#                 )
#             )
#
#         # Concat prev. actions + rewards to the "main" input.
#         if prev_a_r:
#             wrapped_out = tf.concat([wrapped_out] + prev_a_r, axis=1)
#
#         # Push everything through our LSTM.
#         input_dict["obs_flat"] = wrapped_out
#         return super().forward(input_dict, state, seq_lens)
#
#     @override(RecurrentNetwork)
#     def forward_rnn(
#         self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType
#     ):
#         model_out, self._value_out, h, c = self._rnn_model([inputs, seq_lens] + state)
#         return model_out, [h, c]
#
#     @override(ModelV2)
#     def get_initial_state(self) -> List[np.ndarray]:
#         return [
#             np.zeros(self.cell_size, np.float32),
#             np.zeros(self.cell_size, np.float32),
#         ]
#
#     @override(ModelV2)
#     def value_function(self) -> TensorType:
#         return tf.reshape(self._value_out, [-1])
#
#
# class RNNSSBM(SSBMModel, RecurrentSSBMModel):
#     pass
#
#
# RNNSSBM._wrapped_forward = SSBMModel.forward
#

