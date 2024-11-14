from typing import List, Optional

import gymnasium as gym
import numpy as np
from ray.rllib import Policy, SampleBatch
from ray.rllib.models import ModelV2, ModelCatalog
from ray.rllib.models.tf import RecurrentNetwork
from ray.rllib.models.tf.tf_action_dist import Categorical, DiagGaussian, ActionDistribution, TFActionDistribution

from gymnasium.spaces import Discrete, Tuple

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import AlgorithmConfigDict, TensorType

from melee_env.action_space import ActionSpaceStick

tf1, tf, tfv = try_import_tf()


class EpsilonCategorical(Categorical):

    def __init__(self, inputs, model=None, temperature=1., epsilon=0.01):
        self.epsilon = epsilon
        # weird
        super().__init__(inputs, model, temperature)

        self.probs = tf.math.exp(self.inputs - tf.reduce_max(self.inputs, axis=-1, keepdims=True))
        self.epsilon_probs = (1.-self.epsilon) * self.probs + self.epsilon / tf.cast(self.inputs.shape[-1], tf.float32)
        self.inputs = tf.math.log(self.epsilon_probs)


class ClippedGaussian(DiagGaussian):

    def __init__(
            self,
            inputs: List[TensorType],
            model: ModelV2,
            *,
            action_space: Optional[gym.spaces.Space] = None
    ):
        mean, log_std = tf.split(inputs, 2, axis=1)
        self.mean = tf.clip_by_value(mean, -1.05, 1.05)
        self.log_std = tf.clip_by_value(log_std, -1e8, 0)
        self.std = tf.exp(self.log_std)
        # Remember to squeeze action samples in case action space is Box(shape)
        self.zero_action_dim = action_space and action_space.shape == ()
        super().__init__(inputs, model)


class BinaryAutoregressiveDistribution(ActionDistribution):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    def deterministic_sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.deterministic_sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        return (a1, a2)

    def sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        return (a1, a2)

    def logp(self, actions):
        a1, a2 = actions[:, 0], actions[:, 1]
        a1_vec = tf.cast(tf.one_hot(tf.cast(a1, tf.int32), 9), tf.float32)

        a1_logits, a2_logits = self.model.action_model([self.inputs, a1_vec])
        return Categorical(a1_logits).logp(a1) + Categorical(a2_logits).logp(a2)

    def sampled_action_logp(self):
        return self._action_logp

    def entropy(self):
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())

        a1 = a1_dist.entropy()
        a2 = a2_dist.entropy()
        #tf.print("ENTROPY", a1, a2)
        return a1 + a2

    def kl(self, other):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))

        return a1_terms + a2_terms

    def _a1_distribution(self):
        BATCH = tf.shape(self.inputs)[0]
        a1_logits, _ = self.model.action_model([self.inputs, tf.zeros((BATCH, 9))])
        a1_dist = Categorical(a1_logits)
        return a1_dist

    def _a2_distribution(self, a1):
        a1_vec = tf.cast(tf.one_hot(a1, 9), tf.float32)
        _, a2_logits = self.model.action_model([self.inputs, a1_vec])
        a2_dist = Categorical(a2_logits)
        return a2_dist

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 128  # controls model output feature vector size


class TestDummyAutoreg(TFActionDistribution):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    def __init__(
            self,
            inputs: List[TensorType],
            model: ModelV2,
            *,
            action_space: Optional[gym.spaces.Space] = None
    ):
        self.a1_logits, self.a2_logits = tf.split(inputs, 2, axis=1)

        super().__init__(inputs, model)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.deterministic_sample()

        # Return the action tuple.
        return (a1, a2)

    @override(ActionDistribution)
    def logp(self, actions: TensorType) -> TensorType:
        if isinstance(actions, tuple):
            a1, a2 = actions
        else:
            a1, a2 = actions[0], actions[1]
        return Categorical(self.a1_logits).logp(a1) + Categorical(self.a2_logits).logp(a2)

    @override(ActionDistribution)
    def entropy(self):
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())

        a1 = a1_dist.entropy()
        a2 = a2_dist.entropy()
        #tf.print("ENTROPY", a1, a2)
        return a1 + a2

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))

        return a1_terms + a2_terms

    @override(TFActionDistribution)
    def _build_sample_op(self) -> TensorType:
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()

        # Return the action tuple.
        return (a1, a2)

    def _a1_distribution(self):
        a1_dist = Categorical(self.a1_logits)
        return a1_dist

    def _a2_distribution(self, a1):
        a2_dist = Categorical(self.a2_logits)
        return a2_dist

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return action_space[0].n + action_space[1].n  # controls model output feature vector size


class TestDummyAutoreg2(TFActionDistribution):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    def __init__(
            self,
            inputs: List[TensorType],
            model: ModelV2,
            *,
            action_space: Optional[gym.spaces.Space] = None
    ):

        BATCH = tf.shape(inputs)[0]
        self.a1_logits, self.a2_logits_unsure = model.action_model([inputs, tf.zeros((BATCH, 9))])

        super().__init__(inputs, model)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.deterministic_sample()

        # Return the action tuple.
        return (a1, a2)

    @override(ActionDistribution)
    def logp(self, actions: TensorType) -> TensorType:
        if isinstance(actions, tuple):
            a1, a2 = actions
        else:
            a1, a2 = actions[:, 0], actions[:, 1:]

        a1_vec = tf.cast(tf.one_hot(tf.cast(a1, tf.int32), 9), tf.float32)
        _, self.a2_logits = self.model.action_model([self.inputs, a1_vec])

        return Categorical(self.a1_logits).logp(a1) + DiagGaussian(self.a2_logits, self.model).logp(a2)

    @override(ActionDistribution)
    def entropy(self):
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())

        a1 = a1_dist.entropy()
        a2 = a2_dist.entropy()
        #tf.print("ENTROPY", a1, a2)
        return a1 + a2 * 0.0001

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))

        return a1_terms + a2_terms

    @override(TFActionDistribution)
    def _build_sample_op(self) -> TensorType:
        a1_dist = self._a1_distribution()
        a1 = tf.cast(tf.clip_by_value(tf.cast(a1_dist.sample(), tf.float32), clip_value_min=0, clip_value_max=8),
                     tf.int64)

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()

        # Return the action tuple.
        return (a1, a2)

    def _a1_distribution(self):
        a1_dist = Categorical(self.a1_logits)
        return a1_dist

    def _a2_distribution(self, a1):
        a1_vec = tf.cast(tf.one_hot(tf.cast(a1, tf.int32), 9), tf.float32)
        _, self.a2_logits = self.model.action_model([self.inputs, a1_vec])
        a2_dist = DiagGaussian(self.a2_logits, self.model)
        return a2_dist

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 117  # controls model output feature vector size


class TestDummyAutoreg3(TFActionDistribution):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    def __init__(
            self,
            inputs: List[TensorType],
            model: ModelV2,
            *,
            action_space: Optional[gym.spaces.Space] = None
    ):

        BATCH = tf.shape(inputs)[0]
        self.dummy_1 = tf.zeros((BATCH, 16))
        self.last_action_dummy = tf.fill((BATCH,), 9)
        #self.dummy_2 = tf.zeros((BATCH, 9))

        self.a1_logits, self.a2_logits_unsure, self.a3_logits_unsure = model.action_model(
            [inputs, self.dummy_1])

        super().__init__(inputs, model)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()
        a1_vec = tf.cast(tf.one_hot(tf.cast(a1, tf.int32), 16), tf.float32)

        # Sample a2 conditioned on a1.
        a2_dist, a3_dist = self._a2_a3_distribution(a1_vec)
        a2 = a2_dist.deterministic_sample()
        a3 = a3_dist.deterministic_sample()
        #a2_vec = tf.cast(tf.one_hot(tf.cast(a2, tf.int32), 9), tf.float32)

        #a3_dist = self._a3_distribution(a1_vec)
        #a3 = a3_dist.deterministic_sample()

        # Return the action tuple.
        return (a1, a2, a3)

    @override(ActionDistribution)
    def logp(self, actions: TensorType) -> TensorType:
        if isinstance(actions, tuple):
            a1, a2, a3 = actions
        else:
            a1, a2, a3 = actions[:, 0], actions[:, 1], actions[:, 2]

        a1_vec = tf.cast(tf.one_hot(tf.cast(a1, tf.int32), 16), tf.float32)
        #a2_vec = tf.cast(tf.one_hot(tf.cast(a2, tf.int32), 9), tf.float32)
        _, self.a2_logits, self.a3_logits = self.model.action_model([self.inputs, a1_vec])

        return (
                Categorical(self.a1_logits).logp(a1)
                + Categorical(self.a2_logits).logp(a2)
                + Categorical(self.a3_logits).logp(a3)
        )

    @override(ActionDistribution)
    def entropy(self):
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()
        a1_vec = tf.cast(tf.one_hot(tf.cast(a1, tf.int32), 16), tf.float32)
        a2_dist, a3_dist = self._a2_a3_distribution(a1_vec)
        #a2_vec = tf.cast(tf.one_hot(tf.cast(a2, tf.int32), 9), tf.float32)
        #a3_dist = self._a3_distribution(a1_vec)

        h1 = a1_dist.entropy()
        h2 = a2_dist.entropy()
        h3 = a3_dist.entropy()

        #tf.print("ENTROPY", a1, a2, a3)
        return h1 + h2 + h3 #* tf.exp(a2_dist.logp(self.last_action_dummy)) # stop grads !

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a1_vec = tf.cast(tf.one_hot(tf.cast(a1, tf.int32), 16), tf.float32)

        a2_dist, a3_dist = self._a2_a3_distribution(a1_vec)
        other_a2_dist, other_a3_dist = other._a2_a3_distribution(a1_vec)
        a2_terms = a2_dist.kl(other_a2_dist)
        a3_terms = a3_dist.kl(other_a3_dist)

        #a2 = a2_dist.sample()
        #a2_vec = tf.cast(tf.one_hot(tf.cast(a2, tf.int32), 9), tf.float32)
        #a3_terms = self._a3_distribution(a1_vec).kl(other._a3_distribution(a1_vec))

        return a1_terms + a2_terms + a3_terms

    @override(TFActionDistribution)
    def _build_sample_op(self) -> TensorType:
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        # Sample a2 conditioned on a1.
        a1_vec = tf.cast(tf.one_hot(tf.cast(a1, tf.int32), 16), tf.float32)
        a2_dist, a3_dist = self._a2_a3_distribution(a1_vec)
        a2 = a2_dist.sample()
        a3 = a3_dist.sample()

        #a2_vec = tf.cast(tf.one_hot(tf.cast(a2, tf.int32), 9), tf.float32)
        # a3_dist = self._a3_distribution(a1_vec)
        # a3 = a3_dist.sample()

        # Return the action tuple.
        return (a1, a2, a3)

    def _a1_distribution(self):
        a1_dist = Categorical(self.a1_logits)
        return a1_dist

    def _a2_a3_distribution(self, a1_vec):
        _, self.a2_logits, self.a3_logits = self.model.action_model([self.inputs, a1_vec])
        a2_dist = Categorical(self.a2_logits)
        a3_dist = Categorical(self.a3_logits)

        return a2_dist, a3_dist


    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 128  # controls model output feature vector size


class TestDummyAutoreg4(TFActionDistribution):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    def __init__(
            self,
            inputs: List[TensorType],
            model: ModelV2,
            *,
            action_space: Optional[gym.spaces.Space] = None
    ):
        self.stick_pos = ActionSpaceStick().controller_states

        BATCH = tf.shape(inputs)[0]
        #self.dummy_1 = tf.zeros((BATCH, 16))
        self.dummy_1 = tf.zeros((BATCH, 2))
        #self.dummy_2 = tf.zeros((BATCH, 9))

        self.a1_logits, self.a2_logits_unsure = model.action_model(
            [inputs, self.dummy_1])

        super().__init__(inputs, model)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()
        #a1_vec = tf.cast(tf.one_hot(tf.cast(a1, tf.int32), 16), tf.float32)
        a1_vec = tf.gather_nd(self.stick_pos, tf.cast(tf.expand_dims(a1, axis=-1), tf.int32))

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1_vec)
        a2 = a2_dist.deterministic_sample()

        # Return the action tuple.
        return tf.concat([tf.expand_dims(a1, axis=1), tf.expand_dims(a2, axis=1)], axis=-1)

    @override(ActionDistribution)
    def logp(self, actions: TensorType) -> TensorType:

        if isinstance(actions, tuple):
            a1, a2 = actions
        elif len(actions.shape) > 1:
            a1, a2 = tf.split(actions, 2, -1)
            a1 = a1[:, 0]
            a2 = a2[:, 0]
        else:
            a1, a2 = tf.split(actions, 2)

        a1_vec = tf.gather_nd(self.stick_pos, tf.cast(tf.expand_dims(a1, axis=-1), tf.int32))
        #a1_vec = tf.cast(tf.one_hot(tf.cast(a1, tf.int32), 16), tf.float32)
        _, self.a2_logits = self.model.action_model([self.inputs, a1_vec])

        return (
                EpsilonCategorical(self.a1_logits).logp(a1)
                + EpsilonCategorical(self.a2_logits).logp(a2)
        )

    @override(ActionDistribution)
    def entropy(self):
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()
        a1_vec = tf.gather_nd(self.stick_pos, tf.cast(tf.expand_dims(a1, axis=-1), tf.int32))
        #a1_vec = tf.cast(tf.one_hot(tf.cast(a1, tf.int32), 16), tf.float32)
        a2_dist = self._a2_distribution(a1_vec)
        #a2_vec = tf.cast(tf.one_hot(tf.cast(a2, tf.int32), 9), tf.float32)
        #a3_dist = self._a3_distribution(a1_vec)

        h1 = a1_dist.entropy()
        h2 = a2_dist.entropy()

        #tf.print("ENTROPY", a1, a2)
        return h1 + h2

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a1_vec = tf.gather_nd(self.stick_pos, tf.cast(tf.expand_dims(a1, axis=-1), tf.int32))
        #a1_vec = tf.cast(tf.one_hot(tf.cast(a1, tf.int32), 16), tf.float32)

        a2_dist = self._a2_distribution(a1_vec)
        other_a2_dist = other._a2_distribution(a1_vec)
        a2_terms = a2_dist.kl(other_a2_dist)

        return a1_terms + a2_terms

    @override(TFActionDistribution)
    def _build_sample_op(self) -> TensorType:
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        # Sample a2 conditioned on a1.
        a1_vec = tf.gather_nd(self.stick_pos, tf.cast(tf.expand_dims(a1, axis=-1), tf.int32))
        #a1_vec = tf.cast(tf.one_hot(tf.cast(a1, tf.int32), 16), tf.float32)
        a2_dist = self._a2_distribution(a1_vec)
        a2 = a2_dist.sample()

        # Return the action tuple.
        return tf.concat([tf.expand_dims(a1, axis=1), tf.expand_dims(a2, axis=1)], axis=-1)

    def _a1_distribution(self):
        a1_dist = EpsilonCategorical(self.a1_logits)
        return a1_dist

    def _a2_distribution(self, a1_vec):
        _, self.a2_logits = self.model.action_model([self.inputs, a1_vec])
        a2_dist = EpsilonCategorical(self.a2_logits)

        return a2_dist


    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 256  # controls model output feature vector size


class TestDummyAutoreg1(TFActionDistribution):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    def __init__(
            self,
            inputs: List[TensorType],
            model: ModelV2,
            *,
            action_space: Optional[gym.spaces.Space] = None
    ):
        self.stick_pos = ActionSpaceStick().controller_states

        self.a1_logits = model.action_model(
            [inputs])

        super().__init__(inputs, model)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        # Return the action tuple.
        return a1

    @override(ActionDistribution)
    def logp(self, actions: TensorType) -> TensorType:

        return (
                Categorical(self.a1_logits).logp(actions)
        )

    @override(ActionDistribution)
    def entropy(self):
        a1_dist = self._a1_distribution()

        h1 = a1_dist.entropy()
        return h1

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        return a1_terms

    @override(TFActionDistribution)
    def _build_sample_op(self) -> TensorType:
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        return a1

    def _a1_distribution(self):
        a1_dist = Categorical(self.a1_logits)
        return a1_dist


    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 128  # controls model output feature vector size


class AutoregressiveActionModel(TFModelV2):
    """Implements the `.action_model` branch required above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        self.size=128
        self.character_embedding_size = 5
        self.action_state_embedding_size = 64
        self.cell_size = 256
        self.num_outputs = self.cell_size

        super(AutoregressiveActionModel, self).__init__(
            obs_space, action_space, self.num_outputs, model_config, name
        )

        action_state_input = tf.keras.layers.Input(shape=obs_space.original_space[2].shape[0],
                                                   name="action_state_input", dtype=tf.int32)

        binary_input = tf.keras.layers.Input(shape=obs_space.original_space[1].n, name="binary_input")
        continuous_input = tf.keras.layers.Input(shape=obs_space.original_space[0].shape[0], name="continuous_input")

        #character_input = tf.keras.layers.Input(shape=obs_space[2].shape, name="character_input", dtype=tf.int32)

        a1_input = tf.keras.layers.Input(shape=(2,), name="a1_input", dtype=tf.float32)
        ctx_input = tf.keras.layers.Input(shape=(self.cell_size,), name="ctx_input")


        # Output of the model (normally 'logits', but for an autoregressive
        # dist this is more like a context/feature layer encoding the obs)

        # char_embeddings = tf.keras.layers.Embedding(
        #     obs_space[2].high[0], self.character_embedding_size, input_length=obs_space[2].shape[0]
        # )(character_input)
        # char_embeddings = tf.one_hot(character_input, depth=obs_space[2].high[0]+1)


        # char_embeddings = tf.reshape(char_embeddings, (-1,
        #                                                char_embeddings.shape[1]*char_embeddings.shape[2]) )

        # action_state_embeddings = tf.keras.layers.Embedding(
        #     obs_space[3].high[0]+1, self.action_state_embedding_size, input_length=obs_space[3].shape[0]
        # )(action_state_input)
        # action_state_embeddings = tf.one_hot(action_state_input, depth=obs_space[3].high[0]+1)
        #
        # action_state_embeddings = tf.reshape(action_state_embeddings, (-1,
        #                                      action_state_embeddings.shape[1]*action_state_embeddings.shape[2])
        #                                      )
        #
        # obs_input_post_embedding = tf.keras.layers.Concatenate(axis=-1)(
        #     [continuous_input, binary_input,
        #      #char_embeddings,
        #      action_state_embeddings]
        # )

        n_action_states = tf.cast(obs_space.original_space[2].high[0] + 1, tf.int32)
        n_players = obs_space.original_space[2].shape[0]
        action_state_embeddings = tf.keras.layers.Embedding(
            n_action_states, self.action_state_embedding_size, input_length=n_players
        )(action_state_input)
        action_state_embeddings = tf.reshape(action_state_embeddings, (-1,
                                                                       n_players * self.action_state_embedding_size)
                                             )
        obs_input_post_embedding = tf.keras.layers.Concatenate(axis=-1)(
            [continuous_input, binary_input,
             # char_embeddings,
             action_state_embeddings
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


        # P(a1 | obs)
        # a1_hidden = tf.keras.layers.Dense(
        #     32,
        #     name="a1_hidden",
        #     activation=tf.nn.tanh,
        # )(ctx_input)

        a1_logits = tf.keras.layers.Dense(
            action_space[0].n,
            name="a1_logits",
            activation=None,
        )(ctx_input)

        # P(a2 | a1)
        # --note: typically you'd want to implement P(a2 | a1, obs) as follows:
        a2_context = tf.keras.layers.Concatenate(axis=1)(
            [ctx_input, a1_input])

        a2_hidden = tf.keras.layers.Dense(
            action_space[1].n,
            name="a2_hidden",
            activation=tf.nn.tanh,
        )(a2_context)

        a2_logits = tf.keras.layers.Dense(
            action_space[1].n,
            name="a2_logits",
            activation=None,
        )(a2_hidden)

        # V(s)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
        )(lstm_out)

        # Base layers
        self.base_model = tf.keras.Model([continuous_input, binary_input,
                                          #character_input,
                                          action_state_input,
                                          seq_in, state_in_h, state_in_c],
                                         [lstm_out, value_out, state_h, state_c])
        # self.base_model.summary()

        # Autoregressive action sampler
        # self.action_model = tf.keras.Model(
        #     [ctx_input, a1_input], [a1_logits, a2_logits]
        # )
        self.action_model = tf.keras.Model(
            [ctx_input, a1_input], [a1_logits, a2_logits]
        )
        # self.action_model.summary()

        #self.register_variables(self.base_model.variables)
        #self.register_variables(self.action_model.variables)

    def forward(self, input_dict, state, seq_lens):
        #context, self._value_out = self.base_model(input_dict["obs_flat"])
        context, self._value_out, h, c = self.base_model(list(input_dict["obs"]) + [seq_lens] + state)

        return tf.reshape(context, [-1, self.num_outputs]), [h, c]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    @override(ModelV2)
    def get_initial_state(self) -> List[np.ndarray]:
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32)
        ]