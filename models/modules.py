from typing import Dict, Tuple, List, Optional

import gymnasium
import numpy as np
import sonnet as snt
import tensorflow as tf
import tree

from models.melee_embedding import filter_features, ObservationScope
from polaris_melee.action_space import x_swapped_actions


class CategoricalValueHead(snt.Module):
    # https://arxiv.org/pdf/2403.03950

    def __init__(
            self,
            num_bins=50,
            value_bounds=(-5., 5.),
            smoothing_ratio=0.75,
            dims=[],
            name="CategoricalValueHead",
    ):
        super().__init__(name=name)
        self.num_bins = num_bins
        self.value_bounds = value_bounds
        self.bin_width = (self.value_bounds[1] - self.value_bounds[0]) / self.num_bins
        self.support = tf.cast(
            tf.expand_dims(tf.expand_dims(tf.linspace(*self.value_bounds, self.num_bins + 1), axis=0), axis=0),
            tf.float32)
        self.centers = (self.support[0, :, :-1] + self.support[0, :, 1:]) / 2.
        sigma = smoothing_ratio * self.bin_width
        self.sqrt_two_sigma = tf.math.sqrt(2.) * sigma

        self.value_out = snt.nets.MLP(dims + [self.num_bins], name="head")
        self._logits = None

    def __call__(
            self,
            input_
    ):
        self._logits = self.value_out(input_)
        return tf.reduce_sum(self.centers * tf.nn.softmax(self._logits),
                             axis=-1)

    def targets_to_probs(self, targets):
        # this may occur on rare occasion that targets are outside of the set interval.
        targets = tf.clip_by_value(targets, *self.value_bounds)

        cdf_evals = tf.math.erf(
            (self.support - tf.expand_dims(targets, axis=-1))
            / self.sqrt_two_sigma
        )
        z = cdf_evals[:, :, -1:] - cdf_evals[:, :, :1]
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


class OpponentPredictionModule(snt.Module):
    """
    Auxiliary head for opponent prediction loss
    """

    def __init__(
            self,
            observation_space: gymnasium.Space,
            mlp_dims: List[int],
            name: str,
            excluded = (),
    ):
        super().__init__(name=name)
        self.continuous_components = filter_features(observation_space["continuous"], excluded, suffix= "2")
        self.binary_components = filter_features(observation_space["binary"], excluded, suffix="2")
        self.categorical_components = filter_features(observation_space["categorical"], excluded, suffix="2")

        self.continuous_dims = [observation_space["continuous"][c].shape[-1] for c in self.continuous_components]
        self.continuous_dim = sum(self.continuous_dims)
        self.binary_dims = [observation_space["binary"][c].shape[-1] for c in self.binary_components]

        self.binary_dim = sum(self.binary_dims)
        self.categorical_dims = [int(observation_space["categorical"][c].high[0] + 1) for c in
                                 self.categorical_components]
        self.categorical_dim = sum(self.categorical_dims)

        continuous_loss_weight = self.continuous_dim
        binary_loss_weight = self.binary_dim
        categorical_loss_weight = len(self.categorical_dims)

        total_weight = continuous_loss_weight + binary_loss_weight + categorical_loss_weight
        self.continuous_loss_weight = continuous_loss_weight / total_weight
        self.binary_loss_weight = binary_loss_weight / total_weight
        self.categorical_loss_weight = categorical_loss_weight / total_weight

        self.dims = [self.continuous_dim, self.binary_dim, self.categorical_dim]
        self.embedding_size = self.continuous_dim + self.binary_dim + self.categorical_dim
        self._body = snt.nets.MLP(mlp_dims, name="prediction_mlp", activation=tf.nn.elu, activate_final=True)

        self._delta = snt.Linear(self.embedding_size, name="delta", w_init=tf.zeros_initializer())
        self._new = snt.Linear(self.embedding_size, name="new")
        self._forget = snt.Linear(self.embedding_size, name="forget", w_init=tf.zeros_initializer())

        self._pred_continuous = None
        self._pred_binary_logits = None
        self._pred_categoricals_logits = None

        self._binary_loss: float | None = None
        self._continuous_loss: float | None = None
        self._categorical_loss: float | None = None

    def split_heads(self, output):
        return tf.split(output, self.dims, axis=-1)

    def split_categoricals(self, categorical_output):
        return tf.split(categorical_output, self.categorical_dims, axis=-1)

    def reconstruct(self, ground_truth_dict):

        continuous = tf.split(self._pred_continuous, self.continuous_dims, axis=-1)
        binary = tf.split(self._pred_binary_logits, self.binary_dims, axis=-1)

        reconstructed = {}

        reconstructed["continuous"] = {
            f: (continuous[i], ground_truth_dict["continuous"][f]) for i, f in enumerate(self.continuous_components)
        }
        reconstructed["binary"] = {
            f: (binary[i], ground_truth_dict["binary"][f]) for i, f in enumerate(self.binary_components)
        }
        reconstructed["categorical"] = {
            f: (tf.argmax(self._pred_categoricals_logits[i], axis=-1), ground_truth_dict["categorical"][f]) for i, f in enumerate(self.categorical_components)
        }

        self._reconstructed = reconstructed


    def __call__(self, x, delayed_x):
        x = self._body(x)
        delta = self._delta(x)
        new = self._new(x)
        forget = tf.sigmoid(self._forget(x) + 1.)

        pred = forget * (delayed_x + delta) + (1. - forget) * new

        self._forget_p= forget
        self._pred_continuous, self._pred_binary_logits, pred_categorical_logits = self.split_heads(pred)

        self._pred_categoricals_logits = self.split_categoricals(pred_categorical_logits)
        pred_binary = tf.math.exp(self._pred_binary_logits)
        pred_categoricals = [
            tf.nn.softmax(categorical_logit, axis=-1)
            for categorical_logit in self._pred_categoricals_logits
        ]

        return tf.concat([self._pred_continuous, pred_binary] + pred_categoricals,axis=-1)

    def loss(self, true, mask) -> float:
        true_continuous = tf.concat(
            [tf.cast(true["continuous"][c], dtype=tf.float32) for c in self.continuous_components], axis=-1)
        true_binary = tf.concat([tf.cast(true["binary"][c], dtype=tf.float32) for c in self.binary_components], axis=-1)
        true_categoricals = [tf.cast(true["categorical"][c], dtype=tf.int32) for c in self.categorical_components]


        continuous_loss = tf.boolean_mask(
            tf.math.square(true_continuous- self._pred_continuous)
        , mask)
        mean_continuous_loss = tf.reduce_mean(continuous_loss)
        self._mean_continuous_loss = mean_continuous_loss
        self._max_continuous_loss = tf.reduce_max(continuous_loss)

        binary_loss = tf.boolean_mask(tf.keras.losses.binary_crossentropy(
                true_binary, self._pred_binary_logits,
                from_logits=True,
        ), mask)
        # binary_loss = tf.boolean_mask(
        #         tf.math.square(true_binary - self._pred_binary_logits) # testing this
        #     , mask)
        mean_binary_loss = tf.reduce_mean(binary_loss)
        self._mean_binary_loss = mean_binary_loss
        self._max_binary_loss = tf.reduce_max(binary_loss)

        categorical_loss = [
            tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(
                t, p,
                from_logits=True
            ), mask)
            for t, p in zip(true_categoricals, self._pred_categoricals_logits)
        ]
        mean_categorical_loss = tf.reduce_mean(categorical_loss)
        self._mean_categorical_loss = mean_categorical_loss
        self._max_categorical_loss = tf.reduce_max(categorical_loss)

        not_forget_p = 1. - self._forget_p
        forget_entropy = -tf.reduce_mean(
                self._forget_p * tf.math.log(self._forget_p + 1e-8)
                + not_forget_p * tf.math.log(not_forget_p + 1e-8)
        )
        self._prediction_forget_gate_entropy = forget_entropy

        return (
            self.continuous_loss_weight * mean_continuous_loss
            + self.binary_loss_weight * mean_binary_loss
            + self.categorical_loss_weight * mean_categorical_loss

            # force the forget gate to be more decisive
            + forget_entropy * 0.
        )

    def get_metrics(self):
        return {
            "Opponent Prediction Max Continuous Loss": self._max_continuous_loss,
            "Opponent Prediction Max Categorical Loss": self._max_categorical_loss,
            "Opponent Prediction Max Binary Loss": self._max_binary_loss,
            "Opponent Prediction Mean Continuous Loss": self._mean_continuous_loss,
            "Opponent Prediction Mean Categorical Loss": self._mean_categorical_loss,
            "Opponent Prediction Mean Binary Loss": self._mean_binary_loss,
            "Opponent Prediction Forget Entropy": self._prediction_forget_gate_entropy,
            #"rec": self._reconstructed
        }


# From https://github.com/vladfi1/slippi-ai/blob/main/slippi_ai
class LayerNorm(snt.Module):

    def __init__(self, name="LayerNorm"):
        super().__init__(name=name)

    @snt.once
    def _initialize(self, inputs):
        feature_shape = inputs.shape[-1:]
        self.scale = tf.Variable(
            tf.ones(feature_shape, dtype=inputs.dtype),
            name='scale')
        self.bias = tf.Variable(
            tf.zeros(feature_shape, dtype=inputs.dtype),
            name='bias')

    def __call__(self, inputs):
        self._initialize(inputs)

        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        inputs -= mean

        stddev = tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True))
        inputs /= stddev

        inputs *= self.scale
        inputs += self.bias

        return inputs


class FiLM(snt.Module):

    def __init__(
            self,
            dim,
            hidden_sizes=[128],
            name="FiLM",
            activation=tf.nn.relu,
    ):
        super().__init__(name=name)

        # initialise with no transformation
        self._mlp = snt.nets.MLP(hidden_sizes, activate_final=True, activation=activation, name="film_mlp")
        self._scale = snt.Linear(dim, w_init=tf.zeros_initializer(), b_init=tf.ones_initializer(), name="scale")
        self._shift = snt.Linear(dim, w_init=tf.zeros_initializer(), name="shift")
        self.activation = activation

    def __call__(self, x, ctx):
        y = self._mlp(ctx)
        scale = self._scale(y)
        shift = self._shift(y)

        return self.activation(x * scale + shift)


class SymmetryRegulariser:

    def __init__(
            self,
            coeff: float,
            forward,
            action_dist,

    ):
        self.coeff = coeff
        self.forward = forward
        self.action_dist = action_dist


    def __call__(
            self,
            original_action_logits,
            original_values,
            mask,
            obs,
            seq_lens,
            prev_action,
            prev_reward,
            state,
    ):
        # we do not swap prev_action here as we do not use them in the model.

        state = tree.map_structure(
            lambda s: tf.zeros_like(s),
            state
        )

        logits, values = self.forward(
            obs=obs,
            seq_lens=seq_lens,
            prev_action=prev_action,
            prev_reward=prev_reward,
            state=state,
        )

        swapped_logits = tf.gather(logits, x_swapped_actions, axis=-1)

        original_action_dist = self.action_dist(original_action_logits)
        self._kl_loss = tf.reduce_mean(tf.boolean_mask(original_action_dist.kl(swapped_logits), mask))
        self._v_loss = tf.reduce_mean(tf.boolean_mask(tf.math.square(original_values - values), mask))

        return (self._kl_loss + 0.5 * self._v_loss) * self.coeff


    def get_metrics(self):
        return {}
        return {
            "Symmetry KL Loss": self._kl_loss,
            "Symmetry Value Loss": self._v_loss
        }


class ProjectileEmbedder(snt.Module):
    # Need to use a custom attention module to embed projectiles strategically
    # we do not delay projectiles for now (will delay them once I see agents play well around them)

    def __init__(
            self,
            mlp_dims: List[int],
            name: str,
            activation: tf.nn.relu
    ):
        super().__init__(name=name)

        self.context = ["position1", "position2"] # make it self-attention too ?
        self.components = ["projectiles", "owned_projectiles1", "owned_projectiles2"]
        self.mlp_dims = mlp_dims
        self.activation = activation
        self.projectile_mlp = snt.nets.MLP(mlp_dims, activation=activation, name="projectile_mlp")
        #self.weight_mlp = snt.nets.MLP(self.mlp_dims + [1], activation=self.activation, name="weight_mlp")

    def __call__(
            self,
            obs,
    ):
        # reshape projectiles into shape (T, B, num projectiles, F)
        projectiles = tf.concat([obs["ground_truth"]["continuous"][c] for c in self.components], axis=-1)
        shape = tf.shape(projectiles)
        leading = shape[:-1]
        projectiles = tf.reshape(projectiles, tf.concat([leading,[-1, 7]], axis=0))
        num_projectiles = tf.shape(projectiles)[-2]

        ctx = tf.expand_dims(tf.concat([obs["ground_truth"]["continuous"][c] for c in self.context], axis=-1), axis=-2)
        ctx_feature_dim = tf.shape(ctx)[-1]
        ctx = tf.broadcast_to(ctx, tf.concat([leading, [num_projectiles, ctx_feature_dim]], axis=0))

        projectiles = tf.concat([projectiles, ctx], axis=-1)
        projectile_embeds = self.projectile_mlp(projectiles)

        #weights = tf.nn.softmax(self.weight_mlp(projectiles), axis=-2)

        # Max pooling
        return tf.reduce_max(projectile_embeds, axis=-2)


class ResBlock(snt.Module):
    # from https://github.com/vladfi1/slippi-ai/blob/7c2ed1b16a98fac11a25358280c465ba4bcd86db/slippi_ai/networks.py#L371

    def __init__(
      self,
      residual_size: int,
      hidden_size: Optional[int] = None,
      activation=tf.nn.relu,
      name='ResBlock'):
        super().__init__(name=name)
        self.block = snt.Sequential([
        # https://openreview.net/forum?id=B1x8anVFPr recommends putting the layernorm here
        LayerNorm(name=f"{name}_layer_norm"),
        snt.Linear(hidden_size or residual_size, name=f"{name}_linear_1"),
        activation,
        # initialize the resnet as the identity function
        snt.Linear(residual_size, w_init=tf.zeros_initializer(), name=f"{name}_linear_2"),
        ])

    def __call__(self, residual):
        return residual + self.block(residual)


class TransformerLike(snt.Module):
    # from https://github.com/vladfi1/slippi-ai/blob/7c2ed1b16a98fac11a25358280c465ba4bcd86db/slippi_ai/networks.py#L371

    def __init__(
      self,
      hidden_size: int = 128,
      num_layers: int = 1,
      ffw_multiplier: int = 4,
      recurrent_layer = snt.LSTM,
      activation=tf.nn.gelu,
      name='TransformerLike',
    ):
        super().__init__(name=name)
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._ffw_multiplier = ffw_multiplier
        # We need to encode for the first residual
        self._encoder = snt.Linear(hidden_size, name='encoder')

        self._rnns = [
            recurrent_layer(hidden_size, name=f"{name}_rnn_{i}")
            for i in range(num_layers)
        ]
        self._ffws = [
            ResBlock(
                hidden_size, hidden_size * ffw_multiplier,
                activation=activation,
                name=f"{name}_ffw_{i}"
            )
            for i in range(num_layers)
        ]

    def __call__(self, x, states):

        x = self._encoder(x)
        next_states = ()
        for rnn, ffw, s in zip(self._rnns, self._ffws, states):
            o, ns = rnn(x, s)
            next_states += (ns,)
            x = ffw(x + o)
        return x, next_states


    def static_unroll(self, x, states, seq_lens):
        x = self._encoder(x)
        for rnn, ffw, s in zip(self._rnns, self._ffws, states):
            o, _ = snt.static_unroll(
                rnn,
                input_sequence=x,
                initial_state=s,
                sequence_length=seq_lens
            )
            x = ffw(x + o)
        return x

    def get_initial_state(self):
        return tuple(
            rnn.initial_state(1)
            for rnn in self._rnns
        )





