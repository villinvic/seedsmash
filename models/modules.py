from typing import Dict, Tuple

import gymnasium
import sonnet as snt
import tensorflow as tf


class CategoricalValueHead(snt.Module):
    # https://arxiv.org/pdf/2403.03950

    def __init__(
            self,
            num_bins=50,
            value_bounds=(-5., 5.),
            smoothing_ratio=0.75,
            dims = [],
            name="CategoricalValueHead",
    ):
        super().__init__(name=name)
        self.num_bins = num_bins
        self.value_bounds = value_bounds
        self.bin_width = (self.value_bounds[1] - self.value_bounds[0]) / self.num_bins
        self.support = tf.cast(tf.expand_dims(tf.expand_dims(tf.linspace(*self.value_bounds, self.num_bins + 1), axis=0), axis=0),
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


class OpponentPredictionModule(snt.Module):
    """
    Auxiliary head for opponent prediction loss
    """

    def __init__(
            self,
            continuous_components: Tuple[str, ...],
            binary_components: Tuple[str, ...],
            categorical_components: Tuple[str, ...],
            observation_space: gymnasium.Space,
    ):
        self.continuous_components = [f"{c}2" for c in continuous_components]
        self.binary_components = [f"{c}2" for c in binary_components]
        self.categorical_components = [f"{c}2" for c in categorical_components]

        self.continuous_dim = sum(observation_space["continuous"][c].shape[-1] for c in self.continuous_components)
        self.binary_dim = sum(observation_space["binary"][c].shape[-1] for c in self.binary_components)
        self.categorical_dims = [observation_space["categorical"][c].high[0] + 1 for c in self.categorical_components]
        self.categorical_dim = sum(self.categorical_dims)

        self.dims = [self.continuous_dim, self.binary_dim, self.categorical_dim]
        self.embedding_size = self.continuous_dim + self.binary_dim + self.categorical_dim
        self._head = snt.Linear(self.embedding_size, name="head")

        self._pred = None
        self._binary_loss : float | None = None
        self._continuous_loss : float | None = None
        self._categorical_loss : float | None = None


    def split_heads(self, output):
        return tf.split(output, self.dims, axis=-1)

    def split_categoricals(self, categorical_output):
        return tf.split(categorical_output, self.categorical_dims, axis=-1)

    def __call__(self, x):
        self._pred = self._head(x)
        return self._pred

    def loss(self, true, mask) -> float:
        true_continuous = tf.concat([true["continuous"][c] for c in self.continuous_components], axis=-1)
        true_binary = tf.concat([true["binary"][c] for c in self.binary_components], axis=-1)
        true_categoricals = [true["categorical"][c] for c in self.categorical_components]

        pred_continuous, pred_binary, pred_categorical = self.split_heads(self._pred)
        pred_categoricals = self.split_categoricals(pred_categorical)

        self._continuous_loss = tf.reduce_mean(
            tf.boolean_mask(tf.keras.losses.huber(
            true_continuous, mask,
            delta=1.
            ), mask)
        )

        self._binary_loss = tf.reduce_mean(
            tf.boolean_mask(tf.keras.losses.binary_crossentropy(
                true_binary, pred_continuous,
                from_logits=True,
            ), mask)
        )

        self._categorical_loss = tf.reduce_mean([
            tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(
                t, p,
                from_logits=True
            ))
            for t, p in zip(true_categoricals, pred_categoricals)
        ])

        return self._continuous_loss + self._binary_loss + self._categorical_loss

    def get_metrics(self):
        return {
            "Opponent Prediction Continuous Loss": self._continuous_loss,
            "Opponent Prediction Categorical Loss": self._categorical_loss,
            "Opponent Prediction Binary Loss": self._binary_loss,
        }