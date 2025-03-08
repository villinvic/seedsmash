from enum import Enum
from typing import Callable, Dict, Tuple
import tensorflow as tf
import gymnasium


class MeleeObservationType(Enum):
    GLOBAL = 0
    SELF = 1
    OPPONENT = 2


def filter_features(space, excluded, suffix=None):
    return [f for f in space if f not in excluded and (suffix is None or f.endswith(suffix))]


def one_hot_encoding(space: gymnasium.spaces.Box) -> Callable:
    num_classes = int(space.high[0]) + 1
    return lambda feature: tf.one_hot(tf.cast(feature, tf.int32), depth=num_classes)


def squeeze_one_hot(x, single_obs: bool):
    return tf.squeeze(x, axis=0 if single_obs else 2)


FEATURE_SUFFIXES = {
    MeleeObservationType.GLOBAL: None,
    MeleeObservationType.SELF: "1",
    MeleeObservationType.OPPONENT: "2",
}


class MeleeEmbedding:
    def __init__(
            self,
            observation_space: gymnasium.Space,
            observation_type: MeleeObservationType,
            lookups: Dict[str, Callable] = {},
            excluded: Tuple[str, ...] = (),
    ):
        feature_suffix = FEATURE_SUFFIXES[observation_type]
        self.continuous_features = filter_features(observation_space["continuous"], excluded, feature_suffix)
        self.binary_features = filter_features(observation_space["binary"], excluded, feature_suffix)

        categorical_features = filter_features(observation_space["categorical"], excluded, feature_suffix)
        self.categorical_ops = {
            f: lookups.get(
                f.rstrip(feature_suffix),
                # one hot by default
                one_hot_encoding(observation_space["categorical"][f]))
            for f in categorical_features
        }

    def __call__(
            self,
            obs,
            delayed: bool = False,
            single_obs: bool = False
    ):
        if not delayed:
            obs = obs["ground_truth"]

        continuous = [obs["continuous"][k] for k in self.continuous_features]
        binary = [obs["binary"][k] for k in self.binary_features]
        categorical = [squeeze_one_hot(op(obs["categorical"][f]), single_obs) for f, op in self.categorical_ops.items()]

        return continuous + binary + categorical
