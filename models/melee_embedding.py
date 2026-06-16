from enum import Enum
from typing import Callable, Dict, Tuple, List
import tensorflow as tf
import gymnasium
import sonnet as snt


class ObservationScope(Enum):
    GLOBAL = 0
    SELF = 1
    OPPONENT = 2


def filter_features(space, excluded=(), suffix=""):
    if suffix == "":
        return [f for f in space if f not in excluded and not (f.endswith("1") or f.endswith("2"))]
    excluded = [f"{e}{suffix}" for e in excluded]
    return [f for f in space if f not in excluded and f.endswith(suffix)]


def one_hot_encoding(space: gymnasium.spaces.Box) -> Callable:
    num_classes = int(space.high[0]) + 1
    return lambda feature: tf.one_hot(feature, depth=num_classes, dtype=tf.float32)

def squeeze_one_hot(x, single_obs: bool):
    return tf.squeeze(x, axis=0 if single_obs else 2)


FEATURE_SUFFIXES = {
    ObservationScope.GLOBAL: "",
    ObservationScope.SELF: "1",
    ObservationScope.OPPONENT: "2",
}

class MeleeEmbedding:
    def __init__(
            self,
            observation_space: gymnasium.Space,
            observation_type: ObservationScope,
            lookups: Dict[str, Callable] = {},
            excluded: Tuple[str, ...] = (),
    ):
        feature_suffix = FEATURE_SUFFIXES[observation_type]
        self.continuous_features = filter_features(observation_space["continuous"], excluded, feature_suffix)
        self.binary_features = filter_features(observation_space["binary"], excluded, feature_suffix)

        self.categorical_features = filter_features(observation_space["categorical"], excluded, feature_suffix)

        self.categorical_ops = {
            f: lookups.get(
                f.rstrip(feature_suffix),
                # one hot by default
                one_hot_encoding(observation_space["categorical"][f]))
            for f in self.categorical_features
        }

    def __call__(
            self,
            obs,
            delayed: bool = False,
            single_obs: bool = False,
            categorical_logits : bool = False
    ):
        if not delayed:
            obs = obs["ground_truth"]

        continuous = [tf.cast(obs["continuous"][k], tf.float32) for k in self.continuous_features]
        binary = [tf.cast(obs["binary"][k], tf.float32) for k in self.binary_features]

        # use the list and not the dict for the ordering of features (to be consistent with the prediction module)
        categorical = [squeeze_one_hot(self.categorical_ops[f](tf.cast(obs["categorical"][f], tf.int32)), single_obs) for f in self.categorical_features]

        if categorical_logits:
            binary_logits = [x * tf.math.log(10.) for x in binary]
            categorical_logits = [x * tf.math.log(10. * tf.cast(x.shape[-1], tf.float32)) for x in categorical]

            return continuous + binary + categorical, continuous + binary_logits + categorical_logits

        return continuous + binary + categorical


class NonLinearPreprocess(snt.Module):
    def __init__(
            self,
            embed_dim: int,
            name="NonLinearPreprocess",
            activation=tf.keras.activations.relu,
    ):
        super().__init__(name=name)

        self.encoder = snt.Sequential([
            snt.Linear(
                embed_dim,
                name=f"{name}linear",
            ),
            activation
        ],
            name=f"{name}_encoder"
        )

    def __call__(
            self,
            x
    ):
        return self.encoder(x)



def identity(x):
    return x


class EmbeddingGroup:
    def __init__(
            self,
            observation_space: gymnasium.Space,
            observation_type: ObservationScope,
            preprocess: Dict[str, Callable] = {},
            observations: Dict[str, Tuple[str, ...]] = {},
            delayed: bool = False
    ):
        feature_suffix = FEATURE_SUFFIXES[observation_type]
        self.features = {feature_type: [f + feature_suffix for f in observations.get(feature_type, [])] for feature_type in ["categorical", "continuous", "binary"]}

        # Default one-hot encoding for categorical features
        self.categorical_ops = {
            f: preprocess.get(
                f.rstrip(feature_suffix),
                lambda x: one_hot_encoding(observation_space["categorical"][f])(x)  # Default to one-hot
            )
            for f in self.features["categorical"]
        }

        # Default identity function for continuous and binary features
        self.continuous_ops = {
            f: preprocess.get(f, identity) for f in self.features["continuous"]
        }
        self.binary_ops = {
            f: preprocess.get(f, identity) for f in self.features["binary"]
        }

        self.delayed = delayed

    def __call__(self, obs, ground_truth: bool = False, single_obs: bool = False):
        if ground_truth or not self.delayed:
            obs = obs["ground_truth"]

        # Apply processing to each feature type
        continuous = [self.continuous_ops[k](tf.cast(obs["continuous"][k], tf.float32)) for k in self.features["continuous"]]
        binary = [self.binary_ops[k](tf.cast(obs["binary"][k], tf.float32)) for k in self.features["binary"]]
        categorical = [squeeze_one_hot(self.categorical_ops[f](tf.cast(obs["categorical"][f], tf.int32)), single_obs) for f in self.features["categorical"]]

        return continuous + binary + categorical


class MLPEmbedding(snt.Module):

    def __init__(
            self,
            embed_dims: List[int],
            groups: List[EmbeddingGroup],
            name="MLPEmbedding",
            activation=tf.keras.activations.relu,
            parallel: bool = False,
    ):
        super().__init__(name=name)

        self.encoder = snt.nets.MLP(
            embed_dims,
            activation=activation,
            activate_final=True,
            name=f"{name}_encoder",
        )
        self.groups = groups
        self.parallel = parallel

    def __call__(
            self,
            obs,
            ground_truth: bool = False,
            single_obs: bool = False
    ):

        if self.parallel:
            return tf.concat(
                [self.encoder(tf.concat(group(obs, ground_truth, single_obs), axis=-1)) for group
                 in self.groups],
                axis=-1)
        else:
            embed_groups = []
            for group in self.groups:
                embed_groups += group(obs, ground_truth, single_obs)

            return self.encoder(tf.concat(embed_groups, axis=-1))
