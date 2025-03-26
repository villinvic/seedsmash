
from polaris.models import BaseModel

import sonnet as snt
import tree
from gymnasium.spaces import Discrete
import tensorflow as tf

from polaris.experience import SampleBatch
from models.modules import CategoricalValueHead, OpponentPredictionModule, LayerNorm
from models.melee_embedding import MeleeEmbedding, ObservationScope, EmbeddingGroup, NonLinearPreprocess, MLPEmbedding

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import RMSprop
import numpy as np
from polaris.models.utils import CategoricalDistribution

class Structured(BaseModel):
    is_recurrent = True


    def __init__(
            self,
            observation_space,
            action_space: Discrete,
            config,
    ):
        super(Structured, self).__init__(
            name="Structured",
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )
        self.action_dist = CategoricalDistribution
        self.num_outputs = action_space.n

        # RMSProp, from experience, is much less sample efficient.
        self.optimiser = snt.optimizers.Adam(
            learning_rate=config.lr,
            epsilon=1e-5,
        )

        self.action_state_embedding = snt.Embed(
            vocab_size=int(self.observation_space["categorical"]["action1"].high[0])+1,
            embed_dim=config["action_state_embed_dim"],
            densify_gradients=True,
            name="action_state_embedding"
        )
        self.char_embedding = snt.Embed(
            vocab_size=26,
            embed_dim=config["character_embed_dim"],
            densify_gradients=True,
            name="character_embedding"
        )

        self.stage_embedding = snt.Embed(
            vocab_size=int(self.observation_space["categorical"]["stage"].high[0])+1,
            embed_dim=config["stage_id_embed_dim"],
            densify_gradients=True,
            name="stage_embedding"
        )

        self.percent_embedding = NonLinearPreprocess(
            config["percent_embed_dim"],
            activation=tf.keras.activations.swish,
            name="percent_embedding"
        )

        # define observation groups
        frames = [EmbeddingGroup(
            observation_space=observation_space,
            observation_type=scope,
            observations={
                "continuous": ("iasa", "frames_before_next_hitbox", "action_frame", "hitlag_left", "hitstun_left"),
            },
            delayed=(scope==ObservationScope.OPPONENT)
        )
            for scope in [ObservationScope.SELF, ObservationScope.OPPONENT]
        ]

        positions = [EmbeddingGroup(
            observation_space=observation_space,
            observation_type=scope,
            observations={
                "continuous": ("position", "projectile"),
            },
            delayed=(scope == ObservationScope.OPPONENT)
        )
            for scope in [ObservationScope.SELF, ObservationScope.OPPONENT]
        ]
        positions.append(
            EmbeddingGroup(
                observation_space=observation_space,
                observation_type=ObservationScope.GLOBAL,
                observations={
                    "continuous": ("stage_width", "platforms"),
                    "binary": ("platform_presences",)
                }
            )
        )

        # character, actions
        opp_char_spec = EmbeddingGroup(
            observation_space=observation_space,
            observation_type=ObservationScope.OPPONENT,
            preprocess={
                "character": self.char_embedding
            },
            observations={
                "continuous": ("character_stats",),
                "categorical": ("character",),
            },
            delayed=True,
        )


        stage = EmbeddingGroup(
            observation_space=observation_space,
            observation_type=ObservationScope.GLOBAL,
            preprocess={
                "stage": self.stage_embedding
            },
            observations={
                "continuous": ("stage_width", "platforms"),
                "categorical": ("stage",),
                "binary": ("platform_presences",)
            }
        )

        other = [EmbeddingGroup(
            observation_space=observation_space,
            observation_type=scope,
            preprocess={
                #"percent": self.percent_embedding,
                #"action": self.action_state_embedding,
            },
            observations={
                "continuous": ("percent", "shield_strength", "speed_air_x_self", "speed_y_self", "speed_x_attack",
                               "speed_y_attack", "speed_ground_x_self", "consecutive_hits", #"playstyle",
                               "character_specific"),
                "categorical": ("stock", "invulnerability_type", "jumps_left", "action_type", "action"),
                "binary": ("facing", "on_ground")
            },
            delayed=(scope == ObservationScope.OPPONENT)
        )
            for scope in [ObservationScope.SELF, ObservationScope.OPPONENT]
        ]
        other.append(
            EmbeddingGroup(
                observation_space=observation_space,
                observation_type=ObservationScope.GLOBAL,
                observations={
                    "continuous": ("frame",),
                }
            )
        )

        self.position_embedder = MLPEmbedding(
            config["position_embedding_dims"],
            groups=positions,
            name="position_embedder",
            activation=tf.keras.activations.swish
        )

        self.frame_embedder = MLPEmbedding(
            config["frame_embedding_dims"],
            groups=frames,
            name="frame_embedder",
            activation=tf.keras.activations.swish,
            parallel=True
        )

        self.char_specs_embedder = MLPEmbedding(
            config["char_specs_embedding_dims"],
            groups=[opp_char_spec],
            name="char_specs_embedder",
            activation=tf.keras.activations.swish,
        )

        self.stage_embedder = MLPEmbedding(
            config["stage_embedding_dims"],
            groups=[stage],
            name="stage_embedder",
            activation=tf.keras.activations.swish
        )

        self.other_embedder = MLPEmbedding(
            [],
            groups=other,
            name="other_embedder",
        )

        self.embedders = [
            self.position_embedder,
            self.frame_embedder,
            self.char_specs_embedder,
            self.stage_embedder,
            self.other_embedder
        ]


        self.mlp = snt.nets.MLP(config["mlp_dims"], name="mlp", activate_final=True,
                                activation=tf.keras.activations.swish)
        self.layer_norm = LayerNorm(name="LayerNorm")
        self.rnn = snt.LSTM(config["lstm_dim"], name="rnn")
        self.rnn_decoder = snt.Linear(config["mlp_dims"][-1], w_init=tf.zeros_initializer())

        self._policy_head = snt.nets.MLP(config["policy_head_dims"] + [self.num_outputs], name="policy_head",
                                         activation=tf.keras.activations.swish)
        self._value_head = CategoricalValueHead(
            value_bounds=(-4., 4.), dims=config["value_head_dims"], name="value_head", num_bins=51)

        self.prediction_module = OpponentPredictionModule(
            continuous_components=("frames_before_next_hitbox", "speed_air_x_self", "speed_y_self",
                                   "speed_x_attack", "speed_y_attack", "speed_ground_x_self", "position",
                                   "hitlag_left", "hitstun_left", "projectile", "iasa",
                                   ),
            binary_components=("facing", "on_ground"),
            categorical_components=("invulnerability_type", "jumps_left", "action_type"),
            observation_space=observation_space,
            name="prediction_module"
        )

    def embed(self, obs, single_obs=False, ground_truth=False):
        return [
            embedder(obs, ground_truth=ground_truth, single_obs=single_obs) for embedder in self.embedders
        ]

    def forward_single_action_with_extras(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        final_embeddings, next_state = self.single_input(
            obs,
            prev_action,
            prev_reward,
            state
        )

        #opp_ground_truth = self.opponent_embedder(obs, single_obs=True)
        policy_logits = self._policy_head(final_embeddings)
        extras = {
            SampleBatch.VALUES: tf.squeeze(self._value_head(final_embeddings))
        }
        return policy_logits, next_state, extras

    def forward_single_action(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        # faster call by skipping value inference.

        final_embeddings, next_state = self.single_input(
            obs,
            prev_action,
            prev_reward,
            state
        )

        return self._policy_head(final_embeddings), next_state

    def __call__(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens
    ):
        final_embeddings = self.batch_input(
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens
        )
        policy_logits = self._policy_head(final_embeddings)
        self._values = self._value_head(final_embeddings)
        self.prediction_module(final_embeddings)
        return policy_logits, self._values

    def single_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        prev_action = tf.one_hot(tf.cast(prev_action, tf.int32), depth=self.num_outputs, dtype=tf.float32)
        embeds = self.embed(obs, single_obs=True)

        core = self.mlp(
            tf.concat([prev_action] + embeds, axis=-1)
        )

        lstm_out, next_state = self.rnn(self.layer_norm(core), state)
        decoded = self.rnn_decoder(lstm_out)

        return core + decoded, next_state

    def batch_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens
    ):
        prev_action = tf.one_hot(tf.cast(prev_action, tf.int32), depth=self.num_outputs, dtype=tf.float32)

        embeds = self.embed(obs)

        core = self.mlp(
            tf.concat([prev_action] + embeds, axis=-1)
        )

        lstm_out, next_state = snt.static_unroll(
            self.rnn,
            input_sequence=self.layer_norm(core),
            initial_state=state,
            sequence_length=seq_lens
        )
        decoded = self.rnn_decoder(lstm_out)

        return core + decoded

    def get_initial_state(self):
        return snt.LSTMState(
                hidden=np.zeros((1, self.config["lstm_dim"],), dtype=np.float32),
                cell=np.zeros((1, self.config["lstm_dim"],), dtype=np.float32),
        )

    def critic_loss(self, targets):
        # Categorical value function loss
        return self._value_head.loss(targets)

    def aux_loss(
            self,
            *,
            mask,
            obs,
            **kwargs
    ):
        return self.prediction_module.loss(obs["ground_truth"], mask)

    def get_metrics(self) -> dict:

        d = super().get_metrics()
        d.update(self.prediction_module.get_metrics())

        return d

