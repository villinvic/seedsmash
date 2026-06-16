
from polaris.models import BaseModel

import sonnet as snt
import tree
from gymnasium.spaces import Discrete
import tensorflow as tf

from polaris.experience import SampleBatch

from models.attention_module import AttentionModule
from models.modules import CategoricalValueHead, OpponentPredictionModule, LayerNorm, SymmetryRegulariser, FiLM, \
    ProjectileEmbedder, TransformerLike
from models.melee_embedding import MeleeEmbedding, ObservationScope
from models.optimisers import AdamClip

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import RMSprop
import numpy as np
from polaris.models.utils import CategoricalDistribution

class TransformerLikeModel(BaseModel):
    is_recurrent = True


    def __init__(
            self,
            observation_space,
            action_space: Discrete,
            config,
    ):
        super(TransformerLikeModel, self).__init__(
            name="TransformerLikeModel",
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )
        self.action_dist = CategoricalDistribution
        self.num_outputs = action_space.n

        # RMSProp, from experience, is much less sample efficient.
        self.optimiser = AdamClip(
            learning_rate=config.lr,
            epsilon=1e-5,
            nu=5.
        )

        self.self_embedder = MeleeEmbedding(
            observation_space=observation_space,
            observation_type=ObservationScope.SELF,
            excluded = ("character", "character_stats", "owned_projectiles"),
        )

        self.opponent_embedder = MeleeEmbedding(
            observation_space=observation_space,
            observation_type=ObservationScope.OPPONENT,
            excluded=("encoded_action", "owned_projectiles"),
        )

        self.stage_embedder = MeleeEmbedding(
            observation_space=observation_space,
            observation_type=ObservationScope.GLOBAL,
            excluded=("projectiles",),
        )

        self.projectile_embedder = ProjectileEmbedder(config["projectile_mlp_dims"], name="projectile_embedder",
                                                      activation=tf.keras.activations.gelu)

        self.transformer_like = TransformerLike(
            hidden_size=256,
            num_layers=2,
            ffw_multiplier=2,
            recurrent_layer=snt.LSTM,
            name="core"
        )

        self._policy_head = snt.nets.MLP(config["policy_head_dims"] + [self.num_outputs], name="policy_head",
                                         activation=tf.keras.activations.relu)

        self._value_head = snt.nets.MLP(config["value_head_dims"] + [1], name="value_head",
                                         activation=tf.keras.activations.relu)

    def forward_single_action_with_extras(
            self,
            obs,
            prev_action,
            prev_reward,
            state,

    ):
        policy_embeds, value_embeds, next_state = self.single_input(
            obs,
            state
        )

        policy_logits = self._policy_head(policy_embeds)

        extras = {
            SampleBatch.VALUES: tf.squeeze(self._value_head(value_embeds))
        }
        return policy_logits, next_state, extras


    # UNUSED
    def forward_single_action(
            self,
            obs,
            prev_action,
            prev_reward,
            state,
    ):
        # faster call by skipping value inference.

        policy_embeds, _, next_state = self.single_input(
            obs,
            state
        )

        return self._policy_head(policy_embeds), next_state

    def __call__(
            self,
            *,
            obs,
            seq_lens,
            prev_action,
            prev_reward,
            state,

    ):
        policy_embeds, value_embeds = self.batch_input(
            obs,
            state,
            seq_lens
        )
        policy_logits = self._policy_head(policy_embeds)
        self._values = tf.squeeze(self._value_head(value_embeds))

        return policy_logits, self._values

    def single_input(
            self,
            obs,
            state
    ):
        stage_embeds = self.stage_embedder(obs, single_obs=True)
        self_embeds = self.self_embedder(obs, single_obs=True)
        opp_embeds = self.opponent_embedder(obs, delayed=True, single_obs=True)
        projectiles = self.projectile_embedder(obs)

        x = tf.concat(stage_embeds + opp_embeds + self_embeds + [projectiles], axis=-1)
        x_rnn, next_state = self.transformer_like(x, state)

        policy_embeds = x_rnn
        elo_delta = obs["continuous"]["elo_delta1"]
        value_embeds = tf.concat([x_rnn, elo_delta] + self.opponent_embedder(obs, single_obs=True), axis=-1)

        return policy_embeds, value_embeds, next_state

    def batch_input(
            self,
            obs,
            state,
            seq_lens
    ):
        stage_embeds = self.stage_embedder(obs)
        self_embeds = self.self_embedder(obs)
        opp_embeds = self.opponent_embedder(obs, delayed=True)
        projectiles = self.projectile_embedder(obs)

        x = tf.concat(stage_embeds + opp_embeds + self_embeds + [projectiles], axis=-1)
        x_rnn = self.transformer_like.static_unroll(x, state, seq_lens)
        policy_embeds = x_rnn
        elo_delta = obs["continuous"]["elo_delta1"]
        value_embeds = tf.concat([x_rnn, elo_delta] + self.opponent_embedder(obs), axis=-1)
        return policy_embeds, value_embeds

    def get_initial_state(self):
        return self.transformer_like.get_initial_state()

    def critic_loss(self, targets):
        # Categorical value function loss
        return tf.math.square(targets - self._values)

    def aux_loss(
            self,
            *,
            curr_action_logits,
            values,
            mask,
            obs,
            seq_lens,
            prev_action,
            prev_reward,
            state,
            **kwargs
    ):
        return 0.

    def get_metrics(self) -> dict:

        d = super().get_metrics()
        d.update(self.sym_regulariser.get_metrics())

        return d

