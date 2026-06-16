
from polaris.models import BaseModel

import sonnet as snt
import tree
from gymnasium.spaces import Discrete
import tensorflow as tf

from polaris.experience import SampleBatch

from models.attention_module import AttentionModule
from models.modules import CategoricalValueHead, OpponentPredictionModule, LayerNorm, SymmetryRegulariser, FiLM, \
    ProjectileEmbedder
from models.melee_embedding import MeleeEmbedding, ObservationScope
from models.optimisers import AdamClip

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import RMSprop
import numpy as np
from polaris.models.utils import CategoricalDistribution

class Simple(BaseModel):
    is_recurrent = True


    def __init__(
            self,
            observation_space,
            action_space: Discrete,
            config,
    ):
        super(Simple, self).__init__(
            name="Simple",
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
            excluded = ("character", "character_stats", "projectiles"),
        )

        self.opponent_embedder = MeleeEmbedding(
            observation_space=observation_space,
            observation_type=ObservationScope.OPPONENT,
            excluded=("encoded_action", "projectiles"),
        )

        self.stage_embedder = MeleeEmbedding(
            observation_space=observation_space,
            observation_type=ObservationScope.GLOBAL,
            excluded=("projectiles",),
        )

        self.projectile_embedder = ProjectileEmbedder(config["projectile_mlp_dims"], name="projectile_embedder",
                                                      activation=tf.keras.activations.elu)

        self.core_rnn = snt.GRU(config["core_rnn_dim"], name="core_rnn")


        self._policy_head = snt.nets.MLP(config["policy_head_dims"] + [self.num_outputs], name="policy_head",
                                         activation=tf.keras.activations.elu)

        self._value_head = snt.nets.MLP(config["value_head_dims"] + [1], name="value_head",
                                         activation=tf.keras.activations.elu)

        self.sym_regulariser = SymmetryRegulariser(config["symmetry_weight"], forward=self, action_dist=self.action_dist)

        self.film_layer = FiLM(config["core_rnn_dim"], config["film_layers"], name="stage_film", activation=tf.keras.activations.elu)

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
        x_rnn, next_state = self.core_rnn(x, state)

        x_filmed = self.film_layer(x_rnn, tf.concat(stage_embeds, axis=-1))
        policy_embeds = x_filmed
        # I think this is fine to film only the delayed input for values.
        value_embeds = tf.concat([x_filmed] + self.opponent_embedder(obs, single_obs=True), axis=-1)

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
        x_rnn, _ = snt.static_unroll(
            self.core_rnn,
            input_sequence=x,
            initial_state=state,
            sequence_length=seq_lens
        )
        x_filmed = self.film_layer(x_rnn, tf.concat(stage_embeds, axis=-1))
        policy_embeds = x_filmed
        value_embeds = tf.concat([x_filmed] + self.opponent_embedder(obs), axis=-1)
        return policy_embeds, value_embeds

    def get_initial_state(self):
        return (
            np.zeros((1, self.config["core_rnn_dim"],), dtype=np.float32)
        )

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

        symmetry_loss = self.sym_regulariser(
            curr_action_logits,
            values,
            mask,
            obs["x_swapped"],
            seq_lens,
            prev_action,
            prev_reward,
            state,
        )

        return symmetry_loss

    def get_metrics(self) -> dict:

        d = super().get_metrics()
        d.update(self.sym_regulariser.get_metrics())

        return d

