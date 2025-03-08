
from polaris.models import BaseModel

import sonnet as snt
import tree
from gymnasium.spaces import Discrete
import tensorflow as tf

from polaris.experience import SampleBatch
from models.modules import CategoricalValueHead, OpponentPredictionModule
from models.melee_embedding import MeleeEmbedding, MeleeObservationType

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import RMSprop
import numpy as np
from polaris.models.utils import CategoricalDistribution



class Default(BaseModel):
    is_recurrent = True

    def __init__(
            self,
            observation_space,
            action_space: Discrete,
            config,
    ):
        super(Default, self).__init__(
            name="Default",
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

        self.self_embedder = MeleeEmbedding(
            observation_space=observation_space,
            observation_type=MeleeObservationType.SELF,
            lookups={
                "action": self.action_state_embedding
            },
            excluded = ("character", "character_stats"),
        )

        self.opponent_embedder = MeleeEmbedding(
            observation_space=observation_space,
            observation_type=MeleeObservationType.OPPONENT,
            lookups={
                "character": self.char_embedding,
                "action": self.action_state_embedding
            },
        )

        self.stage_embedder = MeleeEmbedding(
            observation_space=observation_space,
            observation_type=MeleeObservationType.GLOBAL
        )

        self.mlp = snt.nets.MLP(config["mlp_dims"], name="mlp")
        self.rnn = snt.LSTM(config["lstm_dim"], name="rnn")

        self._policy_head = snt.nets.MLP(config["policy_head_dims"] + [self.num_outputs], name="policy_head")
        self._value_head = CategoricalValueHead(value_bounds=(-4., 4.), dims=config["value_head_dims"], name="value_head")

        self.prediction_module = OpponentPredictionModule(
            continuous_components=("frames_before_next_hitbox", "speed_air_x_self", "speed_y_self",
                                   "speed_x_attack", "speed_y_attack", "speed_ground_x_self", "position",
                                   "platform_distances", "hitlag_left", "hitstun_left", "projectile", "iasa",
                                   ),
            binary_components=("facing", "on_ground"),
            categorical_components=("invulnerability_type", "jumps_left", "action_type"),
            observation_space=observation_space
        )

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

        opp_ground_truth = self.opponent_embedder(obs, single_obs=True)
        value_input = tf.concat([final_embeddings, opp_ground_truth], axis=-1)
        policy_logits = self._policy_head(final_embeddings)
        extras = {
            SampleBatch.VALUES: tf.squeeze(self._value_head(value_input))
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
        opp_ground_truth = self.opponent_embedder(obs, single_obs=True)
        value_inputs = tf.concat([final_embeddings, opp_ground_truth], axis=-1)
        self._values = self._value_head(value_inputs)
        self.prediction_module(final_embeddings)
        return policy_logits, self._values

    def single_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        prev_action = tf.one_hot(tf.cast(prev_action, tf.int32), depth=self.num_outputs)
        stage_embeds = self.stage_embedder(obs, single_obs=True)
        self_embeds = self.self_embedder(obs, single_obs=True)
        opp_embeds = self.opponent_embedder(obs, delayed=True, single_obs=True)

        core = self.mlp(
            tf.concat([prev_action] + stage_embeds + self_embeds + opp_embeds, axis=-1)
        )
        lstm_out, next_state = self.rnn(core, state)
        residual = lstm_out + core

        return residual, next_state

    def batch_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens
    ):
        prev_action = tf.one_hot(tf.cast(prev_action, tf.int32), depth=self.num_outputs)
        stage_embeds = self.stage_embedder(obs)
        self_embeds = self.self_embedder(obs)
        opp_embeds = self.opponent_embedder(obs, delayed=True)

        core = self.mlp(
            tf.concat([prev_action] + stage_embeds + self_embeds + opp_embeds, axis=-1)
        )

        lstm_out, next_state = snt.static_unroll(
            self.rnn,
            input_sequence=core,
            initial_state=state,
            sequence_length=seq_lens
        )
        residual = lstm_out + core

        return residual

    def get_initial_state(self):
        return snt.LSTMState(
                hidden=np.zeros((1, self.config["rnn_dim"],), dtype=np.float32),
                cell=np.zeros((1, self.config["rnn_dim"],), dtype=np.float32),
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

