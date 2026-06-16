
from polaris.models import BaseModel

import sonnet as snt
import tree
from gymnasium.spaces import Discrete
import tensorflow as tf

from polaris.experience import SampleBatch

from models.attention_module import AttentionModule
from models.modules import CategoricalValueHead, OpponentPredictionModule, LayerNorm
from models.melee_embedding import MeleeEmbedding, ObservationScope

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import RMSprop
import numpy as np
from polaris.models.utils import CategoricalDistribution

class Advanced(BaseModel):
    is_recurrent = True


    def __init__(
            self,
            observation_space,
            action_space: Discrete,
            config,
    ):
        super(Advanced, self).__init__(
            name="Advanced",
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

        self.char_embedding = snt.Embed(
            vocab_size=26,
            embed_dim=config["character_embed_dim"],
            densify_gradients=True,
            name="character_embedding"
        )

        self.stage_embedding = snt.Embed(
            vocab_size=int(self.observation_space["categorical"]["stage"].high[0])+1,
            embed_dim=config["stage_embed_dim"],
            densify_gradients=True,
            name="stage_embedding"
        )

        self.self_embedder = MeleeEmbedding(
            observation_space=observation_space,
            observation_type=ObservationScope.SELF,
            lookups={
                #"action": self.action_state_embedding
            },
            excluded = ("character", "character_stats"),
        )

        self.opponent_embedder = MeleeEmbedding(
            observation_space=observation_space,
            observation_type=ObservationScope.OPPONENT,
            lookups={
                "character": self.char_embedding,
                #"action": self.action_state_embedding
            },
            excluded=("encoded_action",),
        )

        self.stage_embedder = MeleeEmbedding(
            observation_space=observation_space,
            observation_type=ObservationScope.GLOBAL,
            lookups={
                "stage": self.stage_embedding
            }
        )

        self.mlp = snt.nets.MLP(config["mlp_dims"], name="mlp", activate_final=True,
                                activation=tf.keras.activations.swish)

        self.opponent_rnn = snt.LSTM(config["opponent_lstm_dim"], name="opponent_lstm")
        self.core_rnn = snt.GRU(config["core_rnn_dim"], name="core_rnn")

        #self.attention = AttentionModule(config["attention_dim"], config["attention_num_heads"], name="attention")


        self._policy_head = snt.nets.MLP(config["policy_head_dims"] + [self.num_outputs], name="policy_head",
                                         activation=tf.keras.activations.swish)
        self._value_head = CategoricalValueHead(
            value_bounds=(-4., 4.), dims=config["value_head_dims"], name="value_head", num_bins=51)

        self.prediction_module = OpponentPredictionModule(
            continuous_components=("frames_before_next_hitbox", "speed_air_x_self", "speed_y_self",
                                   "speed_x_attack", "speed_y_attack", "speed_ground_x_self", "position",
                                   "platform_distances", "hitlag_left", "hitstun_left", "projectile", "iasa",
                                   "percent", "shield_strength", "action_frame", "character_stats", "projectile",
                                   "consecutive_hits"),
            binary_components=("facing", "on_ground"),
            categorical_components=("invulnerability_type", "jumps_left", "action_type", "attack_state", "action",
                                    "stock", "character"),
            observation_space=observation_space,
            name="prediction_module",
            mlp_dims=config["prediction_mlp_dims"]
        )

        self.layer_norm = LayerNorm(name="LayerNorm")
        self.opponent_rnn_encoder = snt.Linear(config["opponent_lstm_dim"], name="opp_rnn_encoder")
        self.opponent_rnn_decoder = snt.Linear(config["opponent_lstm_dim"], w_init=tf.zeros_initializer(), name="opp_rnn_decoder")

    def forward_single_action_with_extras(
            self,
            obs,
            prev_action,
            prev_reward,
            state,

    ):
        final_embeddings, next_state = self.single_input(
            obs,
            state
        )

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
            state,
    ):
        # faster call by skipping value inference.

        final_embeddings, next_state = self.single_input(
            obs,
            state
        )

        return self._policy_head(final_embeddings), next_state

    def __call__(
            self,
            *,
            obs,
            seq_lens,
            prev_action,
            prev_reward,
            state,

    ):
        final_embeddings = self.batch_input(
            obs,
            state,
            seq_lens
        )
        policy_logits = self._policy_head(final_embeddings)
        self._values = self._value_head(final_embeddings)

        return policy_logits, self._values

    def single_input(
            self,
            obs,
            state
    ):
        stage_embeds = self.stage_embedder(obs, single_obs=True)
        self_embeds = self.self_embedder(obs, single_obs=True)
        self_embeds_delayed = self.self_embedder(obs, delayed=True, single_obs=True)
        opp_embeds = self.opponent_embedder(obs, delayed=True, single_obs=True)

        opp_state = state[0]
        core_state = state[1]

        x = tf.concat(opp_embeds + self_embeds_delayed, axis=-1)
        encoded_x = self.opponent_rnn_encoder(x)
        lstm_out, opp_next_state = self.opponent_rnn(self.layer_norm(encoded_x), opp_state)
        decoded = self.opponent_rnn_decoder(lstm_out)
        res = decoded + encoded_x
        c = tf.concat([res] + stage_embeds, axis=-1)
        undelayed_opp_embeds = self.prediction_module(c)

        core = self.mlp(
            tf.concat(stage_embeds + self_embeds + [undelayed_opp_embeds], axis=-1)
        )

        core_rnn_out, core_next_state = self.core_rnn(
            core,
            core_state,
        )

        return core_rnn_out, (opp_next_state, core_next_state)

    def batch_input(
            self,
            obs,
            state,
            seq_lens
    ):
        stage_embeds = self.stage_embedder(obs)
        self_embeds = self.self_embedder(obs)
        self_embeds_delayed = self.self_embedder(obs, delayed=True)
        opp_embeds = self.opponent_embedder(obs, delayed=True)

        opp_state = state[0]
        core_state = state[1]

        x = tf.concat(opp_embeds + self_embeds_delayed, axis=-1)
        encoded_x = self.opponent_rnn_encoder(x)
        lstm_out, _ = snt.static_unroll(
            self.opponent_rnn,
            input_sequence=self.layer_norm(encoded_x),
            initial_state=opp_state,
            sequence_length=seq_lens
        )
        decoded = self.opponent_rnn_decoder(lstm_out)
        res = decoded + encoded_x
        c = tf.concat([res] + stage_embeds, axis=-1)

        undelayed_opp_embeds = self.prediction_module(c)

        core = self.mlp(
            tf.concat(stage_embeds + self_embeds + [undelayed_opp_embeds], axis=-1)
        )

        core_rnn_out, _ = snt.static_unroll(
            self.core_rnn,
            input_sequence=core,
            initial_state=core_state,
            sequence_length=seq_lens
        )

        return core_rnn_out

    def get_initial_state(self):
        return (
            snt.LSTMState(
                hidden=np.zeros((1, self.config["opponent_lstm_dim"],), dtype=np.float32),
                cell=np.zeros((1, self.config["opponent_lstm_dim"],), dtype=np.float32),
            ),
            np.zeros((1, self.config["core_rnn_dim"],), dtype=np.float32)
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

