import tensorflow as tf
import sonnet as snt

class AttentionModule(snt.Module):
    def __init__(
            self,
            hidden_dim=128,
            num_attention_heads=4,
            name="AttentionModule"
    ):
        super().__init__(name=name)
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads, key_dim=hidden_dim // num_attention_heads,
            name=f"{name}_attention"
        )

    def __call__(self, opponent_embedding, ctx):
        """
        opponent_embedding: (T, B, D)   # Opponent's embedding post rnn
        own_past: (T, B, D)           # Our delayed state
        stage_info: (T, B, D)         # Stage features
        """
        # Concatenate own state + stage info as context

        # Use opponent past as query, our state & stage as keys/values
        attended_context = self.attention(query=opponent_embedding, key=ctx, value=ctx)  # (B, 1, D)
        return attended_context