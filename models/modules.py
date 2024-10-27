import sonnet as snt
import tensorflow as tf


class LayerNorm(snt.Module):
    """
    https://github.com/vladfi1/slippi-ai/blob/main/slippi_ai/networks.py
    Normalize the mean (to 0) and standard deviation (to 1) of the last dimension.
    We use our own instead of sonnet's because sonnet doesn't allow varying rank.
    """

    def __init__(self):
        super().__init__(name='LayerNorm')

    @snt.once
    def _initialize(self, inputs):
        feature_shape = inputs.shape[-1:]
        self.scale = tf.Variable(tf.ones(feature_shape, dtype=inputs.dtype), name='scale')
        self.bias = tf.Variable(tf.zeros(feature_shape, dtype=inputs.dtype), name='bias')

    def __call__(self, inputs):
        self._initialize(inputs)

        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        inputs -= mean

        stddev = tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True))
        inputs /= stddev

        inputs *= self.scale
        inputs += self.bias

        return inputs



class ResLSTMBlock(snt.RNNCore):

  def __init__(self, residual_size, hidden_size=None, name='ResLSTMBlock'):
    super().__init__(name=name)
    self.layernorm = LayerNorm()
    self.lstm = snt.LSTM(hidden_size or residual_size)
    # initialize the resnet as the identity function
    self.decoder = snt.Linear(residual_size, w_init=tf.zeros_initializer())

  def initial_state(self, batch_size):
    return self.lstm.initial_state(batch_size)

  def __call__(self, residual, prev_state):
    x = residual
    x = self.layernorm(x)
    x, next_state = self.lstm(x, prev_state)
    x = self.decoder(x)
    return residual + x, next_state


class ResGRUBlock(snt.RNNCore):

  def __init__(self, residual_size, hidden_size=None, name='ResGRUBlock'):
    super().__init__(name=name)
    self.layernorm = LayerNorm()
    self.gru = snt.GRU(hidden_size or residual_size)
    # initialize the resnet as the identity function
    self.decoder = snt.Linear(residual_size, w_init=tf.zeros_initializer())

  def initial_state(self, batch_size):
    return self.gru.initial_state(batch_size)

  def __call__(self, residual, prev_state):
    x = residual
    x = self.layernorm(x)
    x, next_state = self.gru(x, prev_state)
    x = self.decoder(x)
    return residual + x, next_state


class LayerNormLSTM(snt.LSTM):
    def __init__(self, units, **kwargs):
        super(LayerNormLSTM, self).__init__(units, **kwargs)
        self.units = units

        # Separate layer norms for each gate
        self.ln_input = LayerNorm()
        self.ln_forget = LayerNorm()
        self.ln_cell = LayerNorm()
        self.ln_output = LayerNorm()
        self.ln_next = LayerNorm()


    def __call__(self, inputs, prev_state):
        """See base class."""

        self._initialize(inputs)

        gates_x = tf.matmul(inputs, self._w_i)
        gates_h = tf.matmul(prev_state.hidden, self._w_h)
        gates = gates_x + gates_h + self.b

        i, f, g, o = tf.split(gates, num_or_size_splits=4, axis=1)
        i = self.ln_input(i)
        f = self.ln_forget(f)
        g = self.ln_cell(g)
        o = self.ln_output(o)

        next_cell = tf.sigmoid(f) * prev_state.cell
        next_cell += tf.sigmoid(i) * tf.tanh(g)
        next_hidden = tf.sigmoid(o) * tf.tanh(self.ln_next(next_cell))

        return next_hidden, snt.LSTMState(hidden=next_hidden, cell=next_cell)


class ResItem(snt.Module):
    def __init__(self, embedder, sampler, loss_func, embedding_size, space, residual_size=32):
        super().__init__()

        self.embedder = embedder
        self.sampler = sampler
        self.size = embedding_size
        self.compute_loss = loss_func
        self.space = space

        self.encoder = snt.Linear(embedding_size)
        self.decoder = snt.Linear(residual_size, w_init=tf.zeros_initializer())

    def predict(self, residual, prev_embedding):
        residual_and_prev = tf.concat([residual, prev_embedding], -1)
        logits = self.encoder(residual_and_prev)
        sample = self.sampler(logits)
        sample_embedding = tf.cast(self.embedder(self, sample), tf.float32)
        residual += self.decoder(sample_embedding)
        sample = tf.squeeze(sample)
        if tf.rank(sample) == 0:
            sample = tf.expand_dims(sample)
        return residual, logits, sample, sample_embedding




