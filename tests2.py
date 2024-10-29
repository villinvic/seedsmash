import tensorflow as tf
import numpy as np
T, B = 4, 2
k = 3
actions = np.random.randint(0, 6, (T, B))
padded_actions = tf.pad(actions, [[k - 1, 0], [0, 0]])

last_k_actions = tf.transpose(tf.signal.frame(padded_actions, frame_length=k, frame_step=1, axis=0),
                            [0, 2, 1])

print(padded_actions)
print(last_k_actions
      .shape, last_k_actions)