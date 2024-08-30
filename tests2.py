# import time
#
# import zmq
#
# ctx = zmq.Context()
# socket = ctx.socket(zmq.PUSH)
# socket.bind("ipc://twitchbot")
#
# print("ok push")
#
# for i in range(5):
#
#     print("sending", i)
#     socket.send_pyobj(i)
#     print('sent', i)
#     time.sleep(1)
import numpy as np
from collections import defaultdict

from seedsmash2.utils import ActionStateValues

probs = ActionStateValues({})

batch_size = 256
action_probs = np.float32(np.square(np.arange(len(probs.probs))))
action_probs /= action_probs.sum()

print(action_probs)