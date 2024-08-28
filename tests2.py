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

class tmp:
    def __init__(self):
        pass


x = defaultdict(tmp)




games = np.concatenate([np.random.randint(6000*28_000, 6000*30_000, 10),
                        np.random.randint(6000*25000, 6000*28000, 5)], axis=0)
games -= np.min(games)

delta = np.max(games) - games
p = delta / delta.sum()

print(delta, p, 1/15)