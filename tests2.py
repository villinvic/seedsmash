import time

import zmq

ctx = zmq.Context()
socket = ctx.socket(zmq.PUSH)
socket.bind("ipc://twitchbot")

print("ok push")

for i in range(5):

    print("sending", i)
    socket.send_pyobj(i)
    print('sent', i)
    time.sleep(1)