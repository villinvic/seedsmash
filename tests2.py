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
import time

import numpy as np

import numpy as np


def surround_with_true(arr, n):
    # Step 1: Get the indices of all True values in the input array
    true_indices = np.flatnonzero(arr)

    if len(true_indices) == 0:
        return arr  # No True values, return original array

    # Step 2: Generate the range of indices to be set to True
    start_indices = np.maximum(true_indices - n, 0)  # Ensures index doesn't go negative
    end_indices = np.minimum(true_indices + n, len(arr) - 1)  # Ensures index doesn't exceed array length

    # Step 3: Create an empty boolean array
    output = np.zeros_like(arr, dtype=bool)

    # Step 4: Use NumPy broadcasting to mark ranges as True
    for start, end in zip(start_indices, end_indices):
        output[start:end + 1] = True

    return output

arr = np.random.choice(2, size=30, p=[0.9, 0.1])
print(arr)
t = time.time()
result = surround_with_true(arr, 0)
print(result)
print(time.time()-t)

