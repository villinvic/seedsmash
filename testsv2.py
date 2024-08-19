
import numpy as np

for i in range(100):
    halflife = i / 100. * (18 - 5) + 5
    print(i, halflife, np.exp(-np.log(2) / (halflife * 20)))