import tensorflow as tf
import numpy as np

x = np.full((43,), fill_value=1/43)




for eps in np.linspace(0, 10, 100):
    xn = x.copy()
    xn[0] += eps

    ex = np.exp(xn)

    xn = ex / np.sum(ex)

    ent = - np.sum(
        xn * np.log(1e-8 + xn)
    )
    print(eps, ent, np.max(xn), np.min(xn))