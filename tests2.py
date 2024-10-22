import numpy as np
gamma = 0.993
def discount(r):
    return r*gamma

def discount_combo(r, c):
    c *= 1 / 0.999
    if c < 0:
        c = 0.

    if 50 > r > 0:
        c+= 1

    return c


hit_rewards = np.zeros((20*5), dtype=np.float32)
hit_rewards[[0, 20, 40]] = 10.
hit_rewards[95] = 100.


c = []
ct = 0
for r in hit_rewards:
    ct = discount_combo(r, ct)
    c.append(ct)

boosted_rewards = hit_rewards * (1 + np.array(c)/5)

value = 0.
values = []
for r in reversed(boosted_rewards):
    value = discount(value) + r
    values.append(value)

print(c)
print(boosted_rewards)
print(values)
