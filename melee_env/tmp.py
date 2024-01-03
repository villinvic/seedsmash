import numpy as np


# def filter_last_damage_before_death(damage_array, death_array):
#     filtered_damage = np.zeros_like(damage_array)  # Initialize filtered damage array with all zeros
#     last_damage_indices = np.where(damage_array==1)[0]  # Get indices where damage was taken
#     print(last_damage_indices)
#
#     if len(last_damage_indices) > 0:
#         death_indices = np.where(death_array==1)[0]  # Get indices where death occurred
#
#         if len(death_indices) > 0:
#             last_damage_index = last_damage_indices[np.searchsorted(last_damage_indices, death_indices) - 1]
#             filtered_damage[last_damage_index] = 1.  # Set the last damage before death to 1
#
#     return np.int8(filtered_damage)
#
#
# deaths =  np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) > 0
# damages = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]) > 0
#
# print(filter_last_damage_before_death(damages, deaths))

# def discount(x):
#     return x*0.997 - 1/120
#
# x = 3
# i = 0
# while x > 0:
#     i += 1
#     x = discount(x)
#     print(x, i)

delay = 3
delay_idx = 0

x = [0,1,2,3]

def get_observed_index(undelay=False):
    observed = (delay_idx % (delay + 1))
    if not undelay:
        observed -= delay
    return observed

for delay_idx in range(10):
    print(delay_idx, get_observed_index(), x[get_observed_index(undelay=True)], x[get_observed_index()])
