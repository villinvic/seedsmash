import numpy as np
import tree

xplatform_indices = np.arange(12, dtype=np.int32) % 3 != 0
def swap_platforms(p):
    p = p.copy()
    p[:, :, xplatform_indices] *= -1.
    return p

def swap_encoded_action_sticks(a):
    a = a.copy()
    a[:, :, [0, 2]] *= -1.
    return a

def bin_swap(x):
    return 1.-x

def x_swap(x):
    return -1. * x

def swap_coord(c):
    c = c.copy()
    c[:, :, 0] *= -1.
    return c

def swap_projectiles(p):
    # 0 and 2
    p = p.copy()
    assert p.shape[-1] % 7 == 0
    num_proj = p.shape[-1] // 7
    for i in range(num_proj):
        p[:, :, i * 7 + 0] *= -1
        p[:, :, i * 7 + 2] *= -1

    return p


swapping_functions = dict(
    platforms=swap_platforms, # makes sense for FoD, otherwise, we provide a fake layout
    projectiles=swap_projectiles
)

player_swaps = dict(
    facing=bin_swap,
    speed_air_x_self=x_swap,
    speed_x_attack=x_swap,
    speed_ground_x_self=x_swap,
    encoded_action=swap_encoded_action_sticks,
    position=swap_coord,
    # action # TODO: do we need to swap direction specific moves, such as puff's rest ?
    owned_projectiles=swap_projectiles
)

player_swaps = {
    f"{obs}{port}": func
    for obs, func in player_swaps.items()
    for port in [1, 2]
}

swapping_functions.update(player_swaps)

def swap_for_obs(p, o):
    f = p[-1]
    swap_func = swapping_functions.get(f, lambda x: x)

    return swap_func(o)

def swap_x(observation):
    return tree.map_structure_with_path(
        swap_for_obs,
        observation
    )
