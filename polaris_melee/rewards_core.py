from abc import abstractmethod
from typing import TypedDict, Dict

import numpy as np

from melee import PlayerState, GameState, Action

# reward for winning
W = 4.
# reward for kills
D = 1.
# rewards for 1 percent damage
P = D * (1 / 200)


NEUTRAL_ACTIONS = {
    Action.DEAD_DOWN,
    Action.DEAD_LEFT,
    Action.DEAD_RIGHT,
    Action.DEAD_UP,
    Action.DEAD_FLY_STAR,
    Action.DEAD_FLY_STAR_ICE,
    Action.DEAD_FLY,
    Action.DEAD_FLY_SPLATTER,
    Action.DEAD_FLY_SPLATTER_FLAT,
    Action.DEAD_FLY_SPLATTER_ICE,
    Action.DEAD_FLY_SPLATTER_FLAT_ICE,
    Action.STANDING,
    Action.CROUCHING,
    Action.RUNNING,
    Action.DASHING,
    Action.KNEE_BEND,
    # Action.FALLING,
    # Action.FALLING_AERIAL,
    # Action.FALLING_FORWARD,
    # Action.FALLING_BACKWARD,
    # Action.FALLING_AERIAL_FORWARD,
    # Action.FALLING_AERIAL_BACKWARD,
    Action.JUMPING_BACKWARD,
    Action.JUMPING_FORWARD,
    Action.JUMPING_ARIAL_FORWARD,
    Action.JUMPING_ARIAL_BACKWARD,
    Action.GRAB_PULLING,
    Action.GRAB_PULLING_HIGH,
    Action.GRAB_RUNNING_PULLING
}

NEUTRAL_GROUND_ACTIONS = {
    Action.STANDING,
    Action.CROUCHING,
    Action.RUNNING,
    Action.DASHING,
    Action.KNEE_BEND
}

ROLL_STATES = {Action.ROLL_FORWARD, Action.ROLL_BACKWARD, Action.GROUND_ROLL_FORWARD_UP, Action.GROUND_ROLL_BACKWARD_UP,
               Action.GROUND_ROLL_FORWARD_DOWN, Action.GROUND_ROLL_BACKWARD_DOWN, Action.GROUND_ROLL_SPOT_DOWN,
               Action.FORWARD_TECH, Action.BACKWARD_TECH, Action.WALK_FAST, Action.WALK_MIDDLE, Action.WALK_SLOW,
               Action.ON_HALO_DESCENT, Action.EDGE_ROLL_SLOW, Action.EDGE_GETUP_SLOW, Action.EDGE_ROLL_QUICK,
               Action.EDGE_GETUP_QUICK, Action.EDGE_JUMP_1_QUICK, Action.EDGE_JUMP_2_QUICK,
               Action.EDGE_JUMP_1_SLOW, Action.EDGE_JUMP_2_SLOW
               }
GETUP_ATTACKS = {
    Action.GETUP_ATTACK, Action.EDGE_ATTACK_QUICK, Action.EDGE_ATTACK_SLOW, Action.GROUND_ATTACK_UP
}

INTANGIBLE_STATES = ROLL_STATES | GETUP_ATTACKS | {Action.GROUND_GETUP, Action.GROUND_SPOT_UP}


def discounted_cumsum(x, gamma):
    """Compute the discounted cumulative sum of a 1D array efficiently.

    Args:
        x (np.ndarray): Input array of rewards or advantages (1D).
        gamma (float): Discount factor (0 <= gamma <= 1).

    Returns:
        np.ndarray: Discounted cumulative sum.
    """
    n = len(x)
    y = np.zeros(n, dtype=np.float32)
    y[-1] = x[-1]

    for i in range(n - 2, -1, -1):
        y[i] = x[i] + gamma * y[i + 1]

    return y

class RewardModule:
    """
    Here we need information about both players to compute the rewards
    """

    def __init__(
            self,
            discount: float
    ):
        self.magnitude = 0
        self.discount = discount
        self.trajectory_length = 256
        self.num_trajectories = 0
        self.rs = []
        self.step = 0

        self.name = ""
        self.w = 1.

    def register_as(self, name, w):
        self.name = name
        self.w = w

    @abstractmethod
    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):
        pass

    @abstractmethod
    def reward(self) -> float:
        return 0.

    def on_episode_end(self):
        pass

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) -> Dict[str, float | Dict[str, float]]:
        return {}


class StepRewards(TypedDict):
    stock_rewards: float | RewardModule
    damage_rewards: float | RewardModule