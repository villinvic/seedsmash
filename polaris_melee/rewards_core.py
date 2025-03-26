from abc import abstractmethod
from typing import TypedDict, Dict

import numpy as np

from melee import PlayerState, GameState, Action

# rewards for 1 percent damage
P = 0.005
# reward for kills
D = 1.


NEUTRAL_ACTIONS = {
    Action.STANDING,
    Action.CROUCHING,
    Action.RUNNING,
    Action.DASHING,
    Action.KNEE_BEND,
    Action.FALLING,
    Action.FALLING_AERIAL,
    Action.FALLING_FORWARD,
    Action.FALLING_BACKWARD,
    Action.FALLING_AERIAL_FORWARD,
    Action.FALLING_AERIAL_BACKWARD,
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
    ):
        self.name = self.__class__.__name__
        self.magnitude = 0
        self.discount = 0.994
        self.frameskip = 3
        self.trajectory_length = 256
        self.num_trajectories = 0
        self.rs = []
        self.step = 0
        self.frame = 0

    def track_magnitude(self, new_r: float):
        t = self.step % self.trajectory_length
        if self.step > 0 and t == 0:
            gs = np.mean(np.abs(discounted_cumsum(self.rs, self.discount)))
            self.magnitude = gs / (self.num_trajectories + 1) + self.magnitude * self.num_trajectories / (self.num_trajectories + 1)
            self.num_trajectories += 1
            self.rs = []

        self.rs.append(new_r)
        self.frame += 1
        if self.frame % self.frameskip:
            self.step += 1


    @abstractmethod
    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):
        pass


    @abstractmethod
    def reward(
            self,
            advantage: float,
            opponent_combo_counter: int
    ) -> float:
        return 0.

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) -> Dict[str, float | Dict[str, float]]:
        if not as_opponent:
            return {f"Magnitude[{self.name}]": self.magnitude}
        return {}


class StepRewards(TypedDict):
    core_rewards: float | RewardModule
    action_state_rewards: float | RewardModule
    closeup_rewards: float | RewardModule
    stage_control_rewards: float | RewardModule
    offstage_rewards: float | RewardModule
    neutral_rewards: float | RewardModule
    techskill_rewards: float | RewardModule
