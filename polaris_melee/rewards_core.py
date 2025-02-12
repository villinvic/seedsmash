from abc import abstractmethod
from typing import TypedDict, Dict

from melee import PlayerState, GameState, Action

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
    Action.FALLING_AERIAL_BACKWARD
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


class RewardModule:
    """
    Here we need information about both players to compute the rewards
    """

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
        return {}


class StepRewards(TypedDict):
    core_rewards: float | RewardModule
    action_state_rewards: float | RewardModule
    closeup_rewards: float | RewardModule
    stage_control_rewards: float | RewardModule
    offstage_rewards: float | RewardModule
    neutral_rewards: float | RewardModule
    techskill_rewards: float | RewardModule
