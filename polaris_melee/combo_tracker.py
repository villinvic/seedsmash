import numpy as np
from melee import Action, AttackState, PlayerState

from polaris_melee.compiled_libmelee_framedata import CompiledFrameData
from polaris_melee.rewards_core import NEUTRAL_ACTIONS


UNLOCK_ACTIONS = (
    Action.KNEE_BEND,
    Action.DASHING
)

class ComboTracker:

    def __init__(
            self,
            max_combo: int,
            framedata: CompiledFrameData,
            small_hit_scale=0.1,
            small_hit_percent=4,
            repeated_hit_scale=0.5,
    ):
        self.max_combo = max_combo
        self.framedata = framedata
        self.small_hit_scale = small_hit_scale
        self.small_hit_percent = small_hit_percent
        self.repeated_hit_scale = repeated_hit_scale
        self.current_combo_length = 0
        self.last_action = Action.UNKNOWN_ANIMATION

        self.last_percent = 0
        self.opp_last_percent = 0

        self.combos = []

    def reset(self):
        if self.current_combo_length > 0:
            self.combos.append(self.current_combo_length)
        self.current_combo_length = 0
        self.last_action = Action.UNKNOWN_ANIMATION


    def update(
            self,
            player: PlayerState,
            opponent: PlayerState
    ) -> float:
        """
        Computes next combo length.
        Combo length reset to 0 if opponent escapes.
        """
        curr_action = player.action,
        has_died = player.action.value <= 0xa
        has_killed = opponent.action.value <= 0xa

        dealt_damage = np.maximum(opponent.percent - self.opp_last_percent, 0)
        if (has_died or has_killed or opponent.action in UNLOCK_ACTIONS): # find a way to count combos even when crouch canceling (getup attacks).
            self.reset()
        elif dealt_damage > 1:
            combo_increment = 1
            if dealt_damage < self.small_hit_percent:
                combo_increment *= self.small_hit_scale * dealt_damage
            if self.last_action == curr_action:
                combo_increment *= self.repeated_hit_scale
            self.current_combo_length = self.current_combo_length + combo_increment

        self.opp_last_percent = opponent.percent

        return np.minimum(self.current_combo_length, self.max_combo)

    def get_metrics(self):
        return {
            "Max Combo Length": 0 if len(self.combos) == 0 else np.max(self.combos)
        }



