import numpy as np
from melee import Action, AttackState, PlayerState

from polaris_melee.compiled_libmelee_framedata import CompiledFrameData
from polaris_melee.rewards_core import NEUTRAL_ACTIONS


class ComboTracker:

    def __init__(
            self,
            max_combo: int,
            framedata: CompiledFrameData,
            small_hit_scale=0.1,
            small_hit_percent=5,
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

        dealt_damage = np.maximum(opponent.percent - self.last_percent, 0)

        if has_died or has_killed or opponent.action in NEUTRAL_ACTIONS:
            self.reset()
        elif dealt_damage > 1:
            combo_increment = 1
            if dealt_damage < self.small_hit_percent:
                combo_increment *= self.small_hit_scale * dealt_damage
            if self.last_action == curr_action:
                combo_increment *= self.repeated_hit_scale
            self.current_combo_length = np.minimum(self.current_combo_length + combo_increment, self.max_combo)

        self.last_percent = opponent.percent

        return self.current_combo_length


    def is_opp_attacking(
            self,
            opp_state: PlayerState
    ) -> bool:
        """
        Helper function to know whether a player has initiated an action state that counts as combo breaker.
        """
        char = opp_state.character
        action_state = opp_state.action
        action_frame = opp_state.action_frame

        return (self.framedata.attack_state(char, action_state, action_frame) == AttackState.ATTACKING
                        and action_state not in (Action.GETUP_ATTACK, Action.GROUND_ATTACK_UP))

    def get_metrics(self):
        return {
            "Mean Combo Length": 0 if len(self.combos) == 0 else np.mean(self.combos)
        }



