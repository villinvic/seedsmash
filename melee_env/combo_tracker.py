import numpy as np
from melee import Action, AttackState, PlayerState

from melee_env.compiled_libmelee_framedata import CompiledFrameData


class ComboTracker:

    def __init__(
            self,
            max_combo: int,
            framedata: CompiledFrameData,
            small_hit_scale=0.5,
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

    def reset(self):
        self.current_combo_length = 0
        self.last_action = Action.UNKNOWN_ANIMATION


    def update(
            self,
            dealt_damage: float,
            suffered_damage: float,
            curr_action: Action,
            has_died: bool,
            has_killed: bool,
            opp_state: PlayerState
    ) -> float:
        """
        Computes next combo length.
        Combo length reset to 0 if opponent escapes.
        """

        if suffered_damage or has_died or has_killed or self.is_opp_attacking(opp_state):
            self.reset()
        elif dealt_damage:
            combo_increment = 1
            if dealt_damage < self.small_hit_percent:
                combo_increment *= self.small_hit_scale
            if self.last_action == curr_action:
                combo_increment *= self.repeated_hit_scale
            self.current_combo_length = np.minimum(self.current_combo_length + combo_increment, self.max_combo)

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


