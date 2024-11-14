import numpy as np
from melee import Action, AttackState


class ComboTracker:

    def __init__(
            self,
            observation_manager,
            small_hit_scale=0.5,
            small_hit_percent=5,
            repeated_hit_scale=0.5,
    ):
        self.max_combo = observation_manager.MAX_COMBO
        self.observation_manager = observation_manager
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
            dealt_damage,
            suffered_damage,
            curr_action,
            has_died,
            has_killed,
            opp_state
    ):
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


    def is_opp_attacking(self, opp_state):
        char = opp_state.character
        action_state = opp_state.action
        action_frame = opp_state.action_frame

        return (self.observation_manager.FD.attack_state(char, action_state, action_frame) == AttackState.ATTACKING
                        and action_state not in (Action.GETUP_ATTACK, Action.GROUND_ATTACK_UP))


