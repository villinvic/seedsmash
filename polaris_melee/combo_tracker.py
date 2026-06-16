import numpy as np
from melee import Action, AttackState, PlayerState, Character, character_moves

from polaris_melee.compiled_libmelee_framedata import CompiledFrameData
from polaris_melee.rewards_core import NEUTRAL_ACTIONS
from polaris_melee.utils import HittingMoveTracker

UNLOCK_ACTIONS = (
    Action.KNEE_BEND,
    Action.DASHING
)

class ComboTracker:

    def __init__(
            self,
            max_combo: int,
            framedata: CompiledFrameData,
    ):

        self.max_combo = max_combo
        self.framedata = framedata
        self.current_combo_length = 0
        self.last_action = Action.UNKNOWN_ANIMATION
        self.last_percent = 0
        self.opp_last_percent = 0
        self.has_hit = False
        self.hitting_move_tracker = HittingMoveTracker()

        self.combos = []
        self.cool_states = [Action.WALL_TECH, Action.WALL_TECH_JUMP]

    def reset(self):
        if self.current_combo_length > 0:
            self.combos.append(self.current_combo_length)
        self.current_combo_length = 0
        self.last_action = Action.UNKNOWN_ANIMATION

    def get(self):
        return np.minimum(self.current_combo_length, self.max_combo)

    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            distance: float
    ) -> float:
        """
        Computes next combo length.
        Combo length reset to 0 if opponent escapes.
        """
        has_died = player.action.value <= 0xa
        has_killed = opponent.action.value <= 0xa

        dealt_damage = opponent.percent - self.opp_last_percent

        is_fresh_hit, has_move_ended = self.hitting_move_tracker.update(player.action, dealt_damage, distance) # TODO include hitlag/stun ?

        if (has_died or has_killed or opponent.action in UNLOCK_ACTIONS): # find a way to count combos even when crouch canceling (getup attacks).
            self.reset()
        elif is_fresh_hit:
            self.has_hit = True
        if has_move_ended and self.has_hit:
            self.has_hit = False
            if self.hitting_move_tracker.prev_action_state == Action.GRAB_PUMMEL:
                self.current_combo_length += 1/3
            else:
                self.current_combo_length += 1

        if player.action in self.cool_states and self.last_action not in self.cool_states:
            # why not ...
            self.current_combo_length += 1.


        self.opp_last_percent = opponent.percent
        self.last_action = player.action

        return self.get()

    def get_metrics(self):
        return {
            "Max Combo Length": 0 if len(self.combos) == 0 else np.max(self.combos)
        }



