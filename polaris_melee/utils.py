from typing import Tuple

from melee import Action

class HittingMoveTracker:

    def __init__(self):
        self.has_move_hit = False
        self.is_hitting = False
        self.curr_action_state = Action.SHIELD
        self.prev_action_state = Action.SHIELD

        self.is_fresh_hit = False
        self.has_move_ended = False


    def update(self, action, damage, distance) -> Tuple[bool, bool]:
        self.prev_action_state = self.curr_action_state
        self.curr_action_state = action
        self.is_hitting = damage > 1 or (damage == 1 and distance < 20)
        # count move if it was just actioned:

        self.is_fresh_hit = False
        self.has_move_ended = False
        if self.curr_action_state != self.prev_action_state:
            self.has_move_hit = False
            self.has_move_ended = True
        elif self.is_hitting and not self.has_move_hit:
            self.has_move_hit = True
            self.is_fresh_hit = True

        return self.is_fresh_hit, self.has_move_ended
