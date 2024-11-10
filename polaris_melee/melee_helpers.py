import inspect
from collections import defaultdict

import numpy as np
from melee import LCancelState
from melee.enums import Character, Action
from melee.gamestate import GameState, PlayerState


class Helper:
    """
    Helper that rewards the player for tricky things to do in melee.
    This should not be hackable.
    We do not want agents to spam these.
    """

    weights = {}

    decay = 20.

    def __init__(self, hist_len=6):

        self.previous_player_states = [PlayerState() for _ in range(hist_len)]
        self.hist_len = hist_len
        self.funcs = {
            func.__name__: func
            for func_name, func in inspect.getmembers(self, predicate=inspect.ismethod)
            if not func.__name__.startswith("_")
        }

        self.metrics = defaultdict(float)
        self.name = self.__class__.__name__

    def _advance(self, ps):
        self.previous_player_states.pop(0)
        self.previous_player_states.append(ps)

    def __call__(self, player_state: PlayerState, is_near: bool):
        bonus = 0.
        for func_name, func in self.funcs.items():
            func_bonus = float(func(player_state, is_near))

            total_collected = self.metrics[func_name] * self.weights.get(func_name, 0.0)
            scaled_bonus = func_bonus * self.weights.get(func_name, 0.0) / (1 + 1e-2 * np.exp(total_collected * self.decay))

            self.metrics[func_name] += func_bonus

            bonus += scaled_bonus
        self._advance(player_state)
        return bonus

    def _reset(self):
        self.__init__(hist_len=self.hist_len)

    def _get_metrics(self):
        return self.metrics


class EmptyHelper(Helper):

    def __call__(self, *args, **kwargs):
        return 0.


class TechSkillHelper(Helper):

    LANDING_ACTIONS = (Action.LANDING_SPECIAL, Action.BAIR_LANDING, Action.FAIR_LANDING, Action.DAIR_LANDING,
                       Action.HAMMER_LANDING, Action.UAIR_LANDING, Action.NAIR_LANDING)

    NORMAL_AIR_STATES = (Action.FALLING, Action.FALLING_BACKWARD,
                         Action.FALLING_FORWARD, Action.FALLING_AERIAL, Action.FALLING_AERIAL_FORWARD,
                        Action.FALLING_AERIAL_BACKWARD, Action.JUMPING_ARIAL_BACKWARD, Action.JUMPING_ARIAL_FORWARD,
                         Action.JUMPING_FORWARD, Action.JUMPING_FORWARD)
    weights = {
        "dashing": 0.01,
        "ledge_canceling": 0.03,
        "lcanceling": 0.01,
        "wavelanding": 0.02,
        "wavedash": 0.003, # easy action
        "wavedash_off_platform": 0.02,
        "walljump": 0.05,
        "moonwalk": 0.03
    }

    def __init__(self):
        super().__init__(hist_len=19)


    def dashing(self, player_state: PlayerState, is_near: bool):
        old_state = self.previous_player_states[-1]

        return player_state.action == Action.DASHING and old_state.action != Action.DASHING

    def ledge_canceling(self, player_state: PlayerState, is_near: bool):
        # in landing-lag > in air
        # could be air-dodge into waveland cancel
        return (self.previous_player_states[-1].action in TechSkillHelper.LANDING_ACTIONS
        and not (self.previous_player_states[-4].on_ground
                 and self.previous_player_states[-4] not in TechSkillHelper.NORMAL_AIR_STATES)
        and not player_state.on_ground
        )

    def lcanceling(self, player_state: PlayerState, is_near: bool):
        prev_state = self.previous_player_states[-1]

        return (player_state.lcancel_status == LCancelState.SUCCESSFUL
                and player_state.lcancel_status != prev_state.lcancel_status)

    # def dashdance(self, player_state: PlayerState, is_near: bool):
    #     # todo: here this encourages fast ddance
    #     return (player_state.action == Action.DASHING and  self.previous_player_states[-1].action == Action.TURNING and
    #         self.previous_player_states[-2].action == Action.DASHING
    #         and self.previous_player_states[-3].action == Action.DASHING)

    def wavelanding(self, player_state: PlayerState, is_near: bool):
        # todo: check again
        prev_state = self.previous_player_states[-1]

        was_in_air = all([
            not s.on_ground for s in self.previous_player_states[-7:-1]
        ])
        waveland_ground = player_state.y < 1 and player_state.speed_ground_x_self > 0.2
        waveland_platform = player_state.y > 5

        return (
                (waveland_platform or waveland_ground)
            and player_state.action == Action.LANDING_SPECIAL
            and prev_state.action in TechSkillHelper.NORMAL_AIR_STATES + (Action.AIRDODGE,)
            and was_in_air)

    def wavedash(self, player_state: PlayerState, is_near: bool):
        old_state = self.previous_player_states[-2]
        not_hit = player_state.percent == old_state.percent
        return (
            not_hit and
            player_state.action_frame == 1 and
            player_state.action == Action.LANDING_SPECIAL
            and (old_state.action == Action.KNEE_BEND)
        )

    def wavedash_off_platform(self, player_state: PlayerState, is_near: bool):
        old_state = self.previous_player_states[-1]
        return (
            not player_state.on_ground
            and (old_state.action == Action.LANDING_SPECIAL and abs(old_state.speed_ground_x_self)>1)
        )

    def walljump(self, player_state: PlayerState, is_near: bool):
        old_state = self.previous_player_states[-1]

        # WALL_TECH_JUMP -> both wall tech jump and wall jump
        return (
            player_state.action == Action.WALL_TECH_JUMP
            and old_state.action in TechSkillHelper.NORMAL_AIR_STATES
        )

    # do we make this char specific ?
    def moonwalk(self, player_state: PlayerState, is_near: bool):
        old_state = self.previous_player_states[0]
        prev_state = self.previous_player_states[-1]
        facing = float(player_state.facing) * 2 - 1
        return (old_state.moonwalkwarning
                and (facing * player_state.speed_ground_x_self) < 0 and player_state.on_ground
                ) and player_state.speed_ground_x_self * prev_state.speed_ground_x_self <= 0


class CptFalconHelper(Helper):

    weights = {
        "gentleman": 0.1,
    }

    def __init__(self):
        super().__init__(hist_len=10)

    def gentleman(self, player_state: PlayerState, is_near: bool):
        return is_near and (player_state.action == Action.NEUTRAL_ATTACK_3 and player_state.action_frame == 30)


class MarioHelper(Helper):

    weights = {
        "upb_walljump": 2.,
    }

    def __init__(self):
        super().__init__(hist_len=10)

    def upb_walljump(self, player_state: PlayerState, is_near: bool):
        old_state = self.previous_player_states[-3]
        return old_state.action == Action.UP_B_AIR and player_state.action == Action.WALL_TECH_JUMP

class DocHelper(Helper):

    weights = {
        "upb_cancel": 0.1,
    }

    def __init__(self):
        super().__init__(hist_len=10)

    def upb_cancel(self, player_state: PlayerState, is_near: bool):
        last_states = self.previous_player_states[-3:]
        old_state = self.previous_player_states[-1]
        was_in_up_b = np.any(np.array(ps.action for ps in last_states) == Action.UP_B_GROUND)

        return (was_in_up_b
                and (player_state.action == Action.LANDING_SPECIAL
                and old_state.action != Action.LANDING_SPECIAL)
                )


class DKHelper(Helper):

    weights = {
        "neutral_b_charge": 0.01,
        "neutral_b_full_charge": 1.,

    }

    CHARGING_STATES = (Action.NEUTRAL_B_CHARGING_AIR, Action.NEUTRAL_B_CHARGING)
    FULL_CHARGE_STATES =  (Action.NEUTRAL_B_FULL_CHARGE, Action.NEUTRAL_B_FULL_CHARGE_AIR)

    def __init__(self):
        super().__init__(hist_len=10)

    def neutral_b_charge(self, player_state: PlayerState, is_near: bool):
        old_state = self.previous_player_states[-6]

        return (old_state.action in self.CHARGING_STATES
                and player_state.action in self.CHARGING_STATES
                )

    def neutral_b_full_charge(self, player_state: PlayerState, is_near: bool):
        old_state = self.previous_player_states[-1]

        return (old_state.action not in self.FULL_CHARGE_STATES
                and player_state.action in self.FULL_CHARGE_STATES
                )


class SamusHelper(DKHelper):
    pass

class MewtwoHelper(DKHelper):
    pass


class LinkHelper(Helper):

    weights = {
        "neutral_b_charge": 0.001,
        "wall_hook": 0.1,
    }

    def __init__(self):
        super().__init__(hist_len=10)

    def neutral_b_charge(self, player_state: PlayerState, is_near: bool):
        old_state = self.previous_player_states[-6]

        return (old_state.action in (Action.NEUTRAL_B_CHARGING_AIR, Action.NEUTRAL_B_CHARGING)
                and player_state.action in (Action.NEUTRAL_B_CHARGING_AIR, Action.NEUTRAL_B_CHARGING)
                )

    def wall_hook(self, player_state: PlayerState, is_near: bool):
        # todo: determine action state
        return False


char_helpers = {
    Character.CPTFALCON: CptFalconHelper,
    Character.MARIO: MarioHelper,
    Character.DOC: DocHelper,
    Character.DK: DKHelper,
    Character.SAMUS: SamusHelper,
    Character.LINK: LinkHelper,
    Character.YLINK: LinkHelper,
    Character.MEWTWO: MewtwoHelper,
    # fox/falco waveshines ?
    # sheik, puff, marth, roy ? neutral b charge
    # pichu, pikachu side b

    # TODO: add custom binary obs, for luigi (cyclone charge), dk, samus, mew2
}

class MeleeHelper:
    def __init__(self, port: int, char: Character, techksill=True):

        self.port = port
        self.helpers = [char_helpers.get(char, EmptyHelper)()]
        if techksill:
            self.helpers.append(TechSkillHelper())

    def __call__(self, gamestate: GameState):
        bonus = 0.
        is_near = np.sqrt(np.square(gamestate.players[1].x-gamestate.players[2].x)
                          +np.square(gamestate.players[1].y-gamestate.players[2].y)) < 50
        player_state = gamestate.players[self.port]
        for helper in self.helpers:
            bonus += helper(player_state, is_near)

        return bonus

    def get_metrics(self):

        metrics = {
            helper.name: helper._get_metrics()
            for helper in self.helpers
        }
        return metrics
