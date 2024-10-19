import inspect

import numpy as np
from melee.enums import Character, Action
from melee.gamestate import GameState, PlayerState


class Helper:
    """
    Helper that rewards the player for tricky things to do in melee.
    This should not be hackable.
    We do not want agents to spam these.
    """

    weights = {}

    def __init__(self, hist_len=6):

        self.previous_player_states = [PlayerState() for _ in range(hist_len)]
        self.hist_len = hist_len
        self.funcs = {
            func.__name__: func
            for func in inspect.getmembers(self, predicate=inspect.isfunction)
            if not func.__name__.startswith("_")
        }

        self.metrics = {}
        self.name = self.__class__.__name__

    def _advance(self, ps):
        self.previous_player_states.pop(0)
        self.previous_player_states.append(ps)

    def __call__(self, player_state: PlayerState, is_near: bool):
        bonus = 0.
        for func_name, func in self.funcs.items():
            func_bonus = float(func(player_state, is_near))
            self.metrics[func_name] += func_bonus
            bonus += func_bonus * self.weights[func_name]
        self._advance(player_state)
        return bonus

    def _reset(self):
        Helper.__init__(self, hist_len=self.hist_len)

    def _get_metrics(self):
        return self.metrics


class EmptyHelper(Helper):

    def __call__(self, *args, **kwargs):
        return 0.


class TechSkillHelper(Helper):

    LANDING_ACTIONS = (Action.LANDING_SPECIAL, Action.BAIR_LANDING, Action.FAIR_LANDING, Action.DAIR_LANDING,
                       Action.HAMMER_LANDING, Action.UAIR_LANDING, Action.NAIR_LANDING)

    NORMAL_AIR_STATES: (Action.FALLING, Action.FALLING_FORWARD, Action.FALLING_AERIAL, Action.FALLING_AERIAL_FORWARD,
                        Action.FALLING_AERIAL_BACKWARD)
    bonuses = {
        "ledge_canceling": 1.,
        "dashdance": 0.01,
        "platform_landing": 0.5,
        "wavelanding": 0.5,
        "wavedash": 0.1,
        "wavedash_off_platfrom": 0.5,
        "walljump": 1.,
        "moonwalk": 0.5
    }

    def __init__(self):
        super().__init__(hist_len=10)

    def ledge_canceling(self, player_state: PlayerState, is_near: bool):
        # in landing-lag > in air
        return is_near and (self.previous_player_states[-1].action in TechSkillHelper.LANDING_ACTIONS
        and not player_state.on_ground
        )

    def dashdance(self, player_state: PlayerState, is_near: bool):
        # dont think we need an helper for this
        last_states = self.previous_player_states[-10:] + [player_state]
        action_states = np.array(ps.action for ps in last_states)
        num_dashing_states = np.sum(action_states == Action.DASHING)
        dash_dancing = False
        if num_dashing_states > 0:
            dashing_standing = np.all(np.logical_or(action_states == Action.STANDING, action_states == Action.DASHING))
            if dashing_standing:
                facings = [int(ps.facing) for ps in last_states]
                num_dash_backs = np.diff(facings).sum()
                dash_dancing = num_dash_backs >= 2

        # can we dash for many frames in a row without turning ?
        return dash_dancing

    def platform_landing(self, player_state: PlayerState, is_near: bool):
        old_state = self.previous_player_states[-5]
        return ((not old_state.on_ground) and old_state.speed_y_self >= 0
            and player_state.action == Action.LANDING)

    def wavelanding(self, player_state: PlayerState, is_near: bool):
        old_state = self.previous_player_states[-3]
        return (player_state.action == Action.LANDING and abs(player.speed_ground_x_self)>1
            and (not old_state.on_ground) and old_state.speed_y_self < 0)

    def wavedash(self, player_state: PlayerState, is_near: bool):
        old_state = self.previous_player_states[-1]
        return (
            player_state.action == Action.LANDING
            and (old_state.action == Action.KNEE_BEND)
        )

    def wavedash_off_platform(self, player_state: PlayerState, is_near: bool):
        old_state = self.previous_player_states[-1]
        return (
            not player_state.on_ground
            and (old_state.action == Action.LANDING and abs(player.speed_ground_x_self)>1)
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
        old_state = self.previous_player_states[-3]
        facing = float(player_state.facing) * 2 - 1
        return (old_state.moonwalkwarning
                and (facing * player_state.speed_ground_x_self) < 0 and player_state.on_ground
                )


class CptFalconHelper(Helper):

    bonuses = {
        "gentleman": 0.5,
    }

    def __init__(self):
        super().__init__(hist_len=10)

    def gentleman(self, player_state: PlayerState, is_near: bool):
        return is_near and (False)


class MarioHelper(Helper):

    bonuses = {
        "upb_walljump": 2.,
    }

    def __init__(self):
        super().__init__(hist_len=10)

    def upb_walljump(self, player_state: PlayerState, is_near: bool):
        old_state = self.previous_player_states[-3]
        return old_state.action == Action.UP_B_AIR and player_state.action == Action.WALL_TECH_JUMP

class DocHelper(Helper):

    bonuses = {
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

    bonuses = {
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

    bonuses = {
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
    Character.CPTFALCON: Helper,
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

    None: EmptyHelper,
}

class MeleeHelper:
    def __init__(self, port: int, char: Character, techksill=True):

        self.port = port
        self.helpers = [char_helpers.get(char)()]
        if techksill:
            self.helpers.append(TechSkillHelper())

    def __call__(self, gamestate: GameState):
        bonus = 0.
        is_near = np.sqrt(np.square(gamestate.players[1].x-gamestate.players[2].x)
                          +np.square(gamestate.players[1].y-gamestate.players[2].y)) < 20
        player_state = gamestate.players[self.port]
        for helper in self.helpers:
            bonus += helper(player_state, is_near)

        return bonus

    def get_metrics(self):
        return {
            helper.name: helper._get_metrics()
            for helper in self.helpers
        }



