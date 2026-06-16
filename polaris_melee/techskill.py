import inspect
from collections import defaultdict
from typing import Dict

import numpy as np
from melee import LCancelState, YoshiMoves
from melee.enums import Character, Action
from melee.gamestate import GameState, PlayerState
from polaris_melee.base_rewards import RewardModule
from melee.enums import DKMoves
from polaris_melee.rewards_core import P


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
            for func_name, func in inspect.getmembers(self, predicate=inspect.ismethod)
            if not func.__name__.startswith("_")
        }

        self.metrics = defaultdict(float)
        self.name = self.__class__.__name__

    def _advance(self, ps):
        self.previous_player_states.pop(0)
        self.previous_player_states.append(ps)

    def __call__(self, player_state: PlayerState, distance: bool):
        bonus = 0.
        for func_name, func in self.funcs.items():
            func_bonus = float(func(player_state, distance))
            clean_name = func_name.replace("_", " ").capitalize()

            scaled_bonus = func_bonus * self.weights.get(func_name, 0.0)

            self.metrics[clean_name] += func_bonus
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
        "dashing": P * 0.1,
        "ledge_canceling": P * 6.,
        "lcanceling": P * 3,
        "wavelanding": P * 2.25,
        "wavedash": P * 0.32, # easy action
        "wavedash_off_platform": P * 2.25,
        "walljump": P * 5,
        "edge_drop": P * 3.5,
        "moonwalk": 0.,
        "fast_fall": 0.00, # todo
        "shield_drop": P * 3.,
    }

    def __init__(self):
        super().__init__(hist_len=10)
        self.lcancel_fails = 0
        self.lcancel_successes = 0

    def dashing(self, player_state: PlayerState, distance):
        return player_state.action == Action.DASHING# and self.previous_player_states[-1].action != Action.DASHING

    def fast_fall(self, player_state: PlayerState, distance):
        # todo we should find a way to detect players fast falling, using normall fall speed vs fast fall.
        prev_state = self.previous_player_states[-1]
        faster_falling = prev_state.speed_y_self - player_state.speed_y_self < 0
        return faster_falling and (prev_state.action in self.NORMAL_AIR_STATES and player_state.action in self.NORMAL_AIR_STATES)

    def shield_drop(self, player_state: PlayerState, distance):
        # todo we should find a way to detect players fast falling, using normall fall speed vs fast fall.
        prev_state = self.previous_player_states[-1]
        if player_state.character == Character.YOSHI:
            return (prev_state.action in (Action(YoshiMoves.ShieldDamage.value),)
                and player_state.action == Action.PLATFORM_DROP)

        return (prev_state.action in (Action.SHIELD_STUN, )
                and player_state.action == Action.PLATFORM_DROP)

    def edge_drop(self, player_state: PlayerState, distance):
        prev_state = self.previous_player_states[-1]

        return prev_state.action in (Action.EDGE_HANGING, Action.EDGE_CATCHING) and player_state.action in (
            Action.FALLING, Action.FALLING_FORWARD, Action.FALLING_BACKWARD)

    def ledge_canceling(self, player_state: PlayerState, distance):
        # in landing-lag > in air
        # could be air-dodge into waveland cancel
        return (self.previous_player_states[-1].action in TechSkillHelper.LANDING_ACTIONS
        and not (self.previous_player_states[-4].on_ground
                 and self.previous_player_states[-4] not in TechSkillHelper.NORMAL_AIR_STATES)
        and not player_state.on_ground and distance < 50
        )

    def lcanceling(self, player_state: PlayerState, distance):
        prev_state = self.previous_player_states[-1]

        if player_state.lcancel_status == prev_state.lcancel_status:
            return 0.

        success = player_state.lcancel_status == LCancelState.SUCCESSFUL
        self.lcancel_fails += int(not success)
        self.lcancel_successes += int(success)

        s = np.maximum(1. - distance / 70, 0.)

        return s * (float(success) * 2 - 1)

    # def dashdance(self, player_state: PlayerState, distance):
    #     # todo: here this encourages fast ddance
    #     return (player_state.action == Action.DASHING and  self.previous_player_states[-1].action == Action.TURNING and
    #         self.previous_player_states[-2].action == Action.DASHING
    #         and self.previous_player_states[-3].action == Action.DASHING)

    def wavelanding(self, player_state: PlayerState, distance):
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

    def wavedash(self, player_state: PlayerState, distance):
        old_state = self.previous_player_states[-1]
        not_hit = player_state.percent == old_state.percent

        return (
            not_hit and
            #player_state.action_frame == 1 and
            player_state.action == Action.LANDING_SPECIAL
            and (old_state.action == Action.KNEE_BEND)
        )

    def wavedash_off_platform(self, player_state: PlayerState, distance):
        old_state = self.previous_player_states[-1]
        return (
            not player_state.on_ground
            and (old_state.action == Action.LANDING_SPECIAL and abs(old_state.speed_ground_x_self)>2)
        )

    def walljump(self, player_state: PlayerState, distance):
        old_state = self.previous_player_states[-1]

        # WALL_TECH_JUMP -> both wall tech jump and wall jump
        return (
            player_state.action == Action.WALL_TECH_JUMP
            and old_state.action in TechSkillHelper.NORMAL_AIR_STATES
        )

    # do we make this char specific ?
    def moonwalk(self, player_state: PlayerState, distance):
        # TODO: improve
        old_state = self.previous_player_states[0]
        prev_state = self.previous_player_states[-1]
        facing = float(player_state.facing) * 2 - 1
        return (old_state.moonwalkwarning
                and (facing * player_state.speed_ground_x_self) < 0 and player_state.on_ground
                ) and player_state.speed_ground_x_self * prev_state.speed_ground_x_self <= 0
        #return (facing * player_state.speed_ground_x_self) < 2 and player_state.on_ground

    def _get_metrics(self):
        self.metrics["L-Cancel%"] = 0 if (self.lcancel_successes + self.lcancel_fails == 0) else (
                100 * self.lcancel_successes / (self.lcancel_successes + self.lcancel_fails))
        return self.metrics


class CptFalconHelper(Helper):

    weights = {
        "gentleman": P * 3,
    }

    def __init__(self):
        super().__init__(hist_len=1)

    def gentleman(self, player_state: PlayerState, distance):
        return (player_state.action == Action.NEUTRAL_ATTACK_3 and player_state.action_frame == 30)


class MarioHelper(Helper):

    weights = {
        "upb_walljump": P * 10,
    }

    def __init__(self):
        super().__init__(hist_len=1)

    def upb_walljump(self, player_state: PlayerState, distance):
        # TODO: looks too random.
        old_state = self.previous_player_states[-1]
        return old_state.action == Action.NEUTRAL_B_FULL_CHARGE_AIR and player_state.action == Action.WALL_TECH_JUMP

class DocHelper(Helper):

    weights = {
        "upb_cancel": P * 2,
    }

    def __init__(self):
        super().__init__(hist_len=1)

    def upb_cancel(self, player_state: PlayerState, distance):
        prev_state = self.previous_player_states[-1]

        return (distance < 30 and prev_state.action == Action.NEUTRAL_B_ATTACKING_AIR
                and (player_state.action == Action.LANDING_SPECIAL
                     )
                )


class ChargingHelper(Helper):
    weights = {
        "neutralb_charge": P * 15.,
        "neutralb_discharge": -P * 10.,
    }

    def __init__(self):
        super().__init__(hist_len=1)
        # TODO: move this to "char" helpers
        self.initialised = False

    def neutralb_charge(self, player_state: PlayerState, distance):
        if "character_specific" not in self.previous_player_states[-1].custom:
            return False
        prev_charge = self.previous_player_states[-1].custom["character_specific"]
        curr_charge = player_state.custom["character_specific"]
        return np.maximum(curr_charge - prev_charge, 0)

    def neutralb_discharge(self, player_state: PlayerState, distance):
        if "character_specific" not in self.previous_player_states[-1].custom:
            return False

        prev_charge = self.previous_player_states[-1].custom["character_specific"]
        curr_charge = player_state.custom["character_specific"]

        if player_state.action.value < 0xA:
            return False

        return np.maximum(prev_charge - curr_charge, 0)


class LuigiHelper(Helper):
    weights = {
        "cyclone_charge": P * 5,
    }

    def __init__(self):
        super().__init__(hist_len=1)

    def neutralb_charge(self, player_state: PlayerState, distance):
        if "character_specific" not in self.previous_player_states[-1].custom:
            return False
        prev_charge = self.previous_player_states[-1].custom["character_specific"]
        curr_charge = player_state.custom["character_specific"]
        return curr_charge > prev_charge


class LinkHelper(Helper):

    weights = {
        "neutral_b_charge": 0.001,
        "wall_hook": 0.1,
    }

    def __init__(self):
        super().__init__(hist_len=10)

    def neutral_b_charge(self, player_state: PlayerState, distance):
        old_state = self.previous_player_states[-6]

        return (old_state.action in (Action.NEUTRAL_B_CHARGING_AIR, Action.NEUTRAL_B_CHARGING)
                and player_state.action in (Action.NEUTRAL_B_CHARGING_AIR, Action.NEUTRAL_B_CHARGING)
                )

    def wall_hook(self, player_state: PlayerState, distance):
        # todo: determine action state
        return False


char_helpers = {
    Character.CPTFALCON: CptFalconHelper,
    Character.MARIO: MarioHelper,
    Character.DOC: DocHelper,
    Character.DK: ChargingHelper,
    Character.SAMUS: ChargingHelper,
    Character.MEWTWO: ChargingHelper
    # Character.LINK: LinkHelper,
    # Character.YLINK: LinkHelper,
    # fox/falco waveshines ?
    # sheik, puff, marth, roy ? neutral b charge
    # pichu, pikachu side b

    # TODO: add custom binary obs, for luigi (cyclone charge), dk, samus, mew2
}

class Techskill(RewardModule):
    def __init__(self, char: Character, discount: float):
        super().__init__(discount)
        self.helpers = [
            char_helpers.get(char, EmptyHelper)(),
            TechSkillHelper()
        ]

        self.frame_score = 0

    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):
        self.frame_score = 0
        for helper in self.helpers:
            self.frame_score += helper(player, gamestate.distance)

    def reward(self) -> float:

        return self.frame_score

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) -> Dict[str, float]:
        if as_opponent:
            return {}

        metrics = super().get_metrics(game_length_s, as_opponent)
        for helper in self.helpers:
            metrics.update(helper._get_metrics())
        return metrics
