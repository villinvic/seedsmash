from collections import defaultdict
from typing import Dict, TypedDict, Any

import numpy as np
import tree
from melee import PlayerState, GameState, Action, InvulnerabilityType, EDGE_POSITION, character_moves, Character, \
    YoshiMoves, Stage, stages, Position
from polaris_melee.observation_space import ObsBuilder
from polaris_melee.rewards_core import StepRewards, RewardModule, NEUTRAL_ACTIONS, GETUP_ATTACKS, ROLL_STATES, \
    NEUTRAL_GROUND_ACTIONS, P, D, W
from polaris_melee.techskill import Techskill
from polaris_melee.utils import HittingMoveTracker
from seedsmash.bot import Bot
from seedsmash.elo_matchmaking import SeedSmashMatchmaking


class CloseupReward(RewardModule):

    def __init__(self,
                 discount: float,
                 version: int,
                 aggresivity: float,
                 ):
        super().__init__(discount)
        self.self_position = None
        self.opponent_position = None

        self.y_scale = 0.65
        self.closeup = 0.
        self.good_closeup = 0.
        self.total_closeup = 0.
        self.total_good_closeup = 0.
        self.cumulated_distance = 0.

        self.good_closeup_scale = P / 4
        scale = np.clip(1 - version / 1500, 0., 1.) + aggresivity / 800
        self.hitstun_threshold = 30 - aggresivity / 5
        self.scale = scale * P / 4

    def player_distance(self, a: Position, b: Position):
        dx = a.x - b.x
        dy = (a.y - b.y) * self.y_scale
        return np.sqrt(dx ** 2 + dy ** 2)

    def distance_from_ledge(self, pos: Position, stage: Stage):

        # we could provide absolute horizontal distance from ledge whenever y > 0
        dy2 = (pos.y * self.y_scale)
        dx2 = (abs(pos.x) - stages.EDGE_POSITION[stage])
        return np.sqrt(dx2 ** 2 + dy2 ** 2)

    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):

        if self.self_position is not None:

            prev_dist = self.player_distance(self.self_position, self.opponent_position)
            # freeze the other player's position to get the velocity of the player
            curr_dist = self.player_distance(player.position, self.opponent_position)

            closeup = prev_dist - curr_dist


            is_knockback = (
                abs(player.speed_y_attack)>1e-3 or abs(player.speed_x_attack)>1e-3
            )

            # We do not want to reward closeups with roll states
            if (
                    opponent.action.value > 0xA
                    or (player.action not in ROLL_STATES)
                    or (closeup < 0 and player.action in ROLL_STATES)
                    or (not is_knockback)
            ):

                # Reward running away when the opponent is invulnerable.
                if (opponent.invulnerability_type == InvulnerabilityType.INVULNERABLE
                        and player.invulnerability_type != InvulnerabilityType.INVULNERABLE):
                    closeup = -closeup

            else:
                closeup = 0

            self.closeup = np.minimum(closeup, 5.)
            self.total_closeup += np.maximum(self.closeup, 0.)
            self.cumulated_distance += gamestate.distance

            stage = gamestate.stage
            if (player.position.y < 0 and player.off_stage and not opponent.off_stage) and stage in [
                Stage.POKEMON_STADIUM,
                Stage.FOUNTAIN_OF_DREAMS,
                Stage.DREAMLAND,
                Stage.BATTLEFIELD,
            ]:
                prev_dist = self.distance_from_ledge(self.self_position, stage)
                curr_dist = self.distance_from_ledge(player.position, stage)
            else:
                prev_dist = self.player_distance(self.self_position, self.opponent_position)
                # get actual detla distance, if opponent moved away, we get punished
                curr_dist = self.player_distance(player.position, opponent.position)


            good_closeup = np.minimum(prev_dist - curr_dist, 5.)
            self.good_closeup = good_closeup * int(player.off_stage or opponent.off_stage or opponent.hitstun_frames_left > self.hitstun_threshold)
            self.total_good_closeup += np.maximum(self.good_closeup, 0.)

        self.self_position = player.position
        self.opponent_position = opponent.position


    def reward(
            self,
            advantage: float,
            opponent_combo_counter: int
    ) -> float:

        return self.closeup * self.scale + self.good_closeup * self.good_closeup_scale

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {}
        return {
            "Closeup$s": self.total_closeup / game_length_s,
            "Follow-up$s": self.total_good_closeup / game_length_s,
            "Average Player Distance": self.cumulated_distance / (game_length_s * 20 * 3),
            **super().get_metrics(game_length_s, as_opponent)
        }

class StallingReward(RewardModule):

    def __init__(self,
                 discount: float
                 ):
        super().__init__(discount)

        self.prev_percent = 0
        self.opponent_prev_percent = 0
        self.time_spent_far = 0
        self.distance_threshold = 25
        self.max_frames = 60 * 6
        self.scale = 1.2 * P / 60
        self.offstage = False
        self.stalling_frames = 0

        self.prev_position = Position(0., 0.)
        self.staticity = 0
        self.time_spent_static = 0
        self.staticity_threshold = 0.7
        self.max_static_frames = 20

        self.cumulated_staticity = 0

    def count_frame(self, is_ejected: bool, abs_damage: int, distance: float, abs_movement: float):

        if (
                distance < self.distance_threshold
                or abs_damage > 0
                or is_ejected > 0
        ):
            self.time_spent_far = 0
        else:
            self.time_spent_far += 1

        if self.time_spent_far > self.max_frames:
            self.stalling_frames += 1

        self.staticity = np.maximum(self.staticity_threshold - abs_movement, 0)
        self.cumulated_staticity += self.staticity

        if self.staticity > 0:
            self.time_spent_static += 1
        else:
            self.time_spent_static = 0

    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):

        # also look at deaths/kills to reset the counts:
        is_ejected =  ((abs(player.speed_y_attack) + abs(player.speed_x_attack)) > 0
                       or (abs(opponent.speed_y_attack) + abs(opponent.speed_x_attack)))

        abs_damage = abs(player.percent - self.prev_percent) + abs(opponent.percent - self.opponent_prev_percent)

        abs_movement = np.sqrt(
            (player.position.x - self.prev_position.x)**2
            +  0.3 * (player.position.y - self.prev_position.y)**2
        )

        self.count_frame(is_ejected, abs_damage, gamestate.distance, abs_movement)

        self.prev_percent = player.percent
        self.prev_position = player.position
        self.opponent_prev_percent = opponent.percent
        self.offstage = player.off_stage



    def reward(
            self,
            advantage: float,
            opponent_combo_counter: int
    ) -> float:

        stalling_penalty = int(self.time_spent_far > self.max_frames)
        # we may be stalling offstage, which is bad
        stalling_penalty = stalling_penalty * (1 + int(self.offstage))

        static_penalty = int(self.time_spent_static > self.max_static_frames) * 0. # TODO: see if this is needed ?


        return - self.scale * (stalling_penalty + static_penalty)

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {}
        return {
            "Stalling%": 100 * self.stalling_frames / (game_length_s * 20 * 3),
            "Staticity%": 100 * self.cumulated_staticity / (game_length_s * 20 * 3),

            **super().get_metrics(game_length_s, as_opponent)
        }


class WinReward(RewardModule):

    def __init__(
            self,
            discount: float
    ):
        super().__init__(discount=discount)
        self.is_done = False
        self.win = 0


    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):

        if self.is_done:
            self.win = 0
        else:
            if player.stock == 0 and opponent.stock == 0:
                self.is_done = True
            if opponent.stock == 0:
                self.win = 1
                self.is_done = True


    def reward(
            self,
            advantage: float,
            combo_counter: int
    ) -> float:

        r = self.win * W

        return r

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {}

        return super().get_metrics(game_length_s, as_opponent)


class SDReward(RewardModule):

    def __init__(
            self,
            discount: float
    ):
        super().__init__(discount)

        # we limit coalition as much as possible for death
        self.was_hit_offstage = False
        self.self_destructs = 0
        self.prev_stock = 4
        self.is_sd = False

    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):
        if opponent.hitstun_frames_left > 0 and opponent.off_stage:
            self.was_hit_offstage = True
        elif not opponent.off_stage:
            self.was_hit_offstage = False

        if opponent.stock < self.prev_stock and not self.was_hit_offstage:
            self.is_sd = True
            self.self_destructs += 1
        else:
            self.is_sd = False

        self.prev_stock = opponent.stock


    def reward(
            self,
            advantage: float,
            combo_counter: int
    ) -> float:

        return (
                float(self.is_sd) * D
        )

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) -> Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {
                "Self-Destructs$Game": self.self_destructs,
            }
        return {
            **super().get_metrics(game_length_s, as_opponent)
        }

class StockReward(RewardModule):

    def __init__(
            self,
            discount: float
    ):
        super().__init__(discount)
        self.prev_stock = 4
        self.stock = 4
        self.prev_opp_stock = 4
        self.prev_percent = 0
        self.deaths = []

        self.damaged_offstage = 0

    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):
        self.prev_stock = self.stock
        self.stock = opponent.stock

        if self.stock < self.prev_stock:
            self.deaths.append(self.prev_percent)

        self.prev_percent = opponent.percent

    def reward(
            self,
            advantage: float,
            combo_counter: int
    ) -> float:

        # Using the advantage here allows for bots to learn trading stocks when it is worth.
        death = int(self.stock < self.prev_stock) #* (1 + np.clip(advantage, 0))

        # death boost
        #counter = np.maximum(combo_counter-1, 0)
        #kill_combo_boost = 1 + counter / ObsBuilder.MAX_COMBO


        #kill_r = kill_combo_boost * death

        return (
            death * D
        )

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {
                "Average Death%": 200 if len(self.deaths) == 0 else np.mean(self.deaths),
            }

        return {
            "Average Kill%": 200 if len(self.deaths) == 0 else np.mean(self.deaths),
            ** super().get_metrics(game_length_s, as_opponent)
        }


class DamageReward(RewardModule):

    def __init__(
            self,
            character: Character,
            discount: float
    ):
        super().__init__(discount)
        self.prev_percent = 0
        self.percent = 0

        self.is_getup_attack = False
        self.is_intangible = False
        self.offstage = False

        self.total_offstage_offstage = 0
        self.total_damage = 0
        self.total_intangible_damage = 0
        self.num_shield_blocks = 0

        self.char_moves = character_moves[character]

        # TODO: reverse this module to compute rewards for the opponent, and not us
        self.other_aerial_moves = ("Uair", "Bair", "Dair", "Fair")
        self.smash_moves = ("FrontSmash", "UpSmash", "DownSmash")

        self.opponent_jump_multiplier = 1
        self.prev_opponent_in_hitstun = False
        self.opponent_in_hitstun = False

        if character == Character.YOSHI:
            self.shield_block_state = Action(YoshiMoves.ShieldDamage.value)
            self.shield_states = (Action(YoshiMoves.ShieldHold.value), Action.SHIELD_REFLECT, Action(YoshiMoves.ShieldStartup.value))
        else:
            self.shield_block_state = Action.SHIELD_STUN
            self.shield_states = (Action.SHIELD, Action.SHIELD_REFLECT, Action.SHIELD_START)

        self.prev_action = Action.SHIELD
        self.curr_action = Action.SHIELD

    def is_nair(self, action: Action):
        try:
            return self.char_moves(action.value).name == "Nair"
        except ValueError:
            return False

    def is_other_aerial(self, action: Action):
        try:
            return self.char_moves(action.value).name in self.other_aerial_moves
        except ValueError:
            return False

    def is_smash(self, action: Action):
        try:
            return self.char_moves(action.value).name in self.smash_moves
        except ValueError:
            return False

    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):

        self.prev_percent = self.percent
        self.prev_action = self.curr_action
        self.curr_action = player.action

        self.percent = opponent.percent

        self.is_getup_attack = player.action in GETUP_ATTACKS
        self.offstage = (player.off_stage and opponent.off_stage)
        self.is_intangible = player.invulnerability_type == InvulnerabilityType.INTANGIBLE

        if opponent.character in (Character.JIGGLYPUFF, Character.KIRBY):
            opponent_jumps = opponent.jumps_left / 6
        else:
            opponent_jumps = opponent.jumps_left / 2

        self.opponent_jump_multiplier = 1 + int(opponent_jumps < 0.4) * 0.25
        self.prev_opponent_in_hitstun = self.opponent_in_hitstun
        self.opponent_in_hitstun = opponent.hitstun_frames_left > 0

    def reward(
            self,
            advantage: float,
            combo_counter: int
    ) -> float:

        damage = np.maximum(self.percent - self.prev_percent, 0.)
        damage_r = 0.
        if damage > 0:
            combo_boost = 1 + (1.5 * combo_counter / ObsBuilder.MAX_COMBO)
            damage_r = combo_boost * damage * self.opponent_jump_multiplier
            if self.is_nair(self.prev_action):
                damage_r *= 1.
            elif self.is_other_aerial(self.prev_action):
                damage_r *= 1.05
            elif self.is_smash(self.prev_action):
                damage_r *= 0.8
            if self.prev_opponent_in_hitstun:
                damage_r *= 1.1


        if self.offstage:
            # TODO: I think offstage here is not doing great (makes recovering agent fearfull of taking damage)
            #damage_r *= (1 - advantage)
            self.total_offstage_offstage += damage

        self.total_damage += damage

        blocking = float(self.curr_action == self.shield_block_state and self.prev_action in self.shield_states)
        block_incentivisation = blocking
        self.num_shield_blocks += blocking

        return (
           damage_r * P + block_incentivisation * P * 2.5
        )

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {
                "Damage Incurred$s": self.total_damage / game_length_s,
                "Intangible Damage Incurred$game": self.total_intangible_damage,
                "Offstage Damage Incurred$game": self.total_offstage_offstage,
                "Shield Blocks$game": self.num_shield_blocks,

            }

        return {
            "Damage Dealt$s": self.total_damage / game_length_s,
            "Intangible Damage Dealt$game": self.total_intangible_damage,
            "Offstage Damage Dealt$game": self.total_offstage_offstage,
            ** super().get_metrics(game_length_s, as_opponent)
        }


class ActionStateReward(RewardModule):

    # Could be a creativity stat later

    WALL_TECH_STATES = [
        Action.WALL_TECH,
        Action.WALL_TECH_JUMP,
        Action.CEILING_TECH
    ]

    def __init__(
            self,
            bot: Bot,
            discount: float

    ):
        super().__init__(discount)

        self.get_state_reward = lambda a: bot.action_state_counts.get_reward(a)
        self.get_hit_reward = lambda a: bot.hit_counts.get_reward(a)
        self.state_r = 0.
        self.move_r = 0.

        self.prev_action_state = Action.FALLING
        self.prev_action_name = None
        self.curr_action_state = Action.FALLING
        self.prev_percent = 0
        self.curr_percent = 0

        self.action_state_counts = bot.action_state_counts.tracked.get_counter()
        self.hit_counts = bot.hit_counts.tracked.get_counter()

        self.char_moves = character_moves[bot.character]
        self.used_moves = {
            move.name: 0
            for move in self.char_moves
        }

        self.wall_techs = 0
        self.distance = 0.
        self.hitting_move_tracker = HittingMoveTracker()


        # for visiting a state that is 10x rarer than others, we get 2 P reward
        log10 = np.log(10.)
        self.state_scale = P * 1 / log10
        # for hitting a move that is 10x rarer than others, we get 4 P reward
        self.move_scale = P * 5 / log10

    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState
    ):
        self.prev_action_state = self.curr_action_state
        self.prev_percent = self.curr_percent

        self.curr_percent = opponent.percent
        self.curr_action_state = player.action

        self.distance = gamestate.distance

        action_name = self.action_state_counts.action_name(self.curr_action_state)

        if self.prev_action_name != action_name:
            self.action_state_counts.count(action_name)
            self.state_r = self.get_state_reward(self.curr_action_state)
        else:
            self.state_r = 0.

        if self.curr_action_state in ActionStateReward.WALL_TECH_STATES and self.prev_action_state not in ActionStateReward.WALL_TECH_STATES:
            self.wall_techs += 1
            return

        is_fresh_hit, has_move_ended = self.hitting_move_tracker.update(
            self.curr_action_state,
            damage=(self.curr_percent - self.prev_percent),
            distance=self.distance
        )

        if is_fresh_hit:
            self.hit_counts.count(self.prev_action_name)
            self.move_r = self.get_hit_reward(self.prev_action_state)
        else:
            self.move_r = 0.

        self.prev_action_name = action_name


    def reward(
            self,
            advantage: float,
            combo_counter: int
    ) -> float:
        dist_scale = 1. - np.minimum(self.distance / 200, 1.)
        return self.state_r * self.state_scale * dist_scale + self.move_r * self.move_scale

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {}
        # move_accuracies = {
        #     move_name: 100 if self.used_moves[move_name] == 0 else 100 * self.move_hits[move_name] / self.used_moves[move_name]
        #     for move_name in self.move_hits
        # }
        action_state_counts = self.action_state_counts.get()
        return {
            "Wall Tech$game": self.wall_techs,
            # "__move_hits__": self.move_hits,
            # "__move_uses__": self.used_moves,
            #"__move_accuracies__": move_accuracies,

            "__action_state_counts__": action_state_counts,
            "__hit_counts__": self.hit_counts.get(),
            ** super().get_metrics(game_length_s, as_opponent)

        }


class StageControlReward(RewardModule):

    def __init__(self,
        discount: float

    ):
        super().__init__(discount)
        self.controlled_frames = 0

        self.is_controlling = False

        self.scale = 1 * P / 60  # 1 percent reward for each second we control the stage

    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):
        dist_from_0 = np.sqrt(
            player.position.x**2 + player.position.y**2
        )
        opponent_dist_from_0 = np.sqrt(
            opponent.position.x**2 + opponent.position.y**2
        )
        self.is_controlling = dist_from_0 < opponent_dist_from_0
        self.controlled_frames += int(self.is_controlling)

    def reward(
            self,
            advantage: float,
            combo_counter: int
    ) -> float:
        return int(self.is_controlling) * self.scale

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {}

        return {
            "Stage Control%": 100 * self.controlled_frames / (game_length_s * 20 * 3),
                              ** super().get_metrics(game_length_s, as_opponent)

        }


# class OffStageReward(RewardModule):
#     """
#     Keeps track of successful recoveries
#     """
#
#     def __init__(self,
#                  discount: float
#                  ):
#         super().__init__(discount)
#         self.was_pushed_off = False
#         self.is_pushed_off = False
#
#         self.times_offstage = 0
#         self.suicides = 0
#         self.KOs = 0
#         self.prev_stock = 4
#         self.stock = 4
#
#         self.scale = P * 30 # 33 percent reward for pushing off opponents.
#
#     def dist_to_ledge(
#             self,
#             x: float,
#             y: float,
#             gamestate: GameState
#     ):
#         abs_x = abs(x)
#         edge_pos = EDGE_POSITION[gamestate.stage]
#         return abs(abs_x - edge_pos) if y >= 0 else np.sqrt((abs_x - edge_pos) ** 2 + y ** 2)
#
#     def update(
#             self,
#             player: PlayerState,
#             opponent: PlayerState,
#             gamestate: GameState,
#     ):
#         """
#         Incentives the bot to go back on stage/edge
#         as well as prevent the opponent from getting the ledge and going back on stage.
#         """
#         self.prev_stock = self.stock
#         self.was_pushed_off = self.is_pushed_off
#         self.stock = player.stock
#         is_ejected = abs(player.speed_y_attack) + abs(player.speed_x_attack) > 0
#         dist_to_ledge = self.dist_to_ledge(player.position.x, player.position.y, gamestate)
#
#         if self.stock < self.prev_stock:
#             if self.was_pushed_off:
#                 self.KOs += 1
#             else:
#                 self.suicides += 1
#
#
#         # Only mark the player as offstage here if it was ejected and far from the ledge
#         if player.off_stage and is_ejected and dist_to_ledge > 55:
#             if not self.is_pushed_off:
#                 self.times_offstage += 1
#             self.is_pushed_off = True
#             return
#
#         # set the flag to off if we successfully recovered.
#         if player.action in NEUTRAL_GROUND_ACTIONS or player.action == Action.EDGE_CATCHING:
#             self.is_pushed_off = False
#
#     def reward(
#             self,
#             advantage: float,
#             combo_counter: int
#     ) -> float:
#
#         return (
#                 (int(self.was_pushed_off) - int(self.is_pushed_off)) * self.scale # reward for coming back and penalty for getting pushed off
#         )
#
#     def get_metrics(
#             self,
#             game_length_s: float,
#             as_opponent: bool = False
#     ) ->  Dict[str, float | Dict[str, float]]:
#         if as_opponent:
#             return {
#                 "Successful Edgeguard%": 0 if self.times_offstage == 0 else 100 * self.KOs / self.times_offstage
#             }
#         else:
#             return {
#                 "Successful Recoveries%": 100 if self.times_offstage == 0 else 100 * (1 - (self.KOs / self.times_offstage)),
#                 "Self-Destructs$game": self.suicides,
#             ** super().get_metrics(game_length_s, as_opponent)
#             }


class OffStageReward(RewardModule):

    def __init__(
            self,
            discount: float
    ):
        super().__init__(discount)
        self.prev_percent = 0
        self.percent = 0
        self.offstage = False
        self.prev_action = Action.SHIELD
        self.curr_action = Action.SHIELD
        self.edge_intangible_frames = 0
        self.is_hanging_right = False
        self.edge_intangible_scale = 5 * P / 60
        self.hit_offstage_scale = P * 12.

        self.offstage_aggressions = 0
        self.has_hit_offstage = False
        self.hit_offstage_hitstun = 0
        self.opponent_offstage = False
        self.opponent_hitstun = 0
        self.distance = 100.



    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):

        self.prev_percent = self.percent
        self.prev_action = self.curr_action
        self.curr_action = player.action
        self.percent = opponent.percent

        self.is_hanging_right = (
                player.invulnerability_type in (InvulnerabilityType.INTANGIBLE, InvulnerabilityType.INVULNERABLE)
                and player.action in (Action.EDGE_HANGING, Action.EDGE_CATCHING)
                and opponent.off_stage
        )
        if self.is_hanging_right:
            self.edge_intangible_frames += 1

        self.has_hit_offstage = (
            self.opponent_offstage
            and opponent.hitstun_frames_left > self.opponent_hitstun
            and opponent.off_stage
            and player.off_stage
        )
        if self.has_hit_offstage:
            self.offstage_aggressions += 1
            self.hit_offstage_hitstun = opponent.hitstun_frames_left - self.opponent_hitstun

        self.opponent_hitstun = opponent.hitstun_frames_left
        self.opponent_offstage = opponent.off_stage
        self.distance = gamestate.distance

    def reward(
            self,
            advantage: float,
            combo_counter: int
    ) -> float:

        offstage_hit_reward = np.minimum(float(self.has_hit_offstage) * self.hit_offstage_hitstun / 20, 1.)
        hanging_reward = float(self.is_hanging_right) * np.minimum(1 - self.distance/100, 0.)

        return (
           hanging_reward * self.edge_intangible_scale + offstage_hit_reward * self.hit_offstage_scale
        )

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {
                #"Offstage Damage Incurred$game": self.total_offstage_damage,
            }

        return {
            "Edge Intangible Frames$game": self.edge_intangible_frames,
            "Off-Stage Aggressions$game": self.offstage_aggressions,
            ** super().get_metrics(game_length_s, as_opponent)
        }


class NeutralGameReward(RewardModule):

    def __init__(self,
                 discount: float
                 ):
        super().__init__(discount)
        self.was_neutral = True
        self.is_neutral = True
        self.opponent_was_neutral = True
        self.opponent_is_neutral = True

        self.distance = 0

        self.no_jump = False
        self.neutral_wins = 0
        self.neutral_losses = 0

        self.scale = 5 * P # rewards for winning a neutral trade, or getting out of a non-neutral state.

    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):
        self.was_neutral = self.is_neutral
        self.distance = gamestate.distance
        if (
            player.action in NEUTRAL_ACTIONS and
            player.hitstun_frames_left == 0
            #and (player.on_ground or (not player.off_stage and gamestate.distance > 120))
        ):
            self.is_neutral = True

        if (
            player.hitstun_frames_left > 5
            # or (player.action == Action.SHIELD_STUN and player.shield_strength < 30)
        ):
            if self.is_neutral:
                self.neutral_losses += 1
            self.is_neutral = False
        ###
        self.opponent_was_neutral = self.opponent_is_neutral
        if (
                opponent.action in NEUTRAL_ACTIONS and
                opponent.hitstun_frames_left == 0
                # and opponent.on_ground
        ):
            self.opponent_is_neutral = True

        if (
                opponent.hitstun_frames_left > 5
                # or (opponent.action == Action.SHIELD_STUN and opponent.shield_strength < 30)
        ):
            if self.opponent_is_neutral:
                self.neutral_wins += 1
            self.opponent_is_neutral = False

    def reward(
            self,
            advantage: float,
            combo_counter: int
    ) -> float:
        #scale = 1 - np.minimum(self.distance / 100, 1)
        return (int(self.is_neutral) - int(self.was_neutral)) * self.scale

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {}

        return {
            "Neutral Win%": 50 if self.neutral_wins + self.neutral_losses == 0 else
            100 * self.neutral_wins / (self.neutral_wins + self.neutral_losses),
            **super().get_metrics(game_length_s, as_opponent)

        }

class LedgeRewards(RewardModule):
    """Disable these for now"""
    pass


class PlayerAdvantages:
    def __init__(self, percent_cap: int = 100):
        self.percent_cap = percent_cap
        self.p1_smooth_stocks = 4
        self.p2_smooth_stocks = 4

    def cap_percent(self, percent: int):
        return np.minimum(percent, self.percent_cap)

    def compute_player_advantage(self, gamestate: GameState) -> float:
        # 3 sec interval for stock trading
        lr = 0.0045

        if len(gamestate.players) < 2:
            return 0.

        p1 = gamestate.players[1]
        p2 = gamestate.players[2]

        self.p1_smooth_stocks = self.p1_smooth_stocks * (1 - lr) + p1.stock * lr
        self.p2_smooth_stocks = self.p2_smooth_stocks * (1 - lr) + p2.stock * lr

        p1_stock = round(self.p1_smooth_stocks)
        p2_stock = round(self.p2_smooth_stocks)
        if p1_stock == p2_stock:
            # Last stock, no trading
            if p1_stock == 1:
                return 0
            return (self.cap_percent(p1.percent) - self.cap_percent(p2.percent)) / (self.percent_cap * 5)
        else:
            return (p1_stock - p2_stock) / (p1_stock + p2_stock)


def reward_weighted_sum(rewards: StepRewards, weights: StepRewards) -> float:
    return sum(tree.map_structure(lambda r, w: r*w, rewards, weights ).values())


class PlayerRewards:
    def __init__(self, bot: Bot, elo_delta: float):
        super().__init__()
        preferences = bot.stats

        return_scales: StepRewards = bot.return_scales

        main_scale = return_scales["stock_rewards"]
        damage_target_scale = main_scale / 5

        damage_scale = damage_target_scale / (return_scales["damage_rewards"] + 1e-8)

        # closeup rewards are used to make bots interact early on:
        offset = preferences.aggressivity/1000

        closeup_scale = (damage_target_scale / 2) / (return_scales["closeup_rewards"] + 1e-8)

        action_state_scale = (damage_target_scale / 8) / (return_scales["action_state_rewards"] + 1e-8)
        stage_control_scale = 0. * (damage_target_scale / 10) / (return_scales["stage_control_rewards"] + 1e-8)
        offstage_scale = np.minimum((damage_target_scale / 10) / (return_scales["offstage_rewards"] + 1e-8), 3.)
        neutral_scale = 0. * (damage_target_scale / 3) / (return_scales["neutral_rewards"] + 1e-8)
        techskill_scale = (damage_target_scale / 9) / (return_scales["techskill_rewards"] + 1e-8)

        def log_scale(stat, low=0.1, high=5., zero_zero=True):
            x = (stat - 50) / 50
            if zero_zero and stat == 0:
                return 0.
            if x < 0.:
                return np.exp(-x * np.log(low))
            else:
                return np.exp(x * np.log(high))

        win_prob = SeedSmashMatchmaking.expected_outcome(elo_delta)
        p_scale = (1-win_prob) * 0.1 + 0.45

        n_scale =  win_prob * 0.1 + 0.45 - offset / 2

        self.weights: StepRewards = {
            "win_rewards": 1.,
            "stock_rewards": 1.,
            "sd_rewards": -1.,
            "damage_rewards": damage_scale,
            "action_state_rewards": action_state_scale * log_scale(preferences.creativity, low=0.3, high=2.),
            "closeup_rewards": closeup_scale, #log_scale(preferences.aggressivity, high=3.0),
            "stalling_rewards": 1.,  # log_scale(preferences.aggressivity, high=3.0),
            "stage_control_rewards": stage_control_scale * log_scale(preferences.stagecontrol, low=0.3, high=2.),
            "offstage_rewards": offstage_scale * log_scale(preferences.offstage, low=0.3, high=2.),
            "neutral_rewards": neutral_scale * log_scale(preferences.neutral, low=0.1, high=2.),
            "techskill_rewards": techskill_scale * log_scale(preferences.techskill, low=0.2, high=2.),
        }

        self.opponent_zeros: StepRewards = {
            "win_rewards": 1.,
            "stock_rewards": 1.,
            "sd_rewards": 0.,
            "damage_rewards": 1.,
            "action_state_rewards": 0.,
            "closeup_rewards": 0.,
            "stage_control_rewards": 1.,
            "stalling_rewards": 0.,
            "offstage_rewards": 0.,
            "neutral_rewards": 1.,
            "techskill_rewards": 0.,
        }

        self.p_scales: StepRewards = {
            "win_rewards": np.maximum((1-win_prob), 1/4),
            "stock_rewards": p_scale,
            "sd_rewards": n_scale,
            "damage_rewards": p_scale,
            "action_state_rewards": 1.,
            "closeup_rewards": 1.,
            "stalling_rewards": 1.,
            "stage_control_rewards": p_scale,
            "offstage_rewards": p_scale,
            "neutral_rewards": p_scale,
            "techskill_rewards": 1.,
        }

        self.n_scales: StepRewards = {
            "win_rewards": np.maximum(win_prob, 1/4),
            "stock_rewards": n_scale,
            "sd_rewards": p_scale,
            "damage_rewards": n_scale,
            "action_state_rewards": 1.,
            "closeup_rewards": 1.,
            "stalling_rewards": 1.,
            "stage_control_rewards": n_scale,
            "offstage_rewards": n_scale,
            "neutral_rewards": n_scale,
            "techskill_rewards": 1.,
        }

        self.modules: StepRewards = {
            "win_rewards": WinReward(bot.discount),
            "stock_rewards": StockReward(bot.discount),
            "sd_rewards": SDReward(bot.discount),
            "damage_rewards": DamageReward(bot.character, bot.discount),
            "action_state_rewards": ActionStateReward(bot, bot.discount),
            "closeup_rewards": CloseupReward(bot.discount, bot.version, bot.stats.aggressivity),
            "stalling_rewards": StallingReward(bot.discount),
            "stage_control_rewards": StageControlReward(bot.discount),
            "offstage_rewards": OffStageReward(bot.discount),
            "neutral_rewards": NeutralGameReward(bot.discount),
            "techskill_rewards": Techskill(bot.character, bot.discount)
        }

        for name in self.modules:
            self.modules[name].register_as(name, self.weights[name])

    def accumulate(
            self,
            step_reward: StepRewards,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
            advantage: float,
            combo_counter: int,
    ):

        for name, module in self.modules.items():
            module.update(player, opponent, gamestate)
            r = module.reward(advantage, combo_counter)
            step_reward[name] += r

        return step_reward

    def zero_sum(self, self_rewards: StepRewards, opponent_rewards: StepRewards):
        r = 0
        for name, module in self.modules.items():
            # unscaled, zero_sum magnitude
            unscaled_zero_sum_r = self_rewards[name] - opponent_rewards[name] * self.opponent_zeros[name]
            module.track_magnitude(unscaled_zero_sum_r)

            zero_sum_r = module.w * unscaled_zero_sum_r

            # bias depending on the win probs
            r += np.maximum(zero_sum_r, 0.) * self.p_scales[name] + np.minimum(zero_sum_r, 0.) * self.n_scales[name]

        return r

    def on_episode_end(self):
        for r_module in self.modules.values():
            r_module.on_episode_end()

    def get_metrics(self, episode_length: int, opponent_rewards: "PlayerRewards"):

        d = {}
        game_length_s = episode_length / 20
        for module in self.modules.values():
            d.update(module.get_metrics(game_length_s, as_opponent=False))
        for module in opponent_rewards.modules.values():
            d.update(module.get_metrics(game_length_s, as_opponent=True))
        return d

class RewardFunction:
    def __init__(
            self,
            options: Dict[Any, Bot]
    ):

        self.per_player = {
            p: PlayerRewards(option, options[p].elo - options[(p % 2) + 1].elo)
            for p, option in options.items()
        }

        self.player_advantage = PlayerAdvantages()

    def accumulate(
            self,
            step_rewards: Dict[Any, StepRewards],
            gamestate: GameState,
    ):
        advantage = self.player_advantage.compute_player_advantage(gamestate)
        for port, rew_func in self.per_player.items():
            other_port = 1 + (port % 2)
            adv = advantage if port == 1 else - advantage
            rew_func.accumulate(
                step_rewards[port],
                gamestate.players[port],
                gamestate.players[other_port],
                gamestate,
                adv,
                gamestate.players[port].custom["combo_counter"]
            )

    def zero_sum(
            self,
            step_rewards: Dict[Any, StepRewards]
    ) -> Dict[Any, float]:
        r = {
            port: rew_func.zero_sum(step_rewards[port], step_rewards[(port % 2) + 1])
            for port, rew_func in self.per_player.items()
        }

        return r

    def on_episode_end(self):
        for rew_func in self.per_player.values():
            rew_func.on_episode_end()

    def get_metrics(
            self,
            episode_length: int
    ) -> Dict[Any, Dict[str, float]]:
        return {
            port: rew_func.get_metrics(episode_length, self.per_player[(port % 2) + 1])
            for port, rew_func in self.per_player.items()
        }










