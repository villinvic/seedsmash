from pprint import pprint
from typing import Dict, TypedDict, Any

import numpy as np
import tree
from melee import PlayerState, GameState, Action, InvulnerabilityType, EDGE_POSITION, character_moves
from polaris_melee.observation_space import ObsBuilder
from polaris_melee.rewards_core import StepRewards, RewardModule, NEUTRAL_ACTIONS, GETUP_ATTACKS, ROLL_STATES
from polaris_melee.techskill import Techskill
from seedsmash.bot import Bot
from seedsmash.utils import action_idx



class CloseupReward(RewardModule):

    def __init__(self):
        super().__init__()
        self.self_position = None
        self.opponent_position = None

        self.y_scale = 0.25
        self.closeup = 0.
        self.total_closeup = 0.

    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):

        if self.self_position is not None:
            prev_dx = self.self_position.x - self.opponent_position.x
            prev_dy = (self.self_position.y - self.opponent_position.y) * self.y_scale
            # freeze the other player's position to get the velocity of the player at given port.
            dx = player.position.x - self.opponent_position.x
            dy = (player.position.y - self.opponent_position.y) * self.y_scale

            prev_dist = np.sqrt(
                prev_dx ** 2 + prev_dy ** 2
            )
            dist = np.sqrt(
                dx ** 2 + dy ** 2
            )
            closeup = prev_dist - dist

            # We do not want to reward closeups with roll states
            if (closeup > 0 and player.action not in ROLL_STATES) or closeup < 0:

                # Reward running away when the opponent is invulnerable.
                if (opponent.invulnerability_type == InvulnerabilityType.INVULNERABLE
                        and player.invulnerability_type != InvulnerabilityType.INVULNERABLE):
                    closeup = -closeup

            else:
                closeup = 0

            self.closeup = closeup
            self.total_closeup += closeup

        self.self_position = player.position
        self.opponent_position = opponent.position

    def reward(
            self,
            advantage: float,
            opponent_combo_counter: int
    ) -> float:
        return self.closeup * 2e-3

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {}
        return {
            "Closeup/s": self.total_closeup / game_length_s
        }


class CoreReward(RewardModule):

    def __init__(self):
        super().__init__()
        self.prev_stock = 4
        self.stock = 4
        self.prev_percent = 0
        self.percent = 0

        self.is_opponent_getup_attack = False
        self.is_opponent_intangible = False
        self.offstage = False

        self.total_damage_offstage = 0
        self.total_damage = 0
        self.deaths = []
        self.total_intangible_damage = 0


    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):
        self.prev_stock = self.stock
        self.prev_percent = self.percent

        self.stock = player.stock
        self.percent = player.percent

        self.is_opponent_getup_attack = opponent.action in GETUP_ATTACKS
        self.offstage = (player.off_stage and opponent.off_stage)
        self.is_opponent_intangible = opponent.invulnerability_type == InvulnerabilityType.INTANGIBLE

        if self.stock - self.prev_stock > 0:
            self.deaths.append(self.prev_percent)

    def reward(
            self,
            advantage: float,
            opponent_combo_counter: int
    ) -> float:

        # Using the advantage here allows for bots to learn trading stocks when it is worth.
        death = np.maximum(self.prev_stock - self.stock, 0.) * (1 - advantage)

        damage = np.maximum(self.percent - self.prev_percent, 0.)
        combo_boost = 1 + opponent_combo_counter / ObsBuilder.MAX_COMBO
        damage_r = combo_boost * damage

        # Encourage bots to exploit intangibility from the ledge
        if (not self.is_opponent_getup_attack) and self.is_opponent_intangible:
            damage_r *= 1.5
            self.total_intangible_damage += damage
        if self.offstage:
            # TODO: I think offstage here is not doing great
            damage_r *= (1 - advantage)
            self.total_intangible_damage += damage

        self.total_damage += damage

        return -(
            death + damage * 0.005
        )

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {
                "Average Kill%": self.total_damage * 2 if len(self.deaths) == 0 else np.mean(self.deaths),
                "Damage Dealt/s": self.total_damage / game_length_s,
                "Intangible Damage Dealt/game": self.total_intangible_damage,
                "Offstage Damage Dealt/game": self.total_damage_offstage,
            }

        return {
            "Average Death%": self.total_damage * 2 if len(self.deaths) == 0 else np.mean(self.deaths),
            "Suicides/game": sum([int(p<3) for p in self.deaths]),
            "Damage Incurred/s": self.total_damage / game_length_s,
            "Intangible Damage Incurred/game": self.total_intangible_damage,
            "Offstage Damage Incurred/game": self.total_damage_offstage,
        }


class ActionStateReward(RewardModule):

    # Could be a creativity stat later

    WALL_TECH_STATES = [
        Action.WALL_TECH,
        Action.WALL_TECH_JUMP,
        Action.CEILING_TECH
    ]
    # TODO: metrics

    def __init__(
            self,
            bot: Bot
    ):
        self.values = bot.action_state_counts.get_values()
        self.hit_values = bot.action_state_hit_counts.get_values()

        self.prev_action_state = Action.FALLING
        self.curr_action_state = Action.FALLING
        self.prev_percent = 0
        self.curr_percent = 0

        self.is_hitting = False

        self.action_state_counts = np.zeros((len(Action),), dtype=np.int32)
        self.action_state_hit_counts = np.zeros((len(Action),), dtype=np.int32)

        self.char_moves = character_moves[bot.character]
        self.used_moves = {
            move.name: 0
            for move in self.char_moves
        }

        self.move_hits = self.used_moves.copy()
        self.wall_techs = 0

        self.hit_move = Action.SHIELD
        self.has_move_hit = False

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

        self.action_state_counts[action_idx[self.curr_action_state]] += 1

        if self.curr_action_state in ActionStateReward.WALL_TECH_STATES:
            self.wall_techs += 1
            return


        # filter out 1 damage moves for out of edge damage
        damage = (self.curr_percent - self.prev_percent)
        self.is_hitting = damage > 1
        if self.is_hitting:
            self.action_state_hit_counts[action_idx[self.prev_action_state]] += 1

        try:
            # count move if it was just actioned:
            if self.curr_action_state != self.prev_action_state:
                self.has_move_hit = False
                move = self.char_moves(self.curr_action_state.value)
                self.used_moves[move.name] += 1
                # reset the hit counts
            # count move hit if it did not hit already
            if self.is_hitting:
                move = self.char_moves(self.prev_action_state.value)
                self.move_hits[move.name] += int(self.is_hitting and not self.has_move_hit)
                self.has_move_hit = True
        except ValueError:
            pass

    def reward(
            self,
            advantage: float,
            opponent_combo_counter: int
    ) -> float:
        r = self.values(self.curr_action_state)
        if self.is_hitting:
            # use prev action if moves finished this frame (move interrupted)
            r += self.hit_values(self.prev_action_state)

        return r

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {}
        most_hit_moves = dict(sorted(self.move_hits.items(), key=lambda item: -item[1]))
        most_used_moves = dict(sorted(self.used_moves.items(), key=lambda item: -item[1]))
        move_accuracies = {
            move_name: 100 * self.move_hits[move_name] / self.used_moves[move_name]
            for move_name in self.move_hits
            if self.move_hits[move_name] > 0
        }
        least_accurate_moves = dict(sorted(move_accuracies.items(), key=lambda item: item[1]))
        # move accuracy
        # move damage
        return {
            "Most Hit Moves (By Hit Count)": most_hit_moves,
            "Most Used Moves (By Usage Count)": most_used_moves,
            "Least Accurate Moves (By Accuracy%)": least_accurate_moves,

            "__action_state_counts__": self.action_state_counts,
            "__action_state_hit_counts__": self.action_state_hit_counts

        }


class StageControlReward(RewardModule):

    def __init__(self):

        self.controlled_frames = 0

        self.is_controlling = False

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
            opponent_combo_counter: int
    ) -> float:
        return 0.003 * 0.1 * int(self.is_controlling)

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {}

        return {
            "Stage Control%": 100 * self.controlled_frames / (game_length_s * 20 * 3)
        }


class OffStageReward(RewardModule):
    """
    Keeps track of successful recoveries
    """

    def __init__(self):

        self.was_pushed_off = False
        self.is_pushed_off = False

        self.times_offstage = 0
        self.KOs = 0
        self.prev_stock = 4
        self.stock = 4

    def dist_to_ledge(
            self,
            x: float,
            y: float,
            gamestate: GameState
    ):
        abs_x = abs(x)
        edge_pos = EDGE_POSITION[gamestate.stage]
        return abs(abs_x - edge_pos) if y >= 0 else np.sqrt((abs_x - edge_pos) ** 2 + y ** 2)

    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):
        """
        Incentives the bot to go back on stage/edge
        as well as prevent the opponent from getting the ledge and going back on stage.
        """
        self.prev_stock = self.stock
        self.was_pushed_off = self.is_pushed_off
        self.stock = player.stock
        is_ejected = abs(player.speed_y_attack) + abs(player.speed_x_attack) > 0
        dist_to_ledge = self.dist_to_ledge(player.position.x, player.position.y, gamestate)

        if self.stock - self.prev_stock > 0 and self.was_pushed_off:
            self.KOs += 1

        # Only mark the player as offstage here if it was ejected and far from the ledge
        if player.off_stage and is_ejected and dist_to_ledge > 30:
            if not self.is_pushed_off:
                self.times_offstage += 1
            self.is_pushed_off = True
            return

        # set the flag to off if we successfully recovered.
        if not player.off_stage or player.action == Action.EDGE_CATCHING:
            self.is_pushed_off = False

    def reward(
            self,
            advantage: float,
            opponent_combo_counter: int
    ) -> float:
        return np.maximum(int(self.was_pushed_off) - int(self.is_pushed_off), 0.) * 0.2

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {
                "Successful Edgeguard%": 0 if self.times_offstage == 0 else 100 * self.KOs / self.times_offstage
            }
        else:
            return {
                "Successful Recoveries%": 100 if self.times_offstage == 0 else 100 * (1 - (self.KOs / self.times_offstage))
            }


class NeutralGameReward(RewardModule):

    def __init__(self):
        self.was_neutral = True
        self.is_neutral = True
        self.opponent_was_neutral = True
        self.opponent_is_neutral = True

        self.neutral_wins = 0
        self.neutral_losses = 0

    def update(
            self,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):
        self.was_neutral = self.is_neutral
        if (
            player.action in NEUTRAL_ACTIONS and
            player.hitstun_frames_left == 0 and
            player.on_ground
        ):
            self.is_neutral = True

        if (
            player.hitstun_frames_left > 6 or
            (player.action == Action.SHIELD_STUN and player.shield_strength < 30)
        ):
            if self.is_neutral:
                self.neutral_losses += 1
            self.is_neutral = False
        ###
        self.opponent_was_neutral = self.opponent_is_neutral
        if (
                opponent.action in NEUTRAL_ACTIONS and
                opponent.hitstun_frames_left == 0 and
                opponent.on_ground
        ):
            self.opponent_is_neutral = True

        if (
                opponent.hitstun_frames_left > 6 or
                (opponent.action == Action.SHIELD_STUN and opponent.shield_strength < 30)
        ):
            if self.opponent_is_neutral:
                self.neutral_wins += 1
            self.opponent_is_neutral = False

    def reward(
            self,
            advantage: float,
            opponent_combo_counter: int
    ) -> float:
        # -1 for loosing neutral.
        return np.minimum(float(self.is_neutral) - float(self.was_neutral), 0.) * 0.03

    def get_metrics(
            self,
            game_length_s: float,
            as_opponent: bool = False
    ) ->  Dict[str, float | Dict[str, float]]:
        if as_opponent:
            return {}

        return {
            "Neutral Win%": 50 if self.neutral_wins + self.neutral_losses == 0 else
            100 * self.neutral_wins / (self.neutral_wins + self.neutral_losses)
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
    def __init__(self, bot: Bot):

        preferences = bot.stats
        self.modules: StepRewards = {
            "core_rewards": CoreReward(),
            "action_state_rewards": ActionStateReward(bot),
            "closeup_rewards": CloseupReward(),
            "stage_control_rewards": StageControlReward(),
            "offstage_rewards": OffStageReward(),
            "neutral_rewards": NeutralGameReward(),
            "techskill_rewards": Techskill(bot.character)
        }

        def log_scale(stat, low=5e-2, high=5):
            x = (stat - 50) / 50
            if x < 0.:
                return np.exp(-x * np.log(low))
            else:
                return np.exp(x * np.log(high))


        self.weights: StepRewards = {
            "core_rewards": 1.,
            "action_state_rewards": 1.,
            "closeup_rewards": log_scale(preferences.aggressivity),
            "stage_control_rewards": log_scale(preferences.stagecontrol),
            "offstage_rewards": log_scale(preferences.offstage),
            "neutral_rewards": log_scale(preferences.neutral),
            "techskill_rewards": log_scale(preferences.techskill),
        }

    def accumulate(
            self,
            step_reward: StepRewards,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
            advantage: float,
            opponent_combo_counter: int,
    ):

        for name, module in self.modules.items():
            module.update(player, opponent, gamestate)
            step_reward[name] += module.reward(advantage, opponent_combo_counter)

        return step_reward

    def zero_sum(self, self_rewards: StepRewards, opponent_rewards: StepRewards):
        return reward_weighted_sum(self_rewards, self.weights) - reward_weighted_sum(opponent_rewards, self.weights)

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
            p: PlayerRewards(option)
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
            adv = advantage if port == 1 else -advantage
            rew_func.accumulate(
                step_rewards[port],
                gamestate.players[port],
                gamestate.players[other_port],
                gamestate,
                adv,
                gamestate.custom["combo_counters"][other_port]
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

    def get_metrics(
            self,
            episode_length: int
    ) -> Dict[Any, Dict[str, float]]:
        return {
            port: rew_func.get_metrics(episode_length, self.per_player[(port % 2) + 1])
            for port, rew_func in self.per_player.items()
        }









