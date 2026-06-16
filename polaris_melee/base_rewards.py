from typing import Dict, Any

import numpy as np
import tree
from melee import PlayerState, GameState, Action, InvulnerabilityType, character_moves, Character, \
    YoshiMoves, Stage, stages, Position
from polaris_melee.observation_space import ObsBuilder
from polaris_melee.rewards_core import StepRewards, RewardModule, NEUTRAL_ACTIONS, GETUP_ATTACKS, ROLL_STATES, P, D, W
from polaris_melee.utils import HittingMoveTracker
from seedsmash.bot import Bot


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


    def reward(self) -> float:

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


    def reward(self) -> float:

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


    def reward(self) -> float:

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

    def reward(self) -> float:

        death = int(self.stock < self.prev_stock)

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

    def reward(self) -> float:

        damage = np.maximum(self.percent - self.prev_percent, 0.)

        if self.offstage:
            self.total_offstage_offstage += damage

        self.total_damage += damage

        blocking = float(self.curr_action == self.shield_block_state and self.prev_action in self.shield_states)
        self.num_shield_blocks += blocking

        return P * damage

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


    def reward(self) -> float:
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


class LedgeRewards(RewardModule):
    """Disable these for now"""
    pass


def reward_weighted_sum(rewards: StepRewards, weights: StepRewards) -> float:
    return sum(tree.map_structure(lambda r, w: r*w, rewards, weights).values())


class PlayerRewards:
    def __init__(self, bot: Bot):
        super().__init__()

        self.weights: StepRewards = {
            #"win_rewards": 1.,
            "stock_rewards": 1.,
            #"sd_rewards": -1.,
            "damage_rewards": 1,
            # "action_state_rewards": action_state_scale * log_scale(preferences.creativity, low=0.3, high=2.),
            # "closeup_rewards": closeup_scale, #log_scale(preferences.aggressivity, high=3.0),
            # "stalling_rewards": 1.,  # log_scale(preferences.aggressivity, high=3.0),
            # "stage_control_rewards": stage_control_scale * log_scale(preferences.stagecontrol, low=0.3, high=2.),
            # "offstage_rewards": offstage_scale * log_scale(preferences.offstage, low=0.3, high=2.),
            # "neutral_rewards": neutral_scale * log_scale(preferences.neutral, low=0.1, high=2.),
            # "techskill_rewards": techskill_scale * log_scale(preferences.techskill, low=0.2, high=2.),
        }

        self.opponent_zeros: StepRewards = {
            #"win_rewards": 1.,
            "stock_rewards": 1.,
            #"sd_rewards": 0.,
            "damage_rewards": 1.,
            # "action_state_rewards": 0.,
            # "closeup_rewards": 0.,
            # "stage_control_rewards": 1.,
            # "stalling_rewards": 0.,
            # "offstage_rewards": 0.,
            # "neutral_rewards": 1.,
            # "techskill_rewards": 0.,
        }

        self.modules: StepRewards = {
            #"win_rewards": WinReward(bot.discount),
            "stock_rewards": StockReward(bot.discount),
            #"sd_rewards": SDReward(bot.discount),
            "damage_rewards": DamageReward(bot.character, bot.discount),
            # "action_state_rewards": ActionStateReward(bot, bot.discount),
            # "closeup_rewards": CloseupReward(bot.discount, bot.version, bot.stats.aggressivity),
            # "stalling_rewards": StallingReward(bot.discount),
            # "stage_control_rewards": StageControlReward(bot.discount),
            # "offstage_rewards": OffStageReward(bot.discount),
            # "neutral_rewards": NeutralGameReward(bot.discount),
            # "techskill_rewards": Techskill(bot.character, bot.discount)
        }

        for name in self.modules:
            self.modules[name].register_as(name, self.weights[name])

    def accumulate(
            self,
            step_reward: StepRewards,
            player: PlayerState,
            opponent: PlayerState,
            gamestate: GameState,
    ):

        for name, module in self.modules.items():
            module.update(player, opponent, gamestate)
            r = module.reward()
            step_reward[name] += r

        return step_reward

    def zero_sum(self, self_rewards: StepRewards, opponent_rewards: StepRewards):
        r = 0
        for name, module in self.modules.items():
            # unscaled, zero_sum magnitude
            unscaled_zero_sum_r = self_rewards[name] - opponent_rewards[name] * self.opponent_zeros[name]
            module.track_magnitude(unscaled_zero_sum_r)

            zero_sum_r = module.w * unscaled_zero_sum_r

            r += zero_sum_r

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
            p: PlayerRewards(option)
            for p, option in options.items()
        }

    def accumulate(
            self,
            step_rewards: Dict[Any, StepRewards],
            gamestate: GameState,
    ):
        for port, rew_func in self.per_player.items():
            other_port = 1 + (port % 2)
            rew_func.accumulate(
                step_rewards[port],
                gamestate.players[port],
                gamestate.players[other_port],
                gamestate
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










