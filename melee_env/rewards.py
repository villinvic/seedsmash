import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Union, Dict

from Cython import other_types
from melee import GameState, Character, Action, InvulnerabilityType
import numpy as np

from melee_env.action_space import ControllerInput
from melee_env.melee_helpers import MeleeHelper
from melee_env.observation_space import ObsBuilder


@dataclass
class StepRewards(
):
    """
    Databag for player rewards
    """

    # TODO : reward for hitstun frames !
    # TODO : WALL_TECH = 0xca
    #     WALL_TECH_JUMP = 0xcb
    #     SHIELD_REFLECT = 0xb6
    # TODO : Reward reaching attacks (in_range)
    # Keep track of wall techs, techs, reward for wall techs and wall jumps

    # """
    # Port of the player
    # """
    # port: int

    """
    rewards collected for damaging/taking damage
    """
    damage_inflicted_rewards: np.float32 = 0.
    damage_received_rewards: np.float32 = 0.

    """
    rewards collected for moving toward the opponent
    TODO: add in bot config somehow
    """
    distance_rewards: np.float32 = 0.

    """
    rewards collected for killing/dying
    """
    kill_rewards: np.float32 = 0.
    death_rewards: np.float32 = 0.

    """
    Cost incurred for sweating on the c-stick or the buttons
    """
    energy_costs: np.float32 = 0.

    """
    Off-stage percents inflicted when both players are off-stage
    """
    off_stage_percents_inflicted: np.float32 = 0.

    bad_edge_catches: np.float32 = 0.
    edge_while_opp_invulnerable: np.float32 = 0.
    edge_guarding: np.float32 = 0.
    shield_pressured: np.float32 = 0.
    shield_pressuring: np.float32 = 0.

    helper_bonus: np.float32 = 0.


    # could help
    non_hitting_attacks: np.float32 = 0.


    # TODO: add a possibility to add a scaling between damage and stocks

    def to_dict(self):
        d = asdict(self)
        return d



class DeltaFrame:
    FRAMES_BEFORE_SUICIDE = 60 * 7

    def __init__(self):
        self.last_frame: Union[GameState, None] = None
        self.episode_finished = False

        self.ports = {1, 2}
        self.zero()

    def zero(self):

        self.win = {
            p: 0 for p in self.ports
        }

        self.dstock = {
            p: 0 for p in self.ports
        }

        self.dpercent = {
            p: 0 for p in self.ports
        }

        self.last_hit_frame = {
            p: 0 for p in self.ports
        }

        self.ddist = {
            p: 0 for p in self.ports
        }
        self.dist = {
            p: 0 for p in self.ports
        }

        self.action = {
            p: Action.KIRBY_BLADE_UP for p in self.ports
        }

        self.histun = {
            p: False for p in self.ports
        }

        self.off_stage = False
        self.percent = {
            p: 0 for p in self.ports
        }

    def update(self, frame: GameState):
        if self.episode_finished or len(frame.players) < 2:
            self.zero()
            return

        if self.last_frame is not None:
            dead1 = frame.players[1].stock == 0
            dead2 = frame.players[2].stock == 0
            if dead1 and dead2:
                # weird ?
                self.episode_finished = True
                self.zero()
                return
            elif dead1 or dead2:
                self.episode_finished = True

                self.win[1] = np.float32(dead2)
                self.win[2] = np.float32(dead1)

            # dx_before = np.abs(self.last_frame.players[1].position.x - self.last_frame.players[2].position.x)
            # dy_before = np.abs(self.last_frame.players[1].position.y - self.last_frame.players[2].position.y)
            dx_before = (self.last_frame.players[1].position.x - self.last_frame.players[2].position.x)
            dx2 = dx_before**2
            dy2 = (self.last_frame.players[1].position.y - self.last_frame.players[2].position.y)**2
            dist_before = np.sqrt(dx2 + 0.5*dy2) #np.sqrt(dx2 + 0.1 * dy2)

            for port in self.ports:
                other_port = 1 + (port % 2)
                self.dist[port] = np.sqrt(dx2 + dy2)
                # dx_now = np.abs(frame.players[port].position.x - self.last_frame.players[other_port].position.x)
                # dy_now = np.abs(frame.players[port].position.y - self.last_frame.players[other_port].position.y)
                dx_now = frame.players[port].position.x - self.last_frame.players[other_port].position.x
                dist_now = np.sqrt(
                    dx_now**2 + 0.5*((frame.players[port].position.y - self.last_frame.players[other_port].position.y) ** 2))
                # 0.1

                self.dstock[port] = np.clip(self.last_frame.players[port].stock - frame.players[port].stock, 0., 1.)
                self.percent[port] = frame.players[port].percent
                self.dpercent[port] = np.clip(frame.players[port].percent - self.last_frame.players[port].percent, 0., 50.)
                if self.dpercent[port] > 0 and self.dist[port] < 70:
                    self.last_hit_frame[port] = frame.frame

                self.histun[port] = (self.last_frame.players[port].hitstun_frames_left > 1e-3)

                # DISTANCE


                # ddist = (dx_before - dx_now) + (dy_before - dy_now) * 0.05
                # if frame.players[port] in [Action.ROLL_FORWARD, Action.ROLL_BACKWARD]:
                #     self.ddist[port] = np.clip(np.minimum(ddist, 0), -10, 10)
                # else:
                #     self.ddist[port] = np.clip(ddist, -10, 10)
                self.ddist[port] = np.clip(dist_before-dist_now, -2, 2)

                self.action[port] = self.last_frame.players[port].action

            self.off_stage = self.last_frame.players[1].off_stage and self.last_frame.players[2].off_stage
        else:
            for port in self.ports:
                self.last_hit_frame[port] = frame.frame - self.FRAMES_BEFORE_SUICIDE
        self.last_frame = frame


class RewardFunction:

    def __init__(
            self,
            port,
            bot_config: "BotConfig", # rewards
    ):
        self.port = port
        self.bot_config = bot_config

        agressivity_p = self.bot_config.agressivity / 100
        agressivity_p = 0.45 + agressivity_p * 0.1

        self.damage_inflicted_scale = 0.01 * agressivity_p * self.bot_config._damage_reward_scale
        self.damage_received_scale = 0.01 * (1 - agressivity_p) * self.bot_config._damage_penalty_scale
        self.kill_reward_scale = 1.  #10. * agressivity_p
        self.death_reward_scale = 1. #10. * (1 - agressivity_p)
        self.away_cost_scale = 0.0002 / 3
        self.distance_reward_scale = 3e-4 #3e-5 is from old file #0. * 2.e-3 * self.bot_config._distance_reward_scale
        self.shieldstun_reward_scale = 0. *0.07 * self.bot_config._shieldstun_reward_scale
        self.neutralb_charge_reward_scale = 0. * 0.01 * self.bot_config._neutralb_charge_reward_scale
        self.energy_cost_scale = 0.0003

        # combos are important, as it encourages bot to not get hit, as well as learn setups.
        self.last_hit_action_state = Action.UNKNOWN_ANIMATION

        self.total_kills = 0
        self.previous_last_hit_frame = -120

        self.smooth_stocks = {p: 4 for p in {1, 2}}


        self.num_combos = 0
        self.total_combos = 0
        self.prev_combo_counters = {p: 0 for p in {1, 2}}
        self.combo_counters = {p: 0 for p in {1, 2}}

        self.self_combo_multiplier = 0
        self.opp_combo_multiplier = 0

        self.landed_hits = {
            a: 0 for a in Action
        }

        self.per_frame_metrics = defaultdict(float)
        self.metrics = defaultdict(float)

        self.helper = MeleeHelper(port, bot_config.character)

    def get_frame_rewards(self, delta_frame: DeltaFrame, current_action: ControllerInput, rewards: StepRewards,
                          combo_counters: dict):
        """
        Executed every frame
        """
        port = self.port
        other_port = 1 + (port % 2)

        # advantage should be reflected by stock then percent advantage
        # if at equal stocks, but different percents,
        #clutch_scale = np.clip(delta_frame.last_frame.players[port].percent - 80, 0, 150)

        lr = 0.0045
        # 3 sec interval for stock trading
        for p in self.smooth_stocks:
            self.smooth_stocks[p] = self.smooth_stocks[p] * (1-lr) + delta_frame.last_frame.players[p].stock * lr


        self_value = round(self.smooth_stocks[port]) - np.clip(delta_frame.last_frame.players[port].percent / 150, 0, 0.8)
        other_value = round(self.smooth_stocks[other_port]) - np.clip(delta_frame.last_frame.players[other_port].percent / 150, 0, 0.8)

        max_advantage = 3.8
        norm_advantage = (self_value - other_value) / max_advantage

        # reward kills/deaths
        kill = delta_frame.dstock[other_port]
        base_stock_reward = 1.

        if kill > 0:
            if delta_frame.last_frame.players[other_port].percent <= 5:
                rewards.kill_rewards += 0.2
            else:
                rewards.kill_rewards += base_stock_reward

        death = delta_frame.dstock[port]
        if death > 0:
            rewards.death_rewards += np.clip(base_stock_reward - norm_advantage, 0.3, 1.7)
            if delta_frame.last_frame.players[port].percent <= 5:
                self.metrics["zero_percent_suicides"] += 1.
            else:
                self.metrics["zero_percent_suicides"] += 0.

        self.metrics["win"] += delta_frame.win[port]


        has_dealt_damage = delta_frame.dpercent[other_port] > 0
        has_suffered_damage = delta_frame.dpercent[port] > 0

        is_close = delta_frame.dist[port] < 20
        shield_pressuring = (delta_frame.last_frame.players[other_port].action == Action.SHIELD_STUN
                             and delta_frame.action[other_port] != Action.SHIELD_STUN)
        shield_pressured = ((delta_frame.last_frame.players[port].action == Action.SHIELD_STUN)
                            and delta_frame.action[port] != Action.SHIELD_STUN)


        shield_interaction = (shield_pressuring or shield_pressured) and is_close
        damage_interaction = (has_suffered_damage or has_dealt_damage) and is_close

        if shield_interaction:
            if shield_pressured:
                rewards.shield_pressured += 1
            else:
                rewards.shield_pressuring += 1

        self.metrics["has_dealt_damage"] += int(has_dealt_damage)
        self.metrics["is_close"] += int(is_close)
        self.metrics["shield_pressuring"] += int(shield_pressuring)
        self.metrics["shield_pressured"] += int(shield_pressured)
        self.metrics["power_shield"] += int(delta_frame.last_frame.players[port].is_powershield)

        if shield_interaction or damage_interaction:
            self.metrics["interacting"] += 1.

        # closeup
        closeup = delta_frame.ddist[port]
        if (closeup > 0 and delta_frame.last_frame.players[port].action not in (Action.ROLL_FORWARD, Action.ROLL_BACKWARD)) or closeup < 0:

            p = delta_frame.last_frame.players[port]
            non_attack_speed = abs(p.speed_y_self) + abs(p.speed_air_x_self + p.speed_ground_x_self)
            non_attack_speed_ratio = (
                non_attack_speed
            ) / (
                abs(p.speed_y_attack) + abs(p.speed_x_attack) + non_attack_speed + 1e-8
            )
            closeup *= non_attack_speed_ratio

            if (delta_frame.last_frame.players[other_port].invulnerability_type == InvulnerabilityType.INVULNERABLE
                    and delta_frame.last_frame.players[port].invulnerability_type != InvulnerabilityType.INVULNERABLE):
                closeup = -closeup

            rewards.distance_rewards += closeup
        self.metrics["closeup"] += closeup


        # todo: how are we encouraged to make early kills ?
        # equal percent / stocks
        #   low percent:
        #       deal damage
        #   high percent:
        #       kill
        # equal stocks
        #   low percent:
        #

        # Damage / Combos
        inflicted_percents = delta_frame.dpercent[other_port]
        opponent_percents = delta_frame.last_frame.players[other_port].percent
        inflicted_percents_rewards = 0
        if inflicted_percents > 0:
            # max percent 200
            scale = 1.
            # add a customisable scale here
            #scale = scale / (1 + 3e-3 * np.exp(0.08*self.landed_hits[delta_frame.last_frame.players[port].action]))
            inflicted_percents_rewards = inflicted_percents * scale
            self.landed_hits[delta_frame.last_frame.players[port].action] += inflicted_percents
            rewards.damage_inflicted_rewards += inflicted_percents_rewards

        received_percents = delta_frame.dpercent[port]
        percents = delta_frame.last_frame.players[port].percent
        if received_percents > 0:
            # max percent 200
            scale = 1. - norm_advantage * 0.25
            if percents > 100:
                scale *= np.clip(1 - (percents - 100) / 75, 0, 1)
            if percents > 35 and delta_frame.last_frame.players[port].off_stage:
                scale *= np.clip(1 - (percents - 35) / 55, 0, 1)

            received_percents_rewards = received_percents * scale
            rewards.damage_received_rewards += received_percents_rewards

        # Edgeguarding
        # - on edge when other is offstage but same side
        # - inflicting offstage damage

        edge_catch = (delta_frame.last_frame.players[port].action == Action.EDGE_CATCHING
                      and delta_frame.action[port] != Action.EDGE_CATCHING)
        edge_guarding = (edge_catch and delta_frame.last_frame.players[other_port].off_stage
                         and (delta_frame.dist[port] < 55))
        edge_while_opp_invulnerable = (edge_catch and delta_frame.last_frame.players[other_port].invulnerability_type
                                       == InvulnerabilityType.INVULNERABLE and
                                       delta_frame.last_frame.players[port].invulnerability_type != InvulnerabilityType.INVULNERABLE)

        bad_edge = (edge_catch and not (edge_guarding or edge_while_opp_invulnerable))
        self.metrics["bad_edge"] += int(bad_edge)

        if bad_edge and self.metrics["bad_edge"] > 30:
            rewards.bad_edge_catches += 1
        if edge_guarding:
            rewards.edge_guarding += 1
        if edge_while_opp_invulnerable:
            rewards.edge_while_opp_invulnerable += 1

        if (delta_frame.last_frame.players[port].off_stage and delta_frame.last_frame.players[other_port].off_stage
            and inflicted_percents > 0):
            rewards.off_stage_percents_inflicted += inflicted_percents_rewards
            scale = 1. # 1 - np.clip(opponent_percents / 200, 0, 1)

        self.metrics["off_stage_damages"] += rewards.off_stage_percents_inflicted

        # helper rewards
        rewards.helper_bonus += self.helper(delta_frame.last_frame)

        self.per_frame_metrics["distance"] += delta_frame.dist[port]

        self.prev_combo_counters = self.combo_counters
        self.combo_counters = combo_counters
        if self.prev_combo_counters[port] - self.combo_counters[port] > 0:
            self.num_combos += 1
            self.total_combos += self.prev_combo_counters[port]

        self.self_combo_multiplier = (self.prev_combo_counters[port] / ObsBuilder.MAX_COMBO) + 1
        self.opp_combo_multiplier = (self.prev_combo_counters[other_port] / ObsBuilder.MAX_COMBO) + 1



    def compute(self, rewards: StepRewards):

        total = (
            rewards.kill_rewards * self.self_combo_multiplier
            + (rewards.damage_inflicted_rewards + rewards.off_stage_percents_inflicted) * 0.01 * self.self_combo_multiplier
            + rewards.distance_rewards * 5e-4
            + rewards.edge_guarding * 0.15
            + rewards.edge_while_opp_invulnerable * 0.05
            #+ rewards.shield_pressured * 0.02
            #
            - rewards.death_rewards * self.opp_combo_multiplier
            - (rewards.damage_received_rewards) * 0.01 * self.opp_combo_multiplier
            - rewards.bad_edge_catches * 0.05
            + rewards.helper_bonus
        )

        # print(
        #     self.port,
        #     self.combo_counters,
        #     self.smooth_stocks,
        #     total,
        #     rewards
        #       )
        # if self.port == 2:
        #     input()


        return total

    def get_metrics(self, game_length=1):
        metrics = {
            "per_frame_" + k: v / (game_length * 3)
            for k, v in self.per_frame_metrics.items()
        }
        metrics["avg_combo_length"] = self.total_combos / (self.num_combos + 1e-8)
        metrics.update(self.metrics)
        helper_metrics = self.helper.get_metrics()
        metrics.update(helper_metrics)
        return metrics
