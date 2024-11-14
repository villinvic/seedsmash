import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Union, Dict

from Cython import other_types
from melee import GameState, Character, Action, InvulnerabilityType, PlayerState, Stage, stages
import numpy as np

from polaris_melee.action_space import ControllerInput
from polaris_melee.melee_helpers import MeleeHelper
from polaris_melee.observation_space import ObsBuilder
from seedsmash2.submissions.bot_config import BotConfig


@dataclass
class StepRewards(
):
    """
    Databag for player rewards
    """

    """
    rewards collected for damaging/taking damage
    """
    dealt_damage: np.float32 = 0.
    received_damage: np.float32 = 0.

    """
    rewards collected for moving toward the opponent
    TODO: add in bot config somehow
    """
    closeup: np.float32 = 0.

    """
    rewards collected for killing/dying
    """
    kill: np.float32 = 0.
    death: np.float32 = 0.

    """
    Cost incurred for sweating on the c-stick or the buttons
    """
    energy_costs: np.float32 = 0.

    bad_edge_catches: np.float32 = 0.
    edge_while_opp_invulnerable: np.float32 = 0.
    edge_guarding: np.float32 = 0.

    helper_bonus: np.float32 = 0.

    stage_control: np.float32 = 0.

    # could help
    non_hitting_attacks: np.float32 = 0.


    # TODO: add a possibility to add a scaling between damage and stocks

    def to_dict(self):
        d = asdict(self)
        return d



class DeltaFrame:
    FRAMES_BEFORE_SUICIDE = 60 * 7

    DEATH = "death"
    WIN = "win"
    DISTANCE = "distance"
    DX = "dx"
    CLOSEUP = "closeup"
    DAMAGE = "damage"

    def __init__(self):
        self.last_frame: Union[GameState, None] = None
        self.episode_finished = False
        self.vertical_distance_scale = 0.75
        self.ports = {1, 2}
        self.zero()

    def zero(self):
        self.delta = defaultdict(lambda:{
            p: 0 for p in self.ports
        })
        # keeps track of percents before stock removal (to compute suicides)
        self.percent_until_death = {
            p: 0 for p in self.ports
        }

    def __getitem__(self, item):
        return self.delta[item]

    def __setitem__(self, key, value):
        self.delta[key] = value

    def get_player_dist(self, p1: PlayerState, p2: PlayerState):
        dx2 = (p1.position.x-p2.position.x) ** 2
        dy2 = (p1.position.y-p2.position.y) ** 2
        return np.sqrt(dx2 + self.vertical_distance_scale*dy2)

    def update(self, frame: GameState):
        players = frame.players

        if self.last_frame is None:
            self.last_frame = frame
            return
        players_before = self.last_frame.players
        if self.episode_finished or len(frame.players) < 2:
            self.zero()
            return

        dead1 = players[1].stock == 0
        dead2 = players[2].stock == 0
        if dead1 and dead2:
            # weird ?
            self.episode_finished = True
            self.zero()
            return
        elif dead1 or dead2:
            self.episode_finished = True
            self[DeltaFrame.WIN][1] = int(dead2) - int(dead1)
            self[DeltaFrame.WIN][2] = - self[DeltaFrame.WIN][1]

        dist_before = self.get_player_dist(*players_before.values())
        for port in self.ports:
            other_port = 1 + (port % 2)
            dist_now = self.get_player_dist(players_before[other_port], players[port])
            self[DeltaFrame.DISTANCE][port] = dist_now
            self[DeltaFrame.DX][port] = abs(players[port].position.x-players[other_port].position.x)
            self[DeltaFrame.CLOSEUP][port] = np.clip(dist_before-dist_now, -3, 3)

            # We update the percent until death purposely before updating the stock delta, so that we get the percents
            # we had before dying
            if self[DeltaFrame.DEATH][port] > 0:
                self.percent_until_death[port] = 0
            else:
                self.percent_until_death[port] = np.maximum(players[port].percent, self.percent_until_death[port])

            self[DeltaFrame.DEATH][port] = np.clip(players_before[port].stock - players[port].stock, 0., 1.)
            self[DeltaFrame.DAMAGE][port] = np.clip(players[port].percent - players_before[port].percent, 0., 50.)
        self.last_frame = frame


class RewardFunction:

    ROLL_STATES = (Action.ROLL_FORWARD, Action.ROLL_BACKWARD, Action.GROUND_ROLL_FORWARD_UP, Action.GROUND_ROLL_BACKWARD_UP,
            Action.GROUND_ROLL_FORWARD_DOWN, Action.GROUND_ROLL_BACKWARD_DOWN, Action.GROUND_ROLL_SPOT_DOWN,
            Action.FORWARD_TECH, Action.BACKWARD_TECH, Action.WALK_FAST, Action.WALK_MIDDLE, Action.WALK_SLOW)
    GETUP_ATTACKS = (
        Action.GETUP_ATTACK, Action.EDGE_ATTACK_QUICK, Action.EDGE_ATTACK_SLOW, Action.GROUND_ATTACK_UP
    )

    def __init__(
            self,
            port,
            bot_config: "BotConfig", # rewards
    ):
        self.port = port
        self.bot_config = bot_config

        agressivity_p = self.bot_config.agressivity / 100
        agressivity_p = 0.45 + agressivity_p * 0.1

        # OLD STUFF
        self.damage_inflicted_scale = 0.01 * agressivity_p * self.bot_config._damage_reward_scale
        self.damage_received_scale = 0.01 * (1 - agressivity_p) * self.bot_config._damage_penalty_scale
        self.kill_reward_scale = 1.  #10. * agressivity_p
        self.death_reward_scale = 1. #10. * (1 - agressivity_p)
        self.away_cost_scale = 0.0002 / 3
        self.distance_reward_scale = 3e-4 #3e-5 is from old file #0. * 2.e-3 * self.bot_config._distance_reward_scale
        self.shieldstun_reward_scale = 0. *0.07 * self.bot_config._shieldstun_reward_scale
        self.neutralb_charge_reward_scale = 0. * 0.01 * self.bot_config._neutralb_charge_reward_scale
        self.energy_cost_scale = 0.0003


        self.smooth_stocks = {p: 4 for p in {1, 2}}
        self.num_combos = 0
        self.total_combos = 0
        self.prev_combo_counters = {p: 0 for p in {1, 2}}

        self.percent_cap = 100

        self.per_frame_metrics = defaultdict(float)
        self.metrics = defaultdict(float)
        self.helper = MeleeHelper(port, bot_config.character)

    def cap_percent(self, percent: int):
        return np.minimum(percent, self.percent_cap)

    def compute_player_advantage(self, player: PlayerState, opponent: PlayerState) -> float:
        # 3 sec interval for stock trading
        lr = 0.0045

        port = self.port
        other_port = 1 + (port % 2)
        self.smooth_stocks[port] = self.smooth_stocks[port] * (1 - lr) + player.stock * lr
        self.smooth_stocks[other_port] = self.smooth_stocks[other_port] * (1 - lr) + opponent.stock * lr

        self_stock = round(self.smooth_stocks[self.port])
        opp_stock = round(self.smooth_stocks[other_port])
        if self_stock == opp_stock:
            return (self.cap_percent(player.percent)-self.cap_percent(opponent.percent)) / 100
        else:
            return (self_stock - opp_stock) / (opp_stock + self_stock)

    def compute_stage_control_rewards(self, opponent: PlayerState, stage: Stage) -> float:
        x = opponent.position.x
        y = opponent.position.y

        x_next = x + opponent.speed_x_attack + opponent.speed_ground_x_self + opponent.speed_air_x_self
        y_next = y + opponent.speed_y_attack + opponent.speed_y_self

        x_left, x_right, y_up, y_down = stages.BLASTZONES[stage]

        dx_left = np.square(x_left - x)
        dx_right = np.square(x_right - x)

        dy_up = np.square(y_up - y)
        dy_down = np.square(y_down - y)

        if dx_left < dx_right:
            dx = dx_left
            next_dx = np.square(x_left-x_next)
        else:
            dx = dx_right
            next_dx = np.square(x_right-x_next)

        if dy_up < dy_down:
            dy = dy_up
            next_dy = np.square(y_up - y_next)
        else:
            dy = dy_down
            next_dy = np.square(y_down - y_next)

        scale_up = self.cap_percent(opponent.percent)/self.percent_cap
        ddist = (np.sqrt(dx + dy) - np.sqrt(next_dx + next_dy)) * scale_up

        return ddist

    def compute_damage_rewards(
            self,
            player: PlayerState,
            opponent: PlayerState,
            dealt_damage: int,
            off_stage: bool,
            combo_count: int
    ) -> float:
        combo_boost = 1 + combo_count / (2 * ObsBuilder.MAX_COMBO)
        dealt_damage = dealt_damage * combo_boost
        if opponent.percent > self.percent_cap:
            return int(off_stage) * dealt_damage
        if player.invulnerability_type == InvulnerabilityType.INTANGIBLE and player.action not in self.GETUP_ATTACKS:
            dealt_damage *= 1.1
        if off_stage:
            dealt_damage *= 1.5
        return dealt_damage

    def add_frame_rewards(self, delta_frame: DeltaFrame, current_action: ControllerInput, rewards: StepRewards,
                          combo_counters: dict):
        """
        Executed every frame, those rewards are obtained 3x per action
        """
        port = self.port
        other_port = 1 + (port % 2)
        player = delta_frame.last_frame.players[port]
        opponent = delta_frame.last_frame.players[other_port]

        player_advantage = self.compute_player_advantage(player, opponent)

        # TODO: fix this
        # if delta_frame[DeltaFrame.DEATH][other_port] > 0:
        #     print("death, are percents reset too ?", opponent.percent, delta_frame.percent_until_death[other_port])

        #opponent_suicide = delta_frame.percent_until_death[other_port] < 4 and delta_frame[DeltaFrame.DEATH][other_port] > 0
        rewards.kill += delta_frame[DeltaFrame.DEATH][other_port] #* (1 - 0.8 * float(opponent_suicide))

        rewards.death += delta_frame[DeltaFrame.DEATH][port] * (1 - player_advantage * 0.5)

        closeup = delta_frame[DeltaFrame.CLOSEUP][port]
        if (closeup > 0 and player.action not in self.ROLL_STATES) or closeup < 0:
            non_attack_speed = abs(player.speed_y_self) + abs(player.speed_air_x_self + player.speed_ground_x_self)
            non_attack_speed_ratio = non_attack_speed / (
                    abs(player.speed_y_attack) + abs(player.speed_x_attack) + non_attack_speed + 1e-8
            )

            closeup *= non_attack_speed_ratio
            if (opponent.invulnerability_type == InvulnerabilityType.INVULNERABLE
                    and player.invulnerability_type != InvulnerabilityType.INVULNERABLE):
                closeup = -closeup
        rewards.closeup += closeup

        off_stage = player.off_stage and opponent.off_stage
        rewards.dealt_damage += self.compute_damage_rewards(
            player,
            opponent,
            delta_frame[DeltaFrame.DAMAGE][other_port],
            off_stage,
            self.prev_combo_counters[port]
        )

        stage_control = self.compute_stage_control_rewards(
            opponent,
            delta_frame.last_frame.stage
        )
        rewards.stage_control += stage_control

        if off_stage:
            self.metrics["off_stage_damages"] += delta_frame[DeltaFrame.DAMAGE][other_port]

        # If we are offstage, forget about damage after 50%
        if player.percent > 50 and player.off_stage:
            rewards.received_damage = 0.
        else:
            rewards.received_damage += self.compute_damage_rewards(
                opponent,
                player,
                delta_frame[DeltaFrame.DAMAGE][port],
                off_stage,
                self.prev_combo_counters[other_port]
            )
        # Non damage edgeguarding
        edge_catch = (player.action == Action.EDGE_CATCHING and player.action_frame == 2)
        edge_guarding = (edge_catch and opponent.off_stage and delta_frame[DeltaFrame.DX][port] < 90)
        edge_while_opp_invulnerable = (edge_catch and opponent.invulnerability_type == InvulnerabilityType.INVULNERABLE
                                       and player.invulnerability_type != InvulnerabilityType.INVULNERABLE)

        bad_edge = (edge_catch and not (edge_guarding or edge_while_opp_invulnerable))
        if bad_edge and self.metrics["bad_edge"] > 30:
            rewards.bad_edge_catches += 1
        if edge_guarding:
            rewards.edge_guarding += 1
        if edge_while_opp_invulnerable:
            rewards.edge_while_opp_invulnerable += 1

        # Techskill and char specific stuff
        rewards.helper_bonus += self.helper(delta_frame.last_frame)

        # Metrics ----------------------------------
        # self.metrics["zero_percent_suicides"] += int(
        #     player.percent < 5 and delta_frame[DeltaFrame.DEATH][port] > 0
        # )

        self.metrics[DeltaFrame.WIN] += delta_frame[DeltaFrame.WIN][port]

        is_close = delta_frame[DeltaFrame.DISTANCE][port] < 50
        shield_pressuring = opponent.action == Action.SHIELD_STUN
        shield_pressured = player.action == Action.SHIELD_STUN
        self.metrics["shield_pressure"] += int(shield_pressuring)
        self.metrics["interacting"] += int(
            (shield_pressured or shield_pressuring
             or delta_frame[DeltaFrame.DAMAGE][port] - delta_frame[DeltaFrame.DAMAGE][other_port] != 0
             ) and is_close
        )
        self.metrics["closeup"] += closeup
        self.metrics["stage_control"] += stage_control
        self.per_frame_metrics["distance"] += delta_frame[DeltaFrame.DISTANCE][port]
        self.metrics["bad_edge"] += int(bad_edge)
        self.metrics["edge_guard"] += int(edge_guarding)
        self.metrics["edge_while_opp_invulnerable"] += int(edge_while_opp_invulnerable)

        if self.prev_combo_counters[port] - combo_counters[port] > 0:
            self.num_combos += 1
            self.total_combos += self.prev_combo_counters[port]

        self.prev_combo_counters = combo_counters

    def compute(self, rewards: StepRewards):

        total = (
            rewards.kill
            + 0.009 * rewards.dealt_damage
            + 1e-4 * rewards.closeup
            + 0.1 * rewards.edge_guarding
            + rewards.helper_bonus
            + 3e-3 * rewards.stage_control

            - rewards.death
            - 0.009 * rewards.received_damage
            - rewards.bad_edge_catches * 0.05
        )

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


if __name__ == '__main__':

    bc = BotConfig()

    rf = RewardFunction(
        1, bc
    )

    ps = PlayerState()
    ps.position.x = 180
    ps.speed_x_attack = 12
    ps.speed_y_attack = 0
    ps.percent = 100

    print(rf.compute_stage_control_rewards(
        ps, Stage.FINAL_DESTINATION
    ))