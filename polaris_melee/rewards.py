import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Union, Dict

from Cython import other_types
from melee import GameState, Character, Action, InvulnerabilityType, PlayerState, Stage, stages, Position, EDGE_POSITION
import numpy as np

from polaris_melee.action_space import ControllerInput
from polaris_melee.melee_helpers import MeleeHelper
from polaris_melee.observation_space import ObsBuilder
from seedsmash.bot import Bot
from seedsmash.bots.bot_config import BotConfig


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
    offstage: np.float32 = 0.
    intangibility: np.float32 = 0.

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

    OFFSTAGE = "offstage"
    """
    1 when going back on stage
    -1 when going offstage
    0 when no changes
    """

    def __init__(self):
        self.prev_frame: Union[GameState, None] = None
        self.last_frame: Union[GameState, None] = None
        self.episode_finished = False
        self.vertical_distance_scale = 0.2
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
            self.prev_frame = frame
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
            self[DeltaFrame.OFFSTAGE][port] = int(players[port].off_stage) - int(players_before[port].off_stage)

        self.prev_frame = self.last_frame
        self.last_frame = frame



class RewardFunction:

    ROLL_STATES = (Action.ROLL_FORWARD, Action.ROLL_BACKWARD, Action.GROUND_ROLL_FORWARD_UP, Action.GROUND_ROLL_BACKWARD_UP,
            Action.GROUND_ROLL_FORWARD_DOWN, Action.GROUND_ROLL_BACKWARD_DOWN, Action.GROUND_ROLL_SPOT_DOWN,
            Action.FORWARD_TECH, Action.BACKWARD_TECH, Action.WALK_FAST, Action.WALK_MIDDLE, Action.WALK_SLOW,
            Action.ON_HALO_DESCENT, Action.EDGE_ROLL_SLOW, Action.EDGE_GETUP_SLOW, Action.EDGE_ROLL_QUICK,
                   Action.EDGE_GETUP_QUICK, Action.EDGE_JUMP_1_QUICK, Action.EDGE_JUMP_2_QUICK,
                   Action.EDGE_JUMP_1_SLOW, Action.EDGE_JUMP_2_SLOW
                   )
    GETUP_ATTACKS = (
        Action.GETUP_ATTACK, Action.EDGE_ATTACK_QUICK, Action.EDGE_ATTACK_SLOW, Action.GROUND_ATTACK_UP
    )

    INTANGIBLE_STATES = ROLL_STATES + GETUP_ATTACKS + (Action.GROUND_GETUP, Action.GROUND_SPOT_UP)
    # No EDGE_HANGING/CATCHING as we want the agent to learn to edge hop, or edge stall, edge approach, etc.

    def __init__(
            self,
            port,
            bot: Bot,
            opponent_bot: Bot,
    ):
        self.port = port

        self.smooth_stocks = {p: 4 for p in {1, 2}}
        self.num_combos = 0
        self.total_combos = 0
        self.prev_combo_counters = {p: 0 for p in {1, 2}}

        self.percent_cap = 100

        self.per_frame_metrics = defaultdict(float)
        self.metrics = defaultdict(float)

        # TODO: instanciate only one of these per reward function and compute the delta somehow without having to
        #       deal with the bot preferences
        self.helper = MeleeHelper(port, bot.character)
        self.opponent_helper = MeleeHelper(port, opponent_bot.character)

        self.was_offstage = False
        self.opp_was_offstage = False

        self.preferences = bot.stats

        self.tot = StepRewards()

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
            if self_stock == 1:
                return 0
            return (self.cap_percent(player.percent)-self.cap_percent(opponent.percent)) / (self.percent_cap*5)
        else:
            return (self_stock - opp_stock) / (opp_stock + self_stock)

    def compute_stage_control_rewards(self, player: PlayerState, opponent: PlayerState, stage: Stage) -> float:
        center_x = EDGE_POSITION[stage] / 2

        at_center = abs(player.x) < center_x
        opp_at_center = abs(opponent.x) < center_x
        return int(at_center) - int(opp_at_center)

    def compute_damage_rewards(
            self,
            player: PlayerState,
            opponent: PlayerState,
            dealt_damage: int,
            off_stage: bool,
            combo_count: int,
            advantage: float,
    ) -> float:
        combo_boost = 1 + combo_count / ObsBuilder.MAX_COMBO
        dealt_damage = dealt_damage * combo_boost
        # if opponent.percent > self.percent_cap:
        #     return 0.
        # return int(off_stage) * dealt_damage
        if player.invulnerability_type == InvulnerabilityType.INTANGIBLE and player.action not in self.GETUP_ATTACKS:
            dealt_damage *= 1.5
        if off_stage:
            dealt_damage *= (1 + advantage)
        return dealt_damage

    def dist_to_ledge(
            self,
            x: float,
            y: float,
            edge_position: float
    ):
        abs_x = abs(x)
        return abs(abs_x - edge_position) if y >= 0 else np.sqrt((abs_x - edge_position) ** 2 + y ** 2)

    def has_pos_advantage(
            self,
            player: PlayerState,
            opponent: PlayerState,
            stage: Stage
    ) -> bool:
        stage_edge = stages.EDGE_POSITION[stage]
        abs_x, abs_opp_x = abs(player.position.x), abs(opponent.position.x)
        y, opp_y = player.position.y, opponent.position.y

        # Onstage comparison
        if y >= 0 and opp_y >= 0 and abs_x < stage_edge and abs_opp_x < stage_edge:
            dx = abs_x - abs_opp_x
            # if -10 < dx < 10:  # Similar x positions: prefer lower y
            #     return y < opp_y
            return dx < 0  # Closer to the center stage is better

        # One player offstage
        if abs_x < stage_edge and y >= 0:
            return True  # Player is onstage, opponent is offstage
        if abs_opp_x < stage_edge and opp_y >= 0:
            return False  # Opponent is onstage, player is offstage

        # Both offstage: prioritize being above and closer to ledge, and intangibility
        if player.invulnerability_type == InvulnerabilityType.INTANGIBLE:
            if opponent.invulnerability_type != InvulnerabilityType.INTANGIBLE:
                return True
        elif opponent.invulnerability_type == InvulnerabilityType.INTANGIBLE:
            return False

        player_dist = self.dist_to_ledge(abs_x, y, stage_edge)
        opp_dist = self.dist_to_ledge(abs_opp_x, opp_y, stage_edge)
        return player_dist < opp_dist

    def get_field_reward(
            self,
            player: PlayerState,
            prev_player: PlayerState,
            stage: Stage
    ):
        """
        Like a magnetic field, we want to be attracted to the center
        And push the opponent offstage.
        -> if we are further than opponent, only reward for attraction
        -> if closer than opponent, only reward for repulsion
        : there cannot be coalition, as both players will cross path eventually
        """
        # if advantageous_position:
        #     # go toward opponnent
        #     return closeup
        stage_edge = stages.EDGE_POSITION[stage]
        abs_x = abs(player.position.x)
        scale = abs_x / stage_edge
        # on stage:
        if player.position.y >= 0 and abs_x < stage_edge:
            # go to center
            # normalise to 1, linear function
            prev_dist_from_center = abs(prev_player.position.x)
            field_reward = prev_dist_from_center - abs_x
        else:
            prev_dist_to_ledge = self.dist_to_ledge(abs(prev_player.position.x), prev_player.position.y, stage_edge)
            dist_to_ledge = self.dist_to_ledge(abs_x, player.position.y, stage_edge)
            scale = scale + dist_to_ledge / stage_edge
            # go toward edge
            field_reward = prev_dist_to_ledge - dist_to_ledge

        return field_reward * scale

    def offstage_rewards(
            self,
            player: PlayerState,
            opponent: PlayerState,
            delta_frame: DeltaFrame
    ) -> float:
        """
        Incentives the bot to go back on stage/edge
        as well as prevent the opponent from getting the ledge and going back on stage.
        """
        r = 0.
        port = self.port
        other_port = 1 + (port % 2)
        speed_induced_attack = abs(player.speed_y_attack) + abs(player.speed_x_attack)
        edge_pos = EDGE_POSITION[delta_frame.last_frame.stage]
        if (player.off_stage and speed_induced_attack > 0 and
                self.dist_to_ledge(player.position.x, player.position.y, edge_pos) > 30):
            self.was_offstage = True
        if not player.off_stage or player.action == Action.EDGE_CATCHING:
            if self.was_offstage:
                r += 1
            self.was_offstage = False
        elif delta_frame[DeltaFrame.DEATH][self.port] > 0:
            self.was_offstage = False
        opp_speed_induced_attack = abs(opponent.speed_y_attack) + abs(opponent.speed_x_attack)
        if (opponent.off_stage and opp_speed_induced_attack > 0 and
                self.dist_to_ledge(opponent.position.x, opponent.position.y, edge_pos) > 30):
            self.opp_was_offstage = True
        if not opponent.off_stage or player.action == Action.EDGE_CATCHING:
            if self.opp_was_offstage:
                r -= 1
            self.opp_was_offstage = False
        elif delta_frame[DeltaFrame.DEATH][other_port] > 0:
            self.opp_was_offstage = False

        return r

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
        player_advantage = np.clip(player_advantage, -0.33, 0.33)


        #opponent_suicide = delta_frame.percent_until_death[other_port] < 4 and delta_frame[DeltaFrame.DEATH][other_port] > 0
        rewards.kill += delta_frame[DeltaFrame.DEATH][other_port] * (1 + np.maximum(player_advantage, 0))

        rewards.death += delta_frame[DeltaFrame.DEATH][port] * (1 - np.minimum(player_advantage, 0))

        off_stage = player.off_stage and opponent.off_stage
        dealt_damage_rewards = self.compute_damage_rewards(
            player,
            opponent,
            delta_frame[DeltaFrame.DAMAGE][other_port],
            off_stage,
            self.prev_combo_counters[port],
            player_advantage,
        )
        if off_stage and not self.was_offstage and self.opp_was_offstage:
                # this is not zero sum, but should be fine.
                dealt_damage_rewards *= 1.5

        rewards.dealt_damage += dealt_damage_rewards


        good_intangibility = (player.invulnerability_type == InvulnerabilityType.INTANGIBLE
                              and player.action not in self.INTANGIBLE_STATES
                              and delta_frame[DeltaFrame.DISTANCE][port] < 60)
        opp_good_intangibility = (opponent.invulnerability_type == InvulnerabilityType.INTANGIBLE
                              and opponent.action not in self.INTANGIBLE_STATES
                              and delta_frame[DeltaFrame.DISTANCE][other_port] < 60)
        intangibility_rewards = int(good_intangibility) - int(opp_good_intangibility)
        rewards.intangibility += intangibility_rewards


        closeup = delta_frame[DeltaFrame.CLOSEUP][port]
        if (closeup > 0 and player.action not in self.ROLL_STATES) or closeup < 0:
            # non_attack_speed = abs(player.speed_y_self) + abs(player.speed_air_x_self + player.speed_ground_x_self)
            # non_attack_speed_ratio = non_attack_speed / (
            #         abs(player.speed_y_attack) + abs(player.speed_x_attack) + non_attack_speed + 1e-8
            # )
            # closeup *= non_attack_speed_ratio

            if (opponent.invulnerability_type == InvulnerabilityType.INVULNERABLE
                    and player.invulnerability_type != InvulnerabilityType.INVULNERABLE):
                if player.action not in self.ROLL_STATES:
                    closeup = 0.
                closeup = -closeup

        else:
            closeup = 0
        rewards.closeup += closeup * (1 + player_advantage)

        received_damage_rewards = self.compute_damage_rewards(
            opponent,
            player,
            delta_frame[DeltaFrame.DAMAGE][port],
            off_stage,
            self.prev_combo_counters[other_port],
            player_advantage
        )
        rewards.received_damage += received_damage_rewards

        # Non damage edgeguarding
        edge_catch = (player.action == Action.EDGE_CATCHING and player.action_frame == 2)
        edge_guarding = (edge_catch and opponent.off_stage and delta_frame[DeltaFrame.DX][port] < 90)
        edge_while_opp_invulnerable = (edge_catch and opponent.invulnerability_type == InvulnerabilityType.INVULNERABLE
                                       and player.invulnerability_type != InvulnerabilityType.INVULNERABLE)

        stage_control_rewards = self.compute_stage_control_rewards(player, opponent, delta_frame.last_frame.stage)
        rewards.stage_control += stage_control_rewards

        bad_edge = (edge_catch and not (edge_guarding or edge_while_opp_invulnerable))
        if bad_edge and self.metrics["bad_edge"] > 20:
            rewards.bad_edge_catches += 1
        if edge_guarding:
            rewards.edge_guarding += 1
        if edge_while_opp_invulnerable:
            rewards.edge_while_opp_invulnerable += 1


        offstage_rewards = self.offstage_rewards(player, opponent, delta_frame)
        rewards.offstage += offstage_rewards

        # Techskill and char specific stuff
        # TODO: we compute twice this...
        rewards.helper_bonus += self.helper(delta_frame.last_frame) - self.opponent_helper(delta_frame.last_frame)

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
        self.metrics["offstage"] += offstage_rewards
        self.per_frame_metrics["distance"] += delta_frame[DeltaFrame.DISTANCE][port]
        self.metrics["bad_edge"] += int(bad_edge)
        self.metrics["edge_guard"] += int(edge_guarding)
        self.metrics["edge_while_opp_invulnerable"] += int(edge_while_opp_invulnerable)
        self.metrics["good_intangibility"] += intangibility_rewards
        self.metrics["stage_control"] += stage_control_rewards
        if off_stage:
            self.metrics["off_stage_damages"] += delta_frame[DeltaFrame.DAMAGE][other_port]


        if self.prev_combo_counters[port] - combo_counters[port] > 0:
            self.num_combos += 1
            self.total_combos += self.prev_combo_counters[port]

        self.prev_combo_counters = combo_counters

    def compute(self, rewards: StepRewards):

        total = (
            rewards.kill
            + 0.005 * rewards.dealt_damage
            + rewards.helper_bonus

            + 1e-3 * rewards.closeup
            + 0.025 * 0.03 * rewards.intangibility
            + 0.1 * rewards.offstage
            + 0.003 * 0.05 * rewards.stage_control

            - rewards.death
            - 0.005 * rewards.received_damage
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
