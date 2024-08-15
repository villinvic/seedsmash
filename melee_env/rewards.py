import math
from dataclasses import dataclass, asdict
from typing import Union, Dict
from melee import GameState, Character, Action
import numpy as np
from melee_env.action_space import ControllerInput

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
    rewards collected for winning/loosing
    """
    win_rewards: np.float32 = 0.
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

    # TODO: add a possibility to add a scaling between damage and stocks

    """
    Off-stage percents inflicted when both players are off-stage
    """
    combo_length: np.float32 = 0.

    def to_dict(self):
        d = asdict(self)
        return d



class DeltaFrame:
    FRAMES_BEFORE_SUICIDE = 60 * 9

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

        self.action = {
            p: Action.KIRBY_BLADE_UP for p in self.ports
        }

        self.off_stage = False

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
            dx2 = (self.last_frame.players[1].position.x - self.last_frame.players[2].position.x)**2
            dy2 = (self.last_frame.players[1].position.y - self.last_frame.players[2].position.y)**2
            dist_before = np.sqrt(dx2 + 0.1 * dy2)

            for port in self.ports:
                other_port = 1 + (port % 2)

                # dx_now = np.abs(frame.players[port].position.x - self.last_frame.players[other_port].position.x)
                # dy_now = np.abs(frame.players[port].position.y - self.last_frame.players[other_port].position.y)
                dist_now = np.sqrt(
                    (frame.players[port].position.x - self.last_frame.players[other_port].position.x
                     )**2 + 0.1*(frame.players[port].position.y - self.last_frame.players[other_port].position.y) ** 2)

                self.dstock[port] = np.clip(self.last_frame.players[port].stock - frame.players[port].stock, 0., 1.)
                self.dpercent[port] = np.clip(frame.players[port].percent - self.last_frame.players[port].percent, 0., 50.)
                if self.dpercent[port] > 0 and np.sqrt(dx2 + dy2) < 70:
                    self.last_hit_frame[port] = frame.frame

                # DISTANCE


                # ddist = (dx_before - dx_now) + (dy_before - dy_now) * 0.05
                # if frame.players[port] in [Action.ROLL_FORWARD, Action.ROLL_BACKWARD]:
                #     self.ddist[port] = np.clip(np.minimum(ddist, 0), -10, 10)
                # else:
                #     self.ddist[port] = np.clip(ddist, -10, 10)
                self.ddist[port] = np.clip(dist_before-dist_now, -10, 10)

                self.action[port] = frame.players[port].action

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
        self.damage_inflicted_scale = 0.09 * agressivity_p
        self.damage_received_scale = 0.09 * (1 - agressivity_p)
        self.kill_reward_scale = 10. * agressivity_p
        self.death_reward_scale = 10. * (1 - agressivity_p)
        self.time_cost = 2 * 0.00010416 * self.bot_config.patience / 100
        self.distance_reward_scale = 5.e-3
        self.energy_cost_scale = 2.e-4

        self.combo_counter = 0.
        self.int_combo_counter = 0

        self.last_hit_action_state = Action.KIRBY_BLADE_UP
        self.max_combo_score = 8
        self.linear_discount = 1/60

        self.total_kills = 0

    def discount_combo(self):

        self.combo_counter = np.maximum(0, self.combo_counter - self.linear_discount)
        self.int_combo_counter = math.ceil(self.combo_counter)



    def get_frame_rewards(self, delta_frame: DeltaFrame, current_action: ControllerInput, rewards: StepRewards):
        """
        Executed every frame
        """
        # We should have always port 1 populated

        # For now, we assume the two possible ports are 1 and 2
        port = self.port
        other_port = 1 + (port % 2)

        lose_reward = delta_frame.win[other_port]
        rewards.win_rewards += delta_frame.win[port] - lose_reward
        kill = (delta_frame.dstock[other_port] *
                                 np.float32(delta_frame.last_frame.frame
                                            - delta_frame.last_hit_frame[other_port]
                                            <= delta_frame.FRAMES_BEFORE_SUICIDE))
        rewards.kill_rewards += kill
        self.total_kills += kill

        rewards.death_rewards += delta_frame.dstock[port]

        rewards.damage_inflicted_rewards += delta_frame.dpercent[other_port]
        rewards.damage_received_rewards += delta_frame.dpercent[port]

        if delta_frame.off_stage:
            rewards.off_stage_percents_inflicted += delta_frame.dpercent[other_port]

        rewards.distance_rewards += delta_frame.ddist[port]
        rewards.energy_costs += current_action.energy_cost

        if delta_frame.dpercent[other_port] > 0:
            bonus = 2
            if self.last_hit_action_state == delta_frame.action[port]:
                bonus = 0.1

            self.combo_counter = np.minimum(self.combo_counter + bonus, self.max_combo_score)
            self.last_hit_action_state = delta_frame.action[port]

        if delta_frame.dpercent[port] > 0:
            self.combo_counter = 0.
            self.int_combo_counter = 0

        rewards.combo_length = self.int_combo_counter
        self.discount_combo()


    def compute(self, rewards: StepRewards, opponent_combo_count):

        # TODO: use logspace for bot config !

        combo_gaming_bonus = (1 + 5e-3 * self.int_combo_counter * self.bot_config.combo_game)
        combo_breaking_bonus = (1 + 5e-3 * opponent_combo_count * self.bot_config.combo_breaker)

        # input("ENTER")
        # print(
        #     self.port,
        #     self.int_combo_counter, opponent_combo_count,
        #     rewards.kill_rewards * self.kill_reward_scale,
        #         rewards.death_rewards * self.death_reward_scale,
        #         rewards.win_rewards * self.bot_config.winning_desire, self.total_kills,
        #         (rewards.damage_inflicted_rewards +
        #            rewards.off_stage_percents_inflicted*self.bot_config.off_stage_plays/100)
        #         * self.damage_inflicted_scale * combo_gaming_bonus,
        #       combo_gaming_bonus,
        #         rewards.damage_received_rewards * self.damage_received_scale * combo_breaking_bonus,
        #       combo_breaking_bonus,
        #         rewards.distance_rewards * self.distance_reward_scale)

        return (
                rewards.kill_rewards * self.kill_reward_scale
                - rewards.death_rewards * self.death_reward_scale
                + rewards.win_rewards * self.bot_config.winning_desire * np.float32(self.total_kills>=3)
                + (rewards.damage_inflicted_rewards +
                   rewards.off_stage_percents_inflicted*self.bot_config.off_stage_plays/100)
                * self.damage_inflicted_scale * combo_gaming_bonus
                - rewards.damage_received_rewards * self.damage_received_scale * combo_breaking_bonus
                + rewards.distance_rewards * self.distance_reward_scale
              #  - rewards.energy_costs * self.energy_cost_scale
              #  - self.time_cost
        )
