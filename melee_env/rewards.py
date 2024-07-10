from dataclasses import dataclass, asdict
from typing import Union, Dict
from melee import GameState, Character, Action
import numpy as np
from melee_env.action_space import ControllerInput

@dataclass
class SSBMRewards:
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
    win_reward_scale: np.float32 = 0.  # 5.
    """
    rewards collected for damaging/taking damage
    """
    damage_rewards: np.float32 = 0.
    damage_reward_scale: np.float32 = 0.005
    off_stage_multiplier: np.float32 = 1.

    combo_multiplier: np.float32 = 0.
    """
    rewards collected for moving toward the opponent
    """
    distance_rewards: np.float32 = 0.
    distance_reward_scale: np.float32 = 0.  # 0.00003

    """
    rewards collected for killing/dying
    """
    kill_rewards: np.float32 = 0.
    kill_reward_scale: np.float32 = 1.
    death_rewards: np.float32 = 0.
    death_reward_scale: np.float32 = 1.

    """
    Cost incurred for sweating on the c-stick or the buttons
    We should not punish techskill though
    """
    energy_costs: np.float32 = 0.
    energy_cost_scale: np.float32 = 0.0003

    #shield_reflect_rewards: np.float32 = 0.
    #shield_reflect_scale: np.float32 = 0.2

    time_cost: np.float32 = -0.0002


    # Put that in the rewardfunction class
    # """
    # gives a bonus if hitting an opponent in hitstun
    # also punishes more if getting hit in hitstun
    # """
    # combo_bonus: bool = True
    def to_dict(self):
        d = asdict(self)
        return d

    def total(self):
        return (
            self.kill_rewards * self.kill_reward_scale
            + self.death_rewards * self.death_reward_scale
            + self.win_rewards * self.win_reward_scale
            + self.damage_rewards * self.damage_reward_scale
            + self.distance_rewards * self.distance_reward_scale
            + self.energy_costs * self.energy_cost_scale
            + self.time_cost
                )


class RewardFunction:

    def __init__(self):
        self.last_state: Union[GameState, None] = None
        self.episode_finished = False
        self.combo_counter = {i: 0 for i in range(1, 5)}
        self.last_hit_action_state = {i: Action.KIRBY_BLADE_UP for i in range(1, 5)}
        self.max_combo = 6
        self.combo_gamma = 0.997
        self.linear_discount = 1/120

    def combo_discounter(self, count):
        return np.clip(count * self.combo_gamma - self.linear_discount, 0, np.inf)


    def compute(self, new_state: GameState, current_actions: Dict[int, ControllerInput], rewards: Dict[int, SSBMRewards]):
        # We should have always port 1 populated
        bot_port2 = 2 in rewards
        if not self.episode_finished:
            if len(new_state.players) < 2:
                return

            if self.last_state is not None:
                if new_state.players[1].stock == 0 and new_state.players[2].stock == 0:
                    self.episode_finished = True
                    return
                    # TODO: we go there sometimes
                    #raise NotImplementedError
                else:
                    # WINNING
                    # first player wins if the second player has no more stock
                    dead1 = new_state.players[1].stock == 0
                    dead2 = new_state.players[2].stock == 0
                    self.episode_finished = dead1 or dead2
                    if self.episode_finished:

                        win_score1 = np.float32(new_state.players[2].stock == 0)
                        win_score2 = np.float32(new_state.players[1].stock == 0)
                        rewards[1].win_rewards = win_score1 - win_score2
                        if bot_port2:
                            rewards[2].win_rewards = win_score2 - win_score1

                    if not self.episode_finished:
                        # STOCKS
                        dstock1 = np.clip(self.last_state.players[1].stock - new_state.players[1].stock, 0., 1.)
                        dstock2 = np.clip(self.last_state.players[2].stock - new_state.players[2].stock, 0., 1.)

                        rewards[1].death_rewards -= dstock1
                        rewards[1].kill_rewards += dstock2
                        if bot_port2:
                            rewards[2].death_rewards -= dstock2
                            rewards[2].kill_rewards += dstock1

                        # PERCENTS
                        dpercent1 = np.clip(new_state.players[1].percent - self.last_state.players[1].percent, 0., 50.)
                        dpercent2 = np.clip(new_state.players[2].percent - self.last_state.players[2].percent, 0., 50.)

                        # give a bonus if still in hitsun (if this is somewhat a combo)
                        # A two hit combo is irrelevant (some chars have two hit attacks, they will learn to spam it
                        # otherwise...)
                        c1 = self.combo_counter[1]
                        c2 = self.combo_counter[2]

                        combo_bonus1 = (1. + np.square(c1/3) * rewards[1].combo_multiplier)
                        # TODO use val from player 2
                        combo_bonus2 = (1. + np.square(c2/3) * rewards[2].combo_multiplier)
                        off_stage_bonus = (1. + np.float32(self.last_state.players[1].off_stage
                                        and self.last_state.players[2].off_stage) * rewards[1].off_stage_multiplier)

                        rewards[1].damage_rewards += combo_bonus1 * off_stage_bonus * dpercent2 - dpercent1 * combo_bonus2
                        if bot_port2:
                            rewards[2].damage_rewards += combo_bonus2 * off_stage_bonus * dpercent1 - dpercent2 * combo_bonus1

                        if dpercent2 > 0:
                            # encourages to do varied combos ?
                            bonus = 1.
                            if self.last_hit_action_state[1] == new_state.players[1].action:
                                bonus = 0.5
                            self.combo_counter[1] = np.minimum(self.combo_counter[1]+bonus, self.max_combo)

                            self.combo_counter[2] = 0
                            self.last_hit_action_state[1] = new_state.players[1].action

                        if dpercent1 > 0:
                            bonus = 1.
                            if self.last_hit_action_state[2] == new_state.players[2].action:
                                bonus = 0.5
                            self.combo_counter[2] = np.minimum(self.combo_counter[2] + bonus, self.max_combo)

                            self.combo_counter[1] = 0
                            self.last_hit_action_state[2] = new_state.players[2].action

                        # motivates fast combos ? looses one point after a second.
                        if self.combo_counter[1] > 0:
                            self.combo_counter[1] = self.combo_discounter(self.combo_counter[1])
                        if self.combo_counter[2] > 0:
                            self.combo_counter[2] = self.combo_discounter(self.combo_counter[2])

                        # DISTANCE
                        dist_before = np.sqrt(
                            np.square(self.last_state.players[1].position.x - self.last_state.players[2].position.x)
                            +
                            np.square(self.last_state.players[1].position.y - self.last_state.players[2].position.y)
                        )
                        dist_now1 = np.sqrt(
                            np.square(new_state.players[1].position.x - self.last_state.players[2].position.x)
                            +
                            np.square(new_state.players[1].position.y - self.last_state.players[2].position.y)
                        )
                        dist_now2 = np.sqrt(
                            np.square(self.last_state.players[1].position.x - new_state.players[2].position.x)
                            +
                            np.square(self.last_state.players[1].position.y - new_state.players[2].position.y)
                        )
                        ddist1 = np.clip(
                            dist_before - dist_now1, 0., 5.
                        )
                        ddist2 = np.clip(
                            dist_before - dist_now2, 0., 5.
                        )
                        rewards[1].distance_rewards += ddist1
                        if bot_port2:
                            rewards[2].distance_rewards += ddist2

                        # ACTION RELATED
                        for p, reward in rewards.items():
                            rewards[p].energy_costs -= current_actions[p].energy_cost


            self.last_state = new_state
