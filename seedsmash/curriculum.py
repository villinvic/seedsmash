import copy

import numpy as np

from seedsmash.bots.bot_config import BotConfig
from seedsmash.utils import inject_botconfig


def linear_interpolation(a, b, l):
    return a * l + b * (1-l)


class Curriculum:

    def __init__(
            self,
            config,
            policy_config,
            bot_config: BotConfig
                 ):

        self.config = config
        self.curriculum_max_version = config.curriculum_max_version
        if bot_config.coaching_bot is not None:
            self.curriculum_max_version = self.curriculum_max_version//8

        self.update_freq = self.curriculum_max_version // config.curriculum_num_updates
        self.curriculum_stage = 0.

        self.initial_config = BotConfig(
                agressivity=50,
                reflexion=0,
                winning_desire=0,
                patience=100,
                creativity=50,
                off_stage_plays=0,
                combo_game=0,
                combo_breaker=0
            )

        self.initial_config._distance_reward_scale = 1.
        self.initial_config._damage_penalty_scale = 0.1
        self.initial_config._damage_reward_scale = 3.
        self.initial_config._neutralb_charge_reward_scale = 1.
        self.initial_config._shieldstun_reward_scale = 1.
        self.initial_config._coaching_scale = 1.


        self.target_config = bot_config
        self.current_config = copy.deepcopy(bot_config)
        self.train_config = copy.deepcopy(policy_config)

        inject_botconfig(self.train_config, self.current_config)

        # learn discount factor, start at 0.993
        # TODO: define log or linear update
        # damage scale 1 -> 0.2
        # distance rewards 1 -> 0
        # inflicted_damage 0 -> 1

    def update(self, version):
        self.curriculum_stage = np.minimum(version / self.curriculum_max_version, 1.)
        print(version, self.curriculum_stage, )

        if self.curriculum_stage == 1.:
            self.current_config.coaching_bot = None
            self.target_config.coaching_bot = None

        for k in (BotConfig.__characterisation_fields__
                  + self.current_config.__curriculum_fields__):
            setattr(self.current_config, k, linear_interpolation(
                getattr(self.initial_config, k), getattr(self.target_config, k), 1-self.curriculum_stage
            ))

        inject_botconfig(self.train_config, self.current_config)

    def get_metrics(self):
        d = {
            k: getattr(self.current_config, k)
            for k in (BotConfig.__characterisation_fields__
                      + self.current_config.__curriculum_fields__)
        }
        d.update(
            discount=self.train_config["discount"],
            action_state_reward_scale=self.train_config["action_state_reward_scale"]
        )
        d["curriculum_stage"] = self.curriculum_stage
        return d







