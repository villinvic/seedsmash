import argparse
import os
import sys
import time
import unittest
from functools import lru_cache

import psutil
import ray

import numpy as np
from melee import Character, Stage
import tree
from polaris_melee.enums import PlayerType
from polaris_melee.env import SSBM
from polaris_melee.configs import SSBMConfig, SSBMObsConfig
from seedsmash.bot import Bot, BotConfig

parser = argparse.ArgumentParser()

parser.add_argument('--fm-path', type=str, required=True)
parser.add_argument('--iso', type=str, required=True)

class PolarisEnvTest(unittest.TestCase):

    def test_manual_control(self):


        def do_stuff(env):
            actions = {
                p: env.action_space.sample() #np.random.choice([16, 24, 37, 39], p = [0.4, 0.4,0.1,0.1])
                for p in env.observation_builder.bot_ports
            }
            return actions
            # do stuff


        bot_configs = {1: Bot(**BotConfig(character=Character.FALCO, preferred_stage=Stage.YOSHIS_STORY)._asdict()),
                       2: Bot(**BotConfig(character=Character.DOC, preferred_stage=Stage.YOSHIS_STORY)._asdict())}

        env = SSBM(env_index=0, **ENV_CONFIG)
        env.reset(options=bot_configs)

        while True:

            t = time.time()
            actions = do_stuff(env)
            t2 = time.time()
            _, _, dones, _, _ = env.step(actions)
            t3 = time.time()
            if dones["__all__"]:
                print(env.get_episode_metrics())
                break



if __name__ == '__main__':

    parser.add_argument('unittest_args', nargs='*')
    ARGS = parser.parse_args()
    sys.argv[1:] = ARGS.unittest_args

    obs_config = (
        SSBMObsConfig()
        .character()
        .stage()
        .projectiles()
        .delay(0)
    )

    ENV_CONFIG = (
        SSBMConfig(
            faster_melee_path=ARGS.fm_path,
            exiai_path="",
            iso_path=ARGS.iso
        )
        .obs_config(obs_config)
        .playable_characters([
            Character.CPTFALCON,
        ])
        .playable_stages([
            Stage.YOSHIS_STORY,
            # Stage.YOSHIS_STORY,
            # Stage.POKEMON_STADIUM,
            # Stage.BATTLEFIELD,
            # Stage.DREAMLAND,
            # Stage.FOUNTAIN_OF_DREAMS
        ])
        .player_types([PlayerType.HUMAN_DEBUG, PlayerType.BOT])
        .render()
        .online_delay(0)
        .polling_mode()
    )

    unittest.main()







