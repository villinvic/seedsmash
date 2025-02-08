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
from polaris_melee.configs import SSBMConfig
from seedsmash.bots.bot_config import BotConfig


parser = argparse.ArgumentParser()

parser.add_argument('--fm-path', type=str, required=True)
parser.add_argument('--iso', type=str, required=True)

class PolarisEnvTest(unittest.TestCase):

    def test_manual_control(self):


        def do_stuff(env):
            actions = {
                p: np.random.choice([16, 37, 39], p = [0.8,0.1,0.1])
                for p in env.observation_builder.bot_ports
            }
            return actions
            # do stuff

        @ray.remote(num_cpus=1, num_gpus=0)  # Allocates 2 CPUs and 2 GB of RAM
        def ray_worker():
            #p = psutil.Process()
            # print(p.cpu_affinity())
            # p.cpu_affinity([0])  # Bind to core 0
            #curr_cpu = psutil.Process().cpu_num()

            bot_configs = {1: BotConfig(character="CPTFALCON"),
                           2: BotConfig(character="CPTFALCON")}

            env = SSBM(env_index=0, **ENV_CONFIG)
            env.reset(options=bot_configs)

            for step in range(2048):

                t = time.time()
                actions = do_stuff(env)
                t2 = time.time()

                _, _, dones, _, _ = env.step(actions)

                print(env.get_gamestate().players[2].invulnerability_type)

                t3 = time.time()
                print(t2 - t, t3 - t2, (t3-t)*20)

                # if dones["__all__"]:
                #     break


        object_ref = ray_worker.remote()
        result = ray.get(object_ref)



if __name__ == '__main__':

    parser.add_argument('unittest_args', nargs='*')
    ARGS = parser.parse_args()
    sys.argv[1:] = ARGS.unittest_args

    ENV_CONFIG = (
        SSBMConfig(
            faster_melee_path=ARGS.fm_path,
            exiai_path="",
            iso_path=ARGS.iso
        )
        .playable_characters([
            Character.CPTFALCON,
        ])
        .playable_stages([
            Stage.FINAL_DESTINATION,
            # Stage.YOSHIS_STORY,
            # Stage.POKEMON_STADIUM,
            # Stage.BATTLEFIELD,
            # Stage.DREAMLAND,
            # Stage.FOUNTAIN_OF_DREAMS
        ])
        .player_types([PlayerType.BOT, PlayerType.BOT])
        .render()
        .online_delay(2)
        .polling_mode()
    )

    unittest.main()







