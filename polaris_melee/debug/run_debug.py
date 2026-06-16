import argparse
import os
import sys
import time
import unittest
from functools import lru_cache

import psutil
import ray

import numpy as np
from melee import Character, Stage, Action
import tree
from polaris_melee.enums import PlayerType
from polaris_melee.env import SSBM
from polaris_melee.configs import SSBMConfig, SSBMObsConfig
from seedsmash.bot import Bot, BotConfig, BotStats

parser = argparse.ArgumentParser()

parser.add_argument('--fm-path', type=str, default="")
parser.add_argument('--exiai-path', type=str, default="")
parser.add_argument('--iso', type=str, required=True)
parser.add_argument('--replay-path', type=str, required=True)


class PolarisEnvTest(unittest.TestCase):

    def test_manual_control(self):


        def do_stuff(env):
            actions = {
                p: env.action_space.sample() #np.random.choice([16, 24, 37, 39], p = [0.4, 0.4,0.1,0.1])
                for p in env.observation_builder.bot_ports
            }
            return actions
            # do stuff


        bot_configs = {1: Bot(**BotConfig(character=Character.GAMEANDWATCH, preferred_stage=Stage.YOSHIS_STORY, preferred_move=Action.NAIR, elo=1200)._asdict()),
                       2: Bot(**BotConfig(character=Character.GAMEANDWATCH, preferred_stage=Stage.YOSHIS_STORY,
                                          stats=BotStats(aggressivity=0, adaptability=0, creativity=0, techskill=0,
                                                         neutral=0, stagecontrol=0, offstage=0))._asdict())}

        env = SSBM(env_index=0, **ENV_CONFIG)
        env.reset(options=bot_configs)

        while True:

            t = time.time()
            actions = do_stuff(env)
            t2 = time.time()
            obs, _, dones, _, _ = env.step(actions)
            gs = env.get_gamestate()

            player = gs.players[1]

            # abs_movement = (abs(player.speed_y_attack) + abs(player.speed_x_attack)
            #                 + abs(player.speed_ground_x_self) + abs(player.speed_air_x_self)
            #                 + abs(player.speed_y_self))
            # print(abs_movement, player.speed_x_attack, player.speed_ground_x_self, player.speed_air_x_self)
            print(player.on_ground)
            #print(gs.players[1].custom, gs.player[2].custom)

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
            exiai_path=ARGS.exiai_path,
            iso_path=ARGS.iso,
            replay_path=ARGS.replay_path,
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
        .player_types([PlayerType.HUMAN_DEBUG, PlayerType.HUMAN_DEBUG])
        .render()
        .online_delay(9)
        .polling_mode()
        .save_replays()
    )

    unittest.main()







