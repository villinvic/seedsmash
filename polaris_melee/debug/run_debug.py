import argparse
import sys
import unittest

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

        bot_configs = {1: BotConfig(character="CPTFALCON"),
                       2: BotConfig(character="CPTFALCON")}

        env = SSBM(env_index=0, **ENV_CONFIG)
        env.reset(options=bot_configs)

        for step in range(512):

            actions = {
                p: env.action_space.sample()
                for p in env.observation_builder.bot_ports
            }
            _, _, dones, _, _ = env.step(actions)


            if dones["__all__"]:
                break



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
        .player_types([PlayerType.BOT, PlayerType.HUMAN_DEBUG])
        .render()
        .online_delay(0)
    )

    unittest.main()







