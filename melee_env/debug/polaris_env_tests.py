import argparse
import sys
import unittest

import melee
from melee import Stage, Character, Action

from melee_env.action_space import SSBMActionSpace
from melee_env.enums import PlayerType
from melee_env.polaris_melee import SSBM
from melee_env.configs import SSBMConfig, SSBMObsConfig
from seedsmash2.submissions.bot_config import BotConfig

parser = argparse.ArgumentParser()

parser.add_argument('--exiai-path', type=str, required=True)
parser.add_argument('--fm-path', type=str, required=True)
parser.add_argument('--iso', type=str, required=True)
parser.add_argument('--render', action='store_true')
parser.add_argument('--ffw', action='store_true')


class PolarisEnvTest(unittest.TestCase):

    def test_game_init(self):
        """
        Start up the console and select characters/costumes, selects stage and wait for "ready go !"
        """

        env = SSBM(
            env_index=0,
            **ENV_CONFIG
        )

        options = {
            p: BotConfig(character="CPTFALCON", costume=p)
            for p in env.get_agent_ids()
        }
        try:
            env.reset(options=options)
            gamestate = env.get_gamestate()
            for p in options:
                self.assertEqual(gamestate.players[p].character, melee.Character.CPTFALCON)
                self.assertEqual(gamestate.players[p].costume, p)
                self.assertNotIn(gamestate.players[p].action, [Action.ENTRY, Action.ENTRY_END])

            self.assertGreater(gamestate.frame, -50)
            self.assertEqual(gamestate.menu_state, melee.Menu.IN_GAME)

        finally:
            env.close()


    def test_game_end(self):
        """
        Ends a game after 4 stocks, then resets
        """

        env = SSBM(
            env_index=0,
            **ENV_CONFIG
        )

        action_dict = SSBMActionSpace()
        options = {
            p: BotConfig(character="CPTFALCON", costume=p)
            for p in env.get_agent_ids()
        }
        try:
            env.reset(options=options)

            done = False
            c = 0
            while not done:
                if c % 20 == 0:
                    actions = {
                    p: action_dict.RESET_CONTROLLER
                    for p in options
                    }
                else:
                    actions = {
                    p: action_dict.LEFT
                    for p in options
                    }

                _, _, _, dones, _ = env.step(actions)
                c+=1
                self.assertLess(c, 2000)
                done = dones["__all__"]

            env.reset(options=options)
            gamestate = env.get_gamestate()

            for p in options:
                self.assertEqual(gamestate.players[p].character, melee.Character.CPTFALCON)
                self.assertEqual(gamestate.players[p].costume, p)
                self.assertNotIn(gamestate.players[p].action, [Action.ENTRY, Action.ENTRY_END])

            self.assertGreater(gamestate.frame, -50)
            self.assertEqual(gamestate.menu_state, melee.Menu.IN_GAME)

        finally:
            env.close()

    def test_synched(self):
        """
        Runs several games and checks if they remain synched.
        """

        n_envs = 8

        rendering_config = ENV_CONFIG.copy()
        rendering_config["use_ffw"] = False
        rendering_config["render"] = True

        envs = [SSBM(
                env_index=0,
                **rendering_config
            )] + [

            SSBM(
                env_index=i,
                **ENV_CONFIG
            )
            for i in range(1, n_envs)
        ]

        action_dict = SSBMActionSpace()
        options = {
            p: BotConfig(character="CPTFALCON", costume=p)
            for p in envs[0].get_agent_ids()
        }

        try:
            for env in envs:
                env.reset(options=options)
                env.step_console(60)
                env.step({
                    p: action_dict.LEFT
                    for p in options
                })
                env.step({
                    p: action_dict.LEFT
                    for p in options
                })
                env.step({
                    p: action_dict.WAVEDASH_NEUTRAL
                    for p in options
                })
                env.step({
                    p: action_dict.LEFT
                    for p in options
                })
                env.step({
                    p: action_dict.UP_RIGHT
                    for p in options
                })

            gamestates = [
                env.get_gamestate() for env in envs
            ]
            for p in options:
                self.assertListEqual([gamestate.players[p].character
                                      for gamestate in gamestates], [melee.Character.CPTFALCON]*n_envs)
                self.assertListEqual([gamestate.players[p].action
                                      for gamestate in gamestates], [gamestates[0].players[p].action]*n_envs)
                self.assertListEqual([gamestate.players[p].action_frame
                                      for gamestate in gamestates], [gamestates[0].players[p].action_frame]*n_envs)

        finally:
            for env in envs:
                env.close()


    def test_timeout(self):
        """
        Runs a game until timeout and reset.
        """

        env = SSBM(
            env_index=0,
            **ENV_CONFIG
        )

        action_dict = SSBMActionSpace()

        options = {
            p: BotConfig(character="CPTFALCON", costume=p)
            for p in env.get_agent_ids()
        }
        try:
            env.reset(options=options)
            gamestate = env.get_gamestate()

            done = False
            c = 0
            while not done:
                _, _, dones, _, _ = env.step(
                    {p: action_dict.RESET_CONTROLLER for p in options}
                )
                c += 1
                self.assertLess(c, 10000)
                done = dones["__all__"]
            env.reset(options=options)

            for p in options:
                self.assertEqual(gamestate.players[p].character, melee.Character.CPTFALCON)
                self.assertEqual(gamestate.players[p].costume, p)
                self.assertNotIn(gamestate.players[p].action, [Action.ENTRY, Action.ENTRY_END])

            self.assertGreater(gamestate.frame, -50)
            self.assertEqual(gamestate.menu_state, melee.Menu.IN_GAME)


        finally:
            env.close()


if __name__ == '__main__':
    obs_config = (
        SSBMObsConfig()
        .character()
        .stage()
        .projectiles()
        .delay(0)
    )

    parser.add_argument('unittest_args', nargs='*')
    ARGS = parser.parse_args()
    sys.argv[1:] = ARGS.unittest_args

    ENV_CONFIG = (
        SSBMConfig(
            faster_melee_path=ARGS.fm_path,
            exiai_path=ARGS.exiai_path,
            iso_path=ARGS.iso
        )
        .playable_characters([
            Character.CPTFALCON,
            Character.FOX,
        ])
        .playable_stages([
            Stage.FINAL_DESTINATION,
        ])
        .player_types([PlayerType.BOT, PlayerType.BOT])
        .obs_config(obs_config)
        .online_delay(0)
    )
    if ARGS.render:
        ENV_CONFIG = ENV_CONFIG.render()
    if ARGS.ffw:
        ENV_CONFIG = ENV_CONFIG.use_ffw()

    unittest.main()