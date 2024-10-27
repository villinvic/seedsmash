import sys
from time import sleep

import fire
import numpy as np

from melee import Character, Stage
from melee_env.enums import PlayerType
from melee_env.melee_gym import SSBM
from melee_env.ssbm_config import SSBM_OBS_Config, SSBMConfig

obs_config = (
    SSBM_OBS_Config()
    .max_projectiles(3)
    .stage().character()
    #.ecb()
    .controller_state()
    .delay(1)
)

env_config = (
    SSBMConfig()
    .chars([
        Character.FOX,
        Character.FALCO,
        Character.MARIO,
        Character.DOC,
        Character.MARTH,
        Character.ROY,
        Character.CPTFALCON,
        Character.GANONDORF,
        Character.JIGGLYPUFF,
        Character.LINK,
        Character.LUIGI,
        Character.YLINK
    ])
    .stages([
        Stage.FINAL_DESTINATION,
        Stage.YOSHIS_STORY,
        Stage.POKEMON_STADIUM,
        Stage.BATTLEFIELD,
        Stage.DREAMLAND
        # falcon falco, jiggs falco, marth falcon, jigs falcon, falcon falcon, falcon fox, marth falcon, falco falco,
        # marth falco, jigs marth
    ])
    .players([PlayerType.BOT, PlayerType.BOT])
    .n_eval(0)
    .set_obs_conf(obs_config)

    #.render()
)

class Test:

    # TODO: put this in main code, but still allow equivalent probability to peak each char, the stage comes after.

    before_game_stuck_combs = [
            (Character.FOX, Character.ROY, Stage.POKEMON_STADIUM),
            (Character.FALCO, Character.CPTFALCON, Stage.YOSHIS_STORY),
            (Character.FALCO, Character.GANONDORF, Stage.YOSHIS_STORY),
            (Character.FALCO, Character.JIGGLYPUFF, Stage.YOSHIS_STORY),
            (Character.MARIO, Character.FALCO, Stage.YOSHIS_STORY),
            (Character.MARIO, Character.MARTH, Stage.YOSHIS_STORY),
            (Character.MARTH, Character.LUIGI, Stage.POKEMON_STADIUM),
            (Character.MARTH, Character.DOC, Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.FOX, Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.FALCO,Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.MARIO, Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.MARTH, Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.ROY, Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.JIGGLYPUFF, Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.LINK, Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.YLINK, Stage.POKEMON_STADIUM), # ?
            (Character.ROY, Character.FOX, Stage.BATTLEFIELD), # ?
            (Character.ROY, Character.FALCO, Stage.BATTLEFIELD), # ?
            (Character.ROY, Character.MARIO, Stage.BATTLEFIELD), # ?
            (Character.ROY,Character.DOC, Stage.BATTLEFIELD), # ?
            (Character.ROY, Character.MARTH, Stage.BATTLEFIELD), # ?
            (Character.ROY, Character.ROY, Stage.BATTLEFIELD),
            (Character.ROY, Character.CPTFALCON, Stage.BATTLEFIELD),
            (Character.ROY, Character.GANONDORF, Stage.BATTLEFIELD),
            (Character.ROY,Character.JIGGLYPUFF, Stage.BATTLEFIELD),
            (Character.ROY, Character.LINK, Stage.BATTLEFIELD),
            (Character.ROY, Character.LUIGI, Stage.BATTLEFIELD),
            (Character.ROY, Character.YLINK, Stage.BATTLEFIELD),
            (Character.CPTFALCON, Character.LUIGI, Stage.YOSHIS_STORY),
            (Character.CPTFALCON, Character.ROY, Stage.POKEMON_STADIUM),
            (Character.CPTFALCON, Character.JIGGLYPUFF, Stage.FINAL_DESTINATION),
            (Character.CPTFALCON, Character.LINK, Stage.DREAMLAND),
            (Character.JIGGLYPUFF, Character.CPTFALCON, Stage.BATTLEFIELD),
            (Character.JIGGLYPUFF, Character.LINK, Stage.DREAMLAND),
            (Character.LINK, Character.FOX, Stage.YOSHIS_STORY),
            (Character.LINK, Character.FALCO, Stage.YOSHIS_STORY),
            (Character.LINK, Character.DOC, Stage.YOSHIS_STORY),
            (Character.LINK, Character.MARIO, Stage.YOSHIS_STORY),
            (Character.LINK, Character.MARTH, Stage.YOSHIS_STORY),
            (Character.LINK, Character.ROY, Stage.YOSHIS_STORY),
            (Character.LINK, Character.CPTFALCON, Stage.YOSHIS_STORY),
            (Character.LINK, Character.GANONDORF, Stage.YOSHIS_STORY),
            (Character.LINK, Character.JIGGLYPUFF, Stage.YOSHIS_STORY),
            (Character.LINK, Character.LINK, Stage.YOSHIS_STORY),
            (Character.LINK, Character.LUIGI, Stage.YOSHIS_STORY),
            (Character.LINK, Character.YLINK, Stage.YOSHIS_STORY),
            (Character.LINK, Character.YLINK, Stage.POKEMON_STADIUM),
            (Character.LUIGI, Character.CPTFALCON, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.FOX, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.FOX, Stage.POKEMON_STADIUM),
            (Character.YLINK, Character.FALCO, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.MARIO, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.DOC, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.MARTH, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.ROY, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.CPTFALCON, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.CPTFALCON, Stage.POKEMON_STADIUM),
            (Character.YLINK, Character.GANONDORF, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.YLINK, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.LINK, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.JIGGLYPUFF, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.CPTFALCON, Stage.BATTLEFIELD),
            (Character.YLINK, Character.CPTFALCON, Stage.DREAMLAND),
            (Character.YLINK, Character.LINK, Stage.POKEMON_STADIUM),
            (Character.YLINK, Character.LUIGI, Stage.POKEMON_STADIUM),
            (Character.YLINK, Character.LUIGI, Stage.YOSHIS_STORY)

    #(Character.CPTFALCON,  Character.FOX, Stage.YOSHIS_STORY),

    ]

    ingame_stuck_combs = [
        (Character.DOC, Character.JIGGLYPUFF, Stage.FINAL_DESTINATION),
        (Character.FALCO, Character.LINK, Stage.YOSHIS_STORY),
        (Character.FALCO, Character.YLINK, Stage.YOSHIS_STORY),
        (Character.JIGGLYPUFF, Character.FOX, Stage.POKEMON_STADIUM),
        (Character.DOC, Character.ROY, Stage.POKEMON_STADIUM), # ?
        (Character.DOC, Character.LINK, Stage.POKEMON_STADIUM), # ?
    ]




    def __init__(self, logfile=__file__):
        self.logfile = logfile.split(".")[0]

    def debug(self):

        sys.stdout = open(self.logfile + "_debug.log", 'w')
        sys.stderr = sys.stdout

        env = SSBM(dict(env_config))
        # Test all Chars and Stage combinations and see where it bugs
        for char1 in env.config["chars"]:
            for char2 in env.config["chars"]:
                for stage in env.config["stages"]:

                    env.reset(options=dict(
                        chars=[char1, char2],
                        stage=stage,
                    ))

                    env.close()

                    del env
                    env = SSBM(dict(env_config))

    def debug_final(self):

        sys.stdout = open(self.logfile + "_debug_final.log", 'w')
        sys.stderr = sys.stdout

        env = SSBM(dict(env_config))
        # Test all Chars and Stage combinations and see where it bugs
        c1 = env.config["chars"].copy()
        c2 = env.config["chars"].copy()
        np.random.shuffle(env.config["stages"])
        np.random.shuffle(c1)
        np.random.shuffle(c2)

        for i, char1 in enumerate(c1):
            for char2 in c2:
                for stage in env.config["stages"]:
                    if (char1, char2, stage) not in Test.before_game_stuck_combs + Test.ingame_stuck_combs:
                        env.reset(options=dict(
                            chars=[char1, char2],
                            stage=stage,
                        ))

                        done = False

                        while not done:
                            obs, _, dones, _, _ = env.step(action_dict={aid: env.action_space.sample() for aid in env.get_agent_ids()})
                            done = dones["__all__"]
                            curr_state = env.get_gamestate()

                        print(curr_state.frame, curr_state.players[1].x)


    def debug_stuck(self):
        sys.stdout = open(self.logfile + "_debug_stuck.log", 'w')

        env = SSBM(dict(env_config))

        for char1, char2, stage in Test.before_game_stuck_combs:
            env.reset(options=dict(
                chars=[char1, char2],
                stage=stage,
            ))

            env.close()

            del env
            env = SSBM(dict(env_config))

    def debug_stuck_ingame(self):
        sys.stdout = open(self.logfile + "_debug_stuck_ingame.log", 'w')

        env = SSBM(dict(env_config))

        for char1, char2, stage in Test.ingame_stuck_combs:
            env.reset(options=dict(
                chars=[char1, char2],
                stage=stage,
            ))
            print("Reset done.")

            dummy_action = {
                1: (1, 0, 0),
                2: (3, 0, 0)
            }

            done = False
            while not done:
                _, _, dones, _ = env.step(action_dict=dummy_action)
                done = dones["__all__"]
                curr_state = env.get_gamestate()
                print(curr_state.frame, curr_state.players[1].x)






if __name__ == '__main__':
    fire.Fire(Test)