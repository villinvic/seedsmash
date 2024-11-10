from melee import Character, Stage

from melee_env.enums import PlayerType
from melee_env.polaris_melee import SSBM
from melee_env.configs import SSBM_OBS_Config, SSBMConfig
from seedsmash2.submissions.bot_config import BotConfig

obs_config = (
    SSBM_OBS_Config()
    .character()
    #.ecb()
    .stage()
    .max_projectiles(0)
    #.controller_state()
    .delay(5)
)

env_conf = (
    SSBMConfig()
    .chars([
        # Character.MARIO,
        Character.FOX,
        Character.CPTFALCON,
        # Character.DK,
        # #Character.KIRBY,
        # Character.BOWSER,
        # Character.LINK,
        # #Character.SHEIK,
        # Character.NESS,
        # Character.PEACH,
        # #Character.POPO,
        # Character.PIKACHU,
        # Character.SAMUS,
        # Character.YOSHI,
        # Character.JIGGLYPUFF,
        # Character.MEWTWO,
        # Character.LUIGI,
        # Character.MARTH,
        # #Character.ZELDA,
        # Character.YLINK,
        # Character.DOC,
        # Character.FALCO,
        # Character.PICHU
        # Character.GAMEANDWATCH,
        # Character.GANONDORF,
        # Character.ROY
    ])
    .stages([
        #Stage.FINAL_DESTINATION,
        Stage.YOSHIS_STORY,
        # Stage.POKEMON_STADIUM,
        #Stage.BATTLEFIELD,
        # Stage.DREAMLAND,
        # Stage.FOUNTAIN_OF_DREAMS
    ])
    .players([PlayerType.HUMAN_DEBUG, PlayerType.BOT])
    .n_eval(-100)
    .set_obs_conf(obs_config)

    .render()
    #.debug()
    #.save_replays()
)

dummy_ssbm = SSBM(env_index=0, **dict(env_conf))
bot_configs = {1: BotConfig(character="CPTFALCON"),
               2: BotConfig(character="CPTFALCON")}


dummy_ssbm.reset(options=bot_configs)
done = False
for i in range(20*20):
    dummy_ssbm.step({2:0})
    print("================================[Metrics]================================")
    print(dummy_ssbm.reward_functions[1].get_metrics(i))
    print("=========================================================================")


