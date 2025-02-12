import argparse

from melee import Character, Stage

from polaris_melee.enums import PlayerType
from polaris_melee.env import SSBM
from polaris_melee.configs import SSBMObsConfig, SSBMConfig
from seedsmash.bot import Bot, BotConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fm-path', type=str, required=True)
    parser.add_argument('--iso', type=str, required=True)
    ARGS = parser.parse_args()

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
            .playable_characters([
                Character.CPTFALCON,
            ])
            .playable_stages([
                Stage.YOSHIS_STORY,
            ])
            .player_types([PlayerType.HUMAN_DEBUG, PlayerType.BOT])
            .render()
            .online_delay(0)
            .polling_mode()
    )

    bot_configs = {1: Bot(**BotConfig(character=Character.DOC, preferred_stage=Stage.YOSHIS_STORY)._asdict()),
                   2: Bot(**BotConfig(preferred_stage=Stage.YOSHIS_STORY)._asdict())}
    env = SSBM(env_index=0, **ENV_CONFIG)
    env.reset(options=bot_configs)
    done = False
    while not done:
        obs, reward, done, trunc, info = env.step({2:0})
        print(env.character_specific_observations[1].get())
        # print("================================[Metrics]================================")
        # print(dummy_ssbm
        # print("=========================================================================")

        print()

        done = done["__all__"]

    print(env.get_episode_metrics())

