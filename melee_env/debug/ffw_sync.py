import numpy as np
from melee import Character, Stage
import tree
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


num_envs = 8

bot_configs = {1: BotConfig(character="FOX"),
               2: BotConfig(character="FOX")}


def alert_unsynched_states(path, *v):
    v0 = v[0]
    n_unsynched = 0
    if ("controller_state" in path or "costume" in path
            or "ecb" in path or "ecb_top" in path
            or "ecb_left" in path or "ecb_right" in path
    ):
        return 0

    for vn in v[1:]:
        if np.any(vn != v0):
            n_unsynched += 1

    if n_unsynched > 0:
        print(f"{path}: {v}")
    return n_unsynched


def gamestate2dict(gs):
    if hasattr(gs, "__slots__"):
        result = {}
        for k in gs.__slots__:
            result[k] = gamestate2dict(getattr(gs, k))
    elif isinstance(gs, dict):
        result = {}
        for k, v in gs.items():
            result[k] = gamestate2dict(v)
    elif isinstance(gs, list):
        return {i: gamestate2dict(item) for i, item in enumerate(gs)}
    else:
        result = gs
    return result



for i in range(32):

    env_conf = (
        SSBMConfig()
        .chars([
            Character.MARIO,
            Character.FOX,
            Character.CPTFALCON,
            Character.DK,
            #Character.KIRBY,
            Character.BOWSER,
            Character.LINK,
            #Character.SHEIK,
            Character.NESS,
            Character.PEACH,
            #Character.POPO,
            Character.PIKACHU,
            Character.SAMUS,
            Character.YOSHI,
            Character.JIGGLYPUFF,
            Character.MEWTWO,
            Character.LUIGI,
            Character.MARTH,
            #Character.ZELDA,
            Character.YLINK,
            Character.DOC,
            Character.FALCO,
            Character.PICHU,
            Character.GAMEANDWATCH,
            Character.GANONDORF,
            Character.ROY
        ])
        .stages([
            Stage.FINAL_DESTINATION,
            Stage.YOSHIS_STORY,
            Stage.POKEMON_STADIUM,
            Stage.BATTLEFIELD,
            Stage.DREAMLAND,
            Stage.FOUNTAIN_OF_DREAMS
        ])
        .players([PlayerType.BOT, PlayerType.BOT])
        .n_eval(-100)
        .set_obs_conf(obs_config)

        .render()
        # .debug()
        # .save_replays()
    )

    rendering_ssbm = SSBM(env_index=0, **dict(env_conf))
    ffw_ssbms = [
        SSBM(env_index=i + 1, **dict(env_conf))
        for i in range(num_envs)
    ]

    for env in ffw_ssbms + [rendering_ssbm]:
        env.reset(options=bot_configs)
    input()

    done = False
    for step in range(128):

        actions = {
            p: rendering_ssbm.action_space.sample()
            for p in rendering_ssbm.om.bot_ports
        }

        states = []
        interest = []
        terminateds = []
        for env in ffw_ssbms + [rendering_ssbm]:
            if step % 10 == 0:
                _, _, dones, _, _ = env.step(actions)
            else:
                env.step_nones()
            gs = env.get_gamestate()
            gs.projectiles = []

            states.append(gamestate2dict(env.get_gamestate()))
            interest.append(
                (gs.players[1].action_frame)
            )


            terminateds.append(dones["__all__"])

        print(interest)
        input()

        # synched_states = tree.map_structure_with_path(
        #     alert_unsynched_states,
        #     *states
        # )
        #
        # unsynched = np.array(tree.flatten(synched_states)) > 0
        # if np.any(unsynched):
        #     print(f" Num error {np.sum(unsynched)} | Step {step} ===================================================")
        #     input()

        if any(terminateds):
            break

    print(states)

    for env in ffw_ssbms + [rendering_ssbm]:
        env.close()
        del env







