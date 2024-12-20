
from functools import partial

import numpy as np
from melee.enums import Character, Stage
from melee_env.enums import PlayerType
from melee_env.melee_gym import SSBM
from melee_env.ssbm_config import SSBMConfig, SSBM_OBS_Config
from seedsmash2.game_metrics_callbacks import SSBMCallbacks

from sacred import Experiment
from sacred.observers import FileStorageObserver
from ml_collections import ConfigDict

from seedsmash2.submissions.bot_config import BotConfig

exp_name = 'fc_model_test_impala'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)


obs_config = (
    SSBM_OBS_Config()
    .character()
    #.ecb()
    .stage()
    .max_projectiles(4)  # one falco can generate more than 3 projectiles apparently ?
    #.controller_state()
    .delay(1) # 4 (* 3)
)

# TODO : should add stage idx for walls (walljump and stuff)
# TODO : callback for reward_function shaping update on timesteps,
#           gradually decrease the distance reward
#           increase winning rewards
#           increase gamma ?
#           increase neg_scale
# TODO : Fix continuous input idx -4 and -8

# TODO: CRAZY IDEA
#       viewers can submit bot settings, inject it into the population (every k hours, some come from offsprings/mutation,
#       small proportion is viewer generated instead of freely resampled.)
#       have it keep the name until it gets kicked out, or reproduced (name genetics).


# TODO: track combos we did (action states) and reward for new combos
# TODO: try:next
# #       - feature extractor that learns to predict the state (do we split us and opponnent ?)
#       - use a DenseNet architecture ?

# Chars to include: Sheik, Samus, Pickachu

env_conf = (
    SSBMConfig()
    .chars([
        Character.FOX,
        Character.FALCO,
        Character.CPTFALCON,
        # Character.MARIO,
        # Character.DOC,
        Character.MARTH,
        # Character.ROY,
        # Character.GANONDORF,
        # Character.JIGGLYPUFF,
        # Character.LINK,
        # Character.YLINK
    ])
    .stages([
        Stage.FINAL_DESTINATION,
        #Stage.YOSHIS_STORY, # Why is this so buggy ?
        Stage.POKEMON_STADIUM,
        Stage.BATTLEFIELD,
        Stage.DREAMLAND,

        # Stage.FOUNTAIN_OF_DREAMS  # TODO support FOD
        # falcon falco, jiggs falco, marth falcon, jigs falcon, falcon falcon, falcon fox, marth falcon, falco falco,
        # marth falco, jigs marth
    ])
    .players([PlayerType.BOT, PlayerType.CPU])
    .n_eval(-100)
    .set_obs_conf(obs_config)

    .render()
    #.debug()
    #.save_replays()
)

dummy_ssbm = SSBM(**dict(env_conf))

@ex.config
def my_config():
    env_obj = dummy_ssbm
    env_obj.register()
    env = env_obj.env_id
    del env_obj
    env_config = dict(env_conf)

    num_workers = 64
    policy_path = 'policies.IMPALAMelee'
    model_path = 'models.small_fc'
    policy_class = 'IMPALA'
    model_class = 'SmallFC'
    trajectory_length = 32
    train_batch_size = 1024*8
    max_queue_size = train_batch_size * 10
    max_seq_len = 32

    default_policy_config = {
        'discount': 0.995,  # 0.997
        'entropy_cost': 1e-3, # 5e-5 with impala, or around " 0.3, 0.4
        'popart_std_clip': 1e-2,
        'popart_lr': 1e-1,
        'grad_clip': 2.,
        'lr': 3e-4,
        'rms_prop_rho': 0.99,
        'rms_prop_epsilon': 1e-5,
        'fc_dims': [128, 128],
        'greedy_epsilon': 1e-2,
        }

    policy_params = [dict(
        name="LINK",
        config=default_policy_config.copy(),
        options=BotConfig(
            character="LINK",
            costume=2
        )
    ),
        # dict(
        #     name="FALCON",
        #     config=default_policy_config.copy(),
        #     options=dict(character=Character.CPTFALCON)
        # ),
        # dict(
        #     name="FALCO",
        #     config=default_policy_config.copy(),
        #     options=dict(character=Character.FALCO)
        # )
    ]

    tensorboard_logdir = 'small_fc_tests_fox_falcon'
    report_freq = 5
    episode_metrics_smoothing = 0.8
    training_metrics_smoothing = 0.8


    checkpoint_config = dict(
        checkpoint_frequency=1000,
        checkpoint_path=exp_path,
        stopping_condition={"environment_steps": 1e9},
        keep=3,
    )

    episode_callback_class = partial(
    SSBMCallbacks,
    negative_reward_scale=1., #0.95
)

# Define a simple main function that will use the configuration
@ex.main
def main(_config):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)
    from polaris.trainers import AsyncTrainer


    # TODO: seeding
    # Access the configuration using _config
    c = ConfigDict(_config)
    print("Experiment Configuration:")
    print(c)
    trainer = AsyncTrainer(c, restore=False)
    trainer.run()


# Run the experiment
if __name__ == '__main__':
    ex.run_commandline()


