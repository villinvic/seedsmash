
from functools import partial

import numpy as np
import wandb
from melee.enums import Character, Stage
from melee_env.enums import PlayerType
from melee_env.polaris_melee import SSBM
from melee_env.configs import SSBMConfig, SSBM_OBS_Config
from seedsmash2.game_metrics_callbacks import SSBMCallbacks

from sacred import Experiment
from sacred.observers import FileStorageObserver
from ml_collections import ConfigDict

from seedsmash2.submissions.bot_config import BotConfig

exp_name = 'seedsmash_test_appo'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)


obs_config = (
    SSBM_OBS_Config()
    .character()
    #.ecb()
    .stage()
    .max_projectiles(0)  # one falco can generate more than 3 projectiles apparently ?
    #.controller_state()
    .delay(0) # 4 (* 3)
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
    #.debug()
    #.save_replays()
)

dummy_ssbm = SSBM(**dict(env_conf))

# TODO: spectate command
# TODO: change projectile obs
# TODO: improve visual of scrolling matchup + new challenger approaching ?

# TODO :
# take as input belief state, for ex...:
# - distance travelled
# - actions picked
# - habits: techs
# - option habits: shielding, grabbing, dd, dash back, etc ? list of things ???
# - off stage
# - prefered move
# - moves that hit them
# - make this small !


@ex.config
def my_config():
    env_obj = dummy_ssbm
    env_obj.register()
    env = env_obj.env_id
    del env_obj
    env_config = dict(env_conf)

    # TODO: try batched inference
    num_workers = 64+1
    policy_path = 'polaris.policies.PPO'
    model_path = 'models.debug3'
    policy_class = 'PPO'
    model_class = 'Debug3'
    trajectory_length = 64 # 256 ?
    max_seq_len = 32
    train_batch_size = 12288 #7680
    max_queue_size = train_batch_size * 10
    n_epochs=3
    minibatch_size= train_batch_size//8

    default_policy_config = {
        'discount': 0.993,  # 0.997
        'action_state_reward_scale': 1.,

        'gae_lambda': 0.99, # 0.98
        'entropy_cost': 2.5e-3,#5e-4, # 1e-3 with impala, or around " 0.3, 0.4
        'lr': 5e-4,
        'fc_dims': [128, 128],
        'lstm_dim': 128,
        'grad_clip': 5.,

        # PPO
        'ppo_clip': 0.1, # 0.3
        'initial_kl_coeff': 1.,
        'kl_coeff_speed': 1.,
        'baseline_coeff': 0.5,
        'vf_clip': 10.,
        'kl_target': 1e-2,

        # Coaching
        'imitation_loss_coeff': 1.5e-2,

        'aux_loss_weight': 0.2,
        }

    compute_advantages_on_workers = True
    wandb_logdir = 'logs'
    report_freq = 10
    episode_metrics_smoothing = 0.95
    training_metrics_smoothing = 0.8
    inject_new_bots_freq_s = 60
    # FSP
    update_policy_history_freq = 400
    policy_history_length = 10

    checkpoint_config = dict(
        checkpoint_frequency=50,
        checkpoint_path=exp_path,
        stopping_condition={"environment_steps": 1e10},
        keep=4,
    )

    episode_callback_class = SSBMCallbacks
    negative_reward_scale = 0.95


# Define a simple main function that will use the configuration
@ex.main
def main(_config):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    from seedsmash2.fsp_sync_trainer import FSP

    # TODO: seeding
    # Access the configuration using _config
    c = ConfigDict(_config)
    wandb.init(
        config=c,
        project="Seedsmash",
        mode='online',
        group="debug",
        name="fictitious_action_stacking",
        notes=None,
        dir=c.wandb_logdir
    )
    # print("Experiment Configuration:")
    # print(c)

    trainer = FSP(c, restore=True)
    trainer.run()


# Run the experiment
if __name__ == '__main__':
    ex.run_commandline()


