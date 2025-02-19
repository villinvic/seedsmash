
import wandb
from melee.enums import Character, Stage
from polaris_melee.enums import PlayerType
from polaris_melee.env import SSBM
from polaris_melee.configs import SSBMConfig, SSBMObsConfig
from seedsmash.game_metrics_callbacks import SSBMCallbacks

from sacred import Experiment, Ingredient
from ml_collections import ConfigDict


exp_name = 'seedsmash_test_action_embed'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)

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

obs_config = (
    SSBMObsConfig()
    .character()
    # .ecb()
    .stage()
    .projectiles()
    # .controller_state()
    .delay(4)  # 4 (* 3)
)

@ex.config
def cfg():
    iso = ''
    fm_path = ''
    exiai_path = ''

    if iso == '':
        raise ValueError("Need a path for the melee iso.")
    if fm_path == '':
        raise ValueError("Need a path for the Faster Melee executable.")
    if exiai_path == '':
        raise ValueError("Need a path for the ExiAI executable.")

    env_config = (
        SSBMConfig(
            faster_melee_path=fm_path,
            exiai_path=exiai_path,
            iso_path=iso
        )
        .playable_characters([
            Character.MARIO,
            Character.FOX,
            Character.CPTFALCON,
            Character.DK,
            # Character.KIRBY,
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
        .playable_stages([
            Stage.FINAL_DESTINATION,
            Stage.YOSHIS_STORY,
            Stage.POKEMON_STADIUM,
            Stage.BATTLEFIELD,
            Stage.DREAMLAND,
            Stage.FOUNTAIN_OF_DREAMS
        ])
        .player_types([PlayerType.BOT, PlayerType.BOT])
        .obs_config(obs_config)
        .render()
        .online_delay(0)
        .use_ffw()
        .polling_mode()
    )

    env = SSBM.env_id

    # TODO: try batched inference
    num_workers = 64
    policy_path = 'policies.seedsmash_PPO'
    model_path = 'models.action_state_embed_model'
    policy_class = 'PPO'
    model_class = 'ActionEmbedModel'
    trajectory_length = 512 #128 # 256 ?
    max_seq_len = 32
    train_batch_size = 8192*4
    max_queue_size = train_batch_size * 10
    n_epochs=3
    minibatch_size= train_batch_size//16

    default_policy_config = {
        'discount': 0.994,  # 0.997
        'action_state_reward_scale': 1.,

        'gae_lambda': 0.95, # 0.98
        'entropy_cost': 1e-2,#5e-4, # 1e-3 with impala, or around " 0.3, 0.4
        'lr': 5e-4,

        # PPO
        'grad_clip': 5.,
        'ppo_clip': 0.25, # 0.3
        'initial_kl_coeff': 1.,
        'baseline_coeff': 0.25,
        'vf_clip': 10.,
        'kl_target': 1e-2,

        # seedsmash
        'aux_loss_weight': 0.05,
        'distillation_weight': 0.05,
        'distillation_temperature': 2.,
    }

    compute_advantages_on_workers = True
    wandb_logdir = 'logs'
    report_freq = 5
    episode_metrics_smoothing = 0.95
    training_metrics_smoothing = 0.8

    checkpoint_config = dict(
        checkpoint_frequency=50,
        checkpoint_path=exp_path,
        stopping_condition={"environment_steps": 1e10},
        keep=4,
    )
    episode_callback_class = SSBMCallbacks
    negative_reward_scale = 0.93

    database_game_update_freq_s = 120 # read new bots and push games
    database_state_update_freq_s = 120 # for metrics
    db_address = "192.168.1.100:5000"

    restore = False


@ex.automain
def main(_config):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)
    from seedsmash.seedsmash_sync_trainer import SeedSmashTrainer

    config = ConfigDict(_config)
    SSBM(**config["env_config"]).register()


    wandb.init(
        config=_config,
        project="Seedsmash",
        mode='online',
        group="debug",
        name="seedsmash_test",
        notes=None,
        dir=config["wandb_logdir"]
    )

    trainer = SeedSmashTrainer(config, restore=config.restore)
    trainer.run()