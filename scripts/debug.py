
import wandb
from melee.enums import Character, Stage
from polaris_melee.enums import PlayerType
from polaris_melee.env import SSBM
from polaris_melee.configs import SSBMConfig, SSBMObsConfig
from seedsmash.game_metrics_callbacks import SSBMCallbacks

from sacred import Experiment, Ingredient
from ml_collections import ConfigDict



exp_name = 'falcon_ditto_fsp_debug'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)


obs_config = (
    SSBMObsConfig()
    .character()
    .stage()
    .max_projectiles_per_owner(3)
    .delay(4)  # last used was 5 (half delay)
)

@ex.config
def cfg():
    iso = ''
    fm_path = ''
    exiai_path = ''
    replay_path = ''

    if iso == '':
        raise ValueError("Need a path for the melee iso.")
    if fm_path == '':
        raise ValueError("Need a path for the Faster Melee executable.")
    if exiai_path == '':
        raise ValueError("Need a path for the ExiAI executable.")
    if replay_path == '':
        raise ValueError("Need a path for slippi replays.")


    env_config = (
        SSBMConfig(
            faster_melee_path=fm_path,
            exiai_path=exiai_path,
            iso_path=iso,
            replay_path=replay_path
        )
        .playable_characters([
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

    num_workers = 64
    policy_path = 'policies.seedsmash_PPO'
    model_path = 'models.transformerlike'
    policy_class = 'PPO'
    model_class = 'TransformerLikeModel'
    trajectory_length = 256  # good for gae_lambda = 0.95
    max_seq_len = 32  # 32
    train_batch_size = 8192 * 4
    max_queue_size = train_batch_size * 10
    n_epochs = 32
    minibatch_size = train_batch_size
    allow_older_samples = True

    default_policy_config = {
        # model
        'policy_head_dims': [128],
        'value_head_dims': [128],
        'projectile_mlp_dims': [16, 16],

        'discount': 0.993,
        'action_state_reward_scale': 1.,

        'gae_lambda': 0.95,
        'entropy_cost': 1e-4,
        'lr': 4e-4,

        # PPO
        'grad_clip': 1.,
        'ppo_clip': 0.2,  # 0.3
        'initial_kl_coeff': 1.,
        'baseline_coeff': 0.5,
        'vf_clip': 10.,
        'kl_target': 1e-2,
    }

    compute_advantages_on_workers = True
    wandb_logdir = 'logs'
    report_freq = 5
    episode_metrics_smoothing = 0.95
    training_metrics_smoothing = 0.8
    inject_new_bots_freq_s = 60
    # FSP
    update_policy_history_freq = 100
    policy_history_length = 10

    checkpoint_config = dict(
        checkpoint_frequency=50,
        checkpoint_path=exp_path,
        stopping_condition={"environment_steps": 1e10},
        keep=4,
    )
    episode_callback_class = SSBMCallbacks

    restore = False


@ex.automain
def main(_config):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)
    from seedsmash.fsp_sync_trainer import FSP

    config = ConfigDict(_config)
    SSBM(**config["env_config"]).register()


    wandb.init(
        config=_config,
        project="Seedsmashv2",
        mode='online',
        group="debug2",
        name="falcon_fsp",
        notes=None,
        dir=config["wandb_logdir"]
    )

    trainer = FSP(config, restore=config.restore)
    trainer.run()