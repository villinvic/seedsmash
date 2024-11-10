import os

from models import small_fc
from seedsmash.rllib.algorithm import SeedSmash

#os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["TUNE_RESULT_DIR"] = "ssbm"


from functools import partial

import ray
from ray import tune
from ray.rllib.algorithms.callbacks import make_multi_callbacks, RE3UpdateCallbacks
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.rllib.algorithms.appo.appo import APPOConfig
from melee.enums import Character, Stage
from melee_env.enums import PlayerType

from melee_env.autoregressive_actionspace import AutoregressiveActionModel, TestDummyAutoreg4, TestDummyAutoreg1
from melee_env.general_model import SSBMModelAux
from melee_env.seedsmash_model import SeedSmashModel
from melee_env.random_encoder import RandomEncoder
from melee_env.simple_action_model import EpsilonCategorical, SSBMModel
from melee_env.polaris_melee import SSBM
from melee_env.rllib_utils import SSBMCallbacks, TraceMallocCallback, LeagueImpala, SpectateWorkerTest, SpectateWorker, MultiCharConfig
from melee_env.configs import SSBMConfig, SSBM_OBS_Config


obs_config = (
    SSBM_OBS_Config()
    .character()
    #.ecb()
    .stage()
    .max_projectiles(4)  # one falco can generate more than 3 projectiles apparently ?
    .controller_state()
    .delay(4)
)

# config:
#     multiagent:
#         policies: {
#             "pol0": (None, obs-space, action-space, {"lr": 0.001}),
#             "pol1": (None, obs-space, action-space, {"lr": 0.002}),
#         }


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

env_config = (
    SSBMConfig()
    .chars([
        Character.FOX,
        Character.FALCO,
        Character.CPTFALCON,
        Character.MARIO,
        Character.DOC,
        Character.MARTH,
        Character.ROY,
        Character.GANONDORF,
        Character.JIGGLYPUFF,
        Character.LINK,
        Character.YLINK
    ])
    .stages([
        Stage.FINAL_DESTINATION,
        Stage.YOSHIS_STORY,
        Stage.POKEMON_STADIUM,
        Stage.BATTLEFIELD,
        Stage.DREAMLAND,

        # Stage.FOUNTAIN_OF_DREAMS  # TODO support FOD
        # falcon falco, jiggs falco, marth falcon, jigs falcon, falcon falcon, falcon fox, marth falcon, falco falco,
        # marth falco, jigs marth
    ])
    .players([PlayerType.BOT, PlayerType.BOT])
    .n_eval(-1)
    .set_obs_conf(obs_config)

    .render(idx=16)
    #.debug()
    #.save_replays()
)

# Actually used to visualize, not eval
eval_config = env_config.render().n_eval(0)

register_env("ssbm", lambda c: SSBM(c))

dummy_ssbm = SSBM(dict(env_config))

ma_config = env_config.get_ma_config("v3")

for k, v in ma_config["policies"].items():
    v.observation_space = dummy_ssbm.observation_space[1]
    v.action_space = dummy_ssbm.action_space[1]

ModelCatalog.register_custom_model(
    "disagreement_model",
    DisagreementModel,
)

evaluation = False

# RE3 = partial(
#                 RE3UpdateCallbacks,
#                 beta=0.01,
#                 embeds_dim=64,  # also change the random encoder
#                 k_nn=50,
#                 )

SSBM_RLLIB_CUSTOMS = partial(
    SSBMCallbacks,
    negative_reward_scale=0.95,
    killer_move_reward_scale=0.9,
    observation_manager=dummy_ssbm.om
)

del dummy_ssbm

config = (
    MultiCharConfig().callbacks(make_multi_callbacks([SSBM_RLLIB_CUSTOMS]))
    .environment(env="ssbm", env_config=dict(env_config), disable_env_checking=True)
    .resources(num_learner_workers=1, num_cpus_for_local_worker=1,
               num_cpus_per_worker=1, num_gpus=1)
    .rollouts(num_rollout_workers=110, num_envs_per_worker=1, rollout_fragment_length=512, remote_worker_envs=False,
              batch_mode="truncate_episodes", sample_async=True, create_env_on_local_worker=False)
    .multi_agent(policies=ma_config["policies"],
                 policy_mapping_fn=ma_config["policy_mapping_fn"],
                 policy_map_capacity=3,
                 policy_states_are_swappable=True,
                 )
    .training(lambda_=0.98, clip_param=0.4, vtrace_drop_last_ts=False, gamma=0.997, entropy_coeff=1.1e-3,
              lr=0.0005, train_batch_size=8192, kl_coeff=1., learner_queue_timeout=60*10, learner_queue_size=256,
              vtrace=True, replay_proportion=0., replay_buffer_num_slots=256,
              model={
                "max_seq_len": 32,
                "lstm_use_prev_action": False,
                "_time_major": False,
                "custom_model": "seed_smash"
              })
    .reporting(min_time_s_per_iteration=60, metrics_episode_collection_timeout_s=180)
    .experimental(_disable_action_flattening=True)
    .framework(framework="tf")
          )

config.worker_cls = SpectateWorker

config_dict = config.to_dict()

#
# run_conf = {
#     "callbacks": make_multi_callbacks([SSBM_RLLIB_CUSTOMS]),
#     "env": "ssbm",
#     # "env": "Ant-v4",
#     "env_config": dict(env_config),  # config to pass to env class
#     #"num_workers": 63,
#     "num_rollout_workers": 64,
#     "num_envs_per_worker": 1,
#     "num_gpus": 1.,
#     #"num_gpus_per_worker": (1 - 0.5) / 85.,
#     "replay_proportion": 0.5,
#     "replay_buffer_num_slots": 256,
#     "min_time_s_per_iteration": 20,
#     "multiagent": env_config.ma_config["v3"],
#
#     'observation_filter': "NoFilter",  # ConcurrentMeanStdFilter
#     #'metrics_episode_collection_timeout_s': 0,
#     "max_requests_in_flight_per_sampler_worker": 2,
#     "_disable_action_flattening": True,
#     #"_disable_preprocessor_api": True,
#     "disable_env_checking": True,
#     "model": {
#
#         # "free_log_std": True,
#         # Plugs lstm at the end of our model, automatically calculates the outputs/inputs dims.
#         # Care about bug in rllib rnn_sequencing
#         #"use_lstm": True,
#         # Max seq len for training the LSTM, defaults to 20.
#         "max_seq_len": 32,
#         # Size of the LSTM cell.
#         #"lstm_cell_size": 256,
#         # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
#         "lstm_use_prev_action": False,
#         # Whether to qfeed r_{t-1} to LSTM.
#         # "lstm_use_prev_reward": True, # Normalize
#         # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
#         "_time_major": False,
#
#         "custom_model": "seed_smash",
#         #"custom_action_dist": "epsilon_categorical",
#         #"custom_model_config": {"delay": obs_config["delay"]}
#
#     },
#     "gamma": 0.997,
#     "lambda": 0.98,
#     "entropy_coeff": 1.1e-3,
#     #"sample_async": True,
#
#     "lr": 0.0005,
#     #"minibatch_buffer_size": 2048*2,  # 3024,
#     "train_batch_size": 32768,
#     #"num_sgd_iter": 16,
#     #"grad_clip": 1.,
#     "kl_coeff": 1.,
#     "clip_param": 0.4,
#     "rollout_fragment_length": 128*4,
#     "learner_queue_timeout": 3000,
#     "learner_queue_size": 256,
#     "batch_mode": "truncate_episodes",
#     "vtrace": True,
#     # "normalize_actions": True,
#     "vf_loss_coeff": 0.5, # 0.2
#
#     "framework": "tf",
#     #"eager_tracing": True,
#     # === Evaluation Settings ===
#     # Evaluate with every `evaluation_interval` training iterations.
#     # The evaluation stats will be reported under the "evaluation" metric key.
#     # Note that evaluation is currently not parallelized, and that for Ape-X
#     # metrics are already only reported for the lowest epsilon workers.
#     "evaluation_interval": 100 if evaluation else None,
#     "evaluation_parallel_to_training": evaluation,
#
#     # Number of episodes to run per evaluation period. If using multiple
#     # evaluation workers, we will run at least this many episodes total.
#     "evaluation_duration": 1,
#
#     # Internal flag that is set to True for evaluation workers.
#     # "in_evaluation": False,
#     "evaluation_config": {
#         "env_config": dict(eval_config),
#         "explore": True,
#     },
#     "evaluation_num_workers": int(evaluation),
#
#     "exploration_config": {
#         "type": "StochasticSampling"
#     },
#
#     "enable_async_evaluation": True,
#     "always_attach_evaluation_results": False,
#
#     "max_num_worker_restarts": 5
#
# }

# app = TimerWindow()
# app.load_elapsed_time()
# app.protocol("WM_DELETE_WINDOW", app.on_closing)
# app.protocol("WM_SAVE_YOURSELF", app.on_closing)
# app.mainloop()


ray.init(object_store_memory=12_864_320_000)

exp = tune.run(
    SeedSmash,
    config=config_dict,
    num_samples=1,
    checkpoint_at_end=True,
    checkpoint_freq=100,
    stop={"training_iteration": 800000,
          },
    local_dir='checkpoints',
    reuse_actors=True,
    # resume=True
    #restore="/home/goji/Desktop/GEORGES/ssbm/checkpoints/SeedSmash_2023-08-30_15-33-08/SeedSmash_ssbm_c1f3e_00000_0_2023-08-30_15-33-08/checkpoint_000100"
)

