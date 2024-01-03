import copy
import importlib
import itertools
import logging
import os
import platform
import queue
import threading
import tracemalloc
from collections import defaultdict
from functools import partial
from typing import Dict, Tuple, Optional, DefaultDict, Set, Callable, Type, Union, List


import numpy as np
import ray
from gymnasium import Space
from ray.actor import ActorHandle
from ray.rllib import SampleBatch, RolloutWorker, MultiAgentEnv
from ray.rllib.algorithms import AlgorithmConfig, Algorithm
from ray.rllib.algorithms.impala import Impala
from ray.rllib.algorithms.appo import APPO, APPOConfig
from ray.rllib.algorithms.impala.impala import make_learner_thread, AggregatorWorker, logger
from ray.rllib.algorithms.ppo.ppo import UpdateKL
from ray.rllib.env import EnvContext, ExternalMultiAgentEnv
from ray.rllib.env.base_env import convert_to_base_env
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.evaluation import Episode, AsyncSampler, SyncSampler
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.evaluation.env_runner_v2 import _PerfStats
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.evaluation.rollout_worker import _update_env_seed_if_necessary
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.buffers.mixin_replay_buffer import MixInMultiAgentReplayBuffer
from ray.rllib.models import Preprocessor
from ray.rllib.offline import get_dataset_and_shards, IOContext, InputReader, OutputWriter
from ray.rllib.offline.estimators import OffPolicyEstimator, ImportanceSampling, WeightedImportanceSampling, \
    DirectMethod, DoublyRobust
from ray.rllib.offline.offline_evaluation_utils import remove_time_dim
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, concat_samples
from ray.rllib.utils import override, try_import_tf, check_env, Filter
import psutil
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.actors import create_colocated_actors
from ray.rllib.utils.debug import update_global_seed_if_necessary
from ray.rllib.utils.deprecation import DEPRECATED_VALUE, deprecation_warning
from ray.rllib.utils.error import ERR_MSG_NO_GPUS, HOWTO_CHANGE_CONFIG
from ray.rllib.utils.filter import NoFilter
from ray.rllib.utils.metrics import NUM_ENV_STEPS_TRAINED, NUM_AGENT_STEPS_TRAINED
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.rllib.utils.replay_buffers import ReplayMode
from ray.rllib.utils.typing import AgentID, PolicyID, ResultDict, PartialAlgorithmConfigDict, EnvCreator, EnvType, \
    SampleBatchType, ModelWeights
from ray.rllib.utils.tf_utils import (
    get_gpu_devices as get_tf_gpu_devices,
    get_tf_eager_cls_if_necessary,
)

from ray.rllib.algorithms.callbacks import RE3UpdateCallbacks
from ray.rllib.algorithms.alpha_star.league_builder import LeagueBuilder
import gymnasium as gym
from ray.util import disable_log_once_globally, enable_periodic_logging, log_once
from ray.util.iter import ParallelIteratorWorker

from melee import Action, Character
from melee_env.observation_space_v2 import ObsBuilder
from melee_env.rewards import SSBMRewards

tf1, tf, tfv = try_import_tf()

class SelfPlay(LeagueBuilder):

    def __init__(self, algo: Algorithm, algo_config):
        super().__init__(algo, algo_config=algo_config)

        self.train_counter = 1
        self.previous_policies = []
        self.to_remove = None

    def build_league(self, result: ResultDict) -> None:
        if self.train_counter % 250 == 0:

            local_worker = self.algo.workers.local_worker()

            trainable_policies = local_worker.get_policies_to_train()
            non_trainable_policies = (
                    set(local_worker.policy_map.keys()) - trainable_policies
            )
            to_remove = None
            if len(self.previous_policies) > 4:
                to_remove = self.previous_policies.pop(0)
                non_trainable_policies.remove(to_remove)
                if self.to_remove is not None:
                    non_trainable_policies.remove(self.to_remove)

            added_policy = self.algo.get_policy("TRAINED")
            added_policy_name = f"old_policy_{self.train_counter}"
            print("ADDING POLICY:", added_policy_name)
            self.previous_policies.append(added_policy_name)
            non_trainable_policies.add(added_policy_name)

            print("BIBIBOB", non_trainable_policies)

            # Update our mapping function accordingly.
            def policy_mapping_fn(agent_id, episode, worker: RolloutWorker, **kwargs):
                # TODO : swap sometimes the agents ? right now index one is always the trained policy
                if agent_id == 1 or len(non_trainable_policies) == 0:
                    return "TRAINED"
                else:

                    older = np.random.choice(list(non_trainable_policies))
                    return np.random.choice(["TRAINED", older], p=[0.5, 0.5])

            self.algo.add_policy(added_policy_name, policy=added_policy, policy_mapping_fn=policy_mapping_fn,
                                 policies_to_train=["TRAINED"],
                                 observation_space=added_policy.observation_space,
                                 action_space=added_policy.action_space)
            print("NEW POLICY ADDED:", added_policy_name)
            if self.to_remove is not None:
                self.algo.remove_policy(self.to_remove)
                print("POLICY REMOVED:", added_policy_name)

            if to_remove is not None:
                self.to_remove = to_remove



        self.train_counter += 1

    def __getstate__(self) -> Optional[Dict]:
        return {"previous_policies": self.previous_policies.copy(),
                "train_counter": self.train_counter,
                "to_remove": self.to_remove}

    def __setstate__(self, state: Dict):

        self.train_counter = state.get("train_counter", 0)
        self.previous_policies = state.get("previous_policies", [])
        self.to_remove = state.get("to_remove", None)


class MultiCharConfig(APPOConfig):

    def __init__(self, algo_class=None):
        """Initializes a APPOConfig instance."""
        super().__init__(algo_class=algo_class or APPO)






class LeagueImpala(Impala):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.league_builder: LeagueBuilder

    @override(Impala)
    def setup(self, config: AlgorithmConfig):
        super().setup(config)

        self.league_builder = SelfPlay(algo=self, algo_config=self.config)

    @override(Impala)
    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.update(
            {
                "league_builder": self.league_builder.__getstate__(),
            }
        )
        return state

    @override(Impala)
    def __setstate__(self, state: dict) -> None:
        state_copy = state.copy()
        super().__setstate__(state_copy)
        self.league_builder.__setstate__(state.pop("league_builder", {}))

    @override(Impala)
    def step(self) -> ResultDict:
        # Perform a full step (including evaluation).
        result = super().step()

        # Based on the (train + evaluate) results, perform a step of
        # league building.
        self.league_builder.build_league(result=result)
        print("Train step:", self.league_builder.train_counter)

        return result


class SpectateWorkerTest(APPO):

    @override(APPO)
    def setup(self, config: AlgorithmConfig):

        # Setup our config: Merge the user-supplied config dict (which could
        # be a partial config dict) with the class' default.
        if not isinstance(config, AlgorithmConfig):
            assert isinstance(config, PartialAlgorithmConfigDict)
            config_obj = self.get_default_config()
            if not isinstance(config_obj, AlgorithmConfig):
                assert isinstance(config, PartialAlgorithmConfigDict)
                config_obj = AlgorithmConfig().from_dict(config_obj)
            config_obj.update_from_dict(config)
            config_obj.env = self._env_id
            self.config = config_obj

        # Set Algorithm's seed after we have - if necessary - enabled
        # tf eager-execution.
        update_global_seed_if_necessary(self.config.framework_str, self.config.seed)

        self._record_usage(self.config)

        # Create the callbacks object.
        self.callbacks = self.config.callbacks_class()

        if self.config.log_level in ["WARN", "ERROR"]:
            logger.info(
                f"Current log_level is {self.config.log_level}. For more information, "
                "set 'log_level': 'INFO' / 'DEBUG' or use the -v and "
                "-vv flags."
            )
        if self.config.log_level:
            logging.getLogger("ray.rllib").setLevel(self.config.log_level)

        # Create local replay buffer if necessary.
        self.local_replay_buffer = self._create_local_replay_buffer_if_necessary(
            self.config
        )

        # Create a dict, mapping ActorHandles to sets of open remote
        # requests (object refs). This way, we keep track, of which actors
        # inside this Algorithm (e.g. a remote RolloutWorker) have
        # already been sent how many (e.g. `sample()`) requests.
        self.remote_requests_in_flight: DefaultDict[
            ActorHandle, Set[ray.ObjectRef]
        ] = defaultdict(set)

        self.workers: Optional[WorkerSet] = None
        self.train_exec_impl = None

        # Offline RL settings.
        input_evaluation = self.config.get("input_evaluation")
        if input_evaluation is not None and input_evaluation is not DEPRECATED_VALUE:
            ope_dict = {str(ope): {"type": ope} for ope in input_evaluation}
            deprecation_warning(
                old="config.input_evaluation={}".format(input_evaluation),
                new="config.evaluation(evaluation_config=config.overrides("
                    f"off_policy_estimation_methods={ope_dict}"
                    "))",
                error=True,
                help="Running OPE during training is not recommended.",
            )
            self.config.off_policy_estimation_methods = ope_dict

        # Deprecated way of implementing Trainer sub-classes (or "templates"
        # via the `build_trainer` utility function).
        # Instead, sub-classes should override the Trainable's `setup()`
        # method and call super().setup() from within that override at some
        # point.
        # Old design: Override `Trainer._init`.
        _init = False
        try:
            self._init(self.config, self.env_creator)
            _init = True
        # New design: Override `Trainable.setup()` (as indented by tune.Trainable)
        # and do or don't call `super().setup()` from within your override.
        # By default, `super().setup()` will create both worker sets:
        # "rollout workers" for collecting samples for training and - if
        # applicable - "evaluation workers" for evaluation runs in between or
        # parallel to training.
        # TODO: Deprecate `_init()` and remove this try/except block.
        except NotImplementedError:
            pass

        # Only if user did not override `_init()`:
        if _init is False:
            # - Create rollout workers here automatically.
            # - Run the execution plan to create the local iterator to `next()`
            #   in each training iteration.
            # This matches the behavior of using `build_trainer()`, which
            # has been deprecated.
            self.workers = WorkerSetWithSpectator(
                env_creator=self.env_creator,
                validate_env=self.validate_env,
                default_policy_class=self.get_default_policy_class(self.config),
                config=self.config,
                num_workers=self.config.num_rollout_workers,
                local_worker=True,
                logdir=self.logdir,
            )

            # TODO (avnishn): Remove the execution plan API by q1 2023
            # Function defining one single training iteration's behavior.
            if self.config._disable_execution_plan_api:
                # Ensure remote workers are initially in sync with the local worker.
                self.workers.sync_weights()
            # LocalIterator-creating "execution plan".
            # Only call this once here to create `self.train_exec_impl`,
            # which is a ray.util.iter.LocalIterator that will be `next`'d
            # on each training iteration.
            else:
                self.train_exec_impl = self.execution_plan(
                    self.workers, self.config, **self._kwargs_for_execution_plan()
                )

            # Now that workers have been created, update our policies
            # dict in config[multiagent] (with the correct original/
            # unpreprocessed spaces).
            self.config["multiagent"][
                "policies"
            ] = self.workers.local_worker().policy_dict

        # Compile, validate, and freeze an evaluation config.
        self.evaluation_config = self.config.get_evaluation_config_object()
        self.evaluation_config.validate()
        self.evaluation_config.freeze()

        # Evaluation WorkerSet setup.
        # User would like to setup a separate evaluation worker set.
        # Note: We skip workerset creation if we need to do offline evaluation
        if self._should_create_evaluation_rollout_workers(self.evaluation_config):
            _, env_creator = self._get_env_id_and_creator(
                self.evaluation_config.env, self.evaluation_config
            )

            # Create a separate evaluation worker set for evaluation.
            # If evaluation_num_workers=0, use the evaluation set's local
            # worker for evaluation, otherwise, use its remote workers
            # (parallelized evaluation).
            self.evaluation_workers: WorkerSet = WorkerSet(
                env_creator=env_creator,
                validate_env=None,
                default_policy_class=self.get_default_policy_class(self.config),
                config=self.evaluation_config,
                num_workers=self.config.evaluation_num_workers,
                logdir=self.logdir,
            )

            if self.config.enable_async_evaluation:
                self._evaluation_weights_seq_number = 0

        self.evaluation_dataset = None
        if (
                self.evaluation_config.off_policy_estimation_methods
                and not self.evaluation_config.ope_split_batch_by_episode
        ):
            # the num worker is set to 0 to avoid creating shards. The dataset will not
            # be repartioned to num_workers blocks.
            logger.info("Creating evaluation dataset ...")
            ds, _ = get_dataset_and_shards(self.evaluation_config, num_workers=0)

            # Dataset should be in form of one episode per row. in case of bandits each
            # row is just one time step. To make the computation more efficient later
            # we remove the time dimension here.
            parallelism = self.evaluation_config.evaluation_num_workers or 1
            batch_size = max(ds.count() // parallelism, 1)
            self.evaluation_dataset = ds.map_batches(
                remove_time_dim, batch_size=batch_size
            )
            logger.info("Evaluation dataset created")

        self.reward_estimators: Dict[str, OffPolicyEstimator] = {}
        ope_types = {
            "is": ImportanceSampling,
            "wis": WeightedImportanceSampling,
            "dm": DirectMethod,
            "dr": DoublyRobust,
        }
        for name, method_config in self.config.off_policy_estimation_methods.items():
            method_type = method_config.pop("type")
            if method_type in ope_types:
                deprecation_warning(
                    old=method_type,
                    new=str(ope_types[method_type]),
                    error=True,
                )
                method_type = ope_types[method_type]
            elif isinstance(method_type, str):
                logger.log(0, "Trying to import from string: " + method_type)
                mod, obj = method_type.rsplit(".", 1)
                mod = importlib.import_module(mod)
                method_type = getattr(mod, obj)
            if isinstance(method_type, type) and issubclass(
                    method_type, OfflineEvaluator
            ):
                # TODO(kourosh) : Add an integration test for all these
                # offline evaluators.
                policy = self.get_policy()
                if issubclass(method_type, OffPolicyEstimator):
                    method_config["gamma"] = self.config.gamma
                self.reward_estimators[name] = method_type(policy, **method_config)
            else:
                raise ValueError(
                    f"Unknown off_policy_estimation type: {method_type}! Must be "
                    "either a class path or a sub-class of ray.rllib."
                    "offline.offline_evaluator::OfflineEvaluator"
                )
            # TODO (Rohan138): Refactor this and remove deprecated methods
            # Need to add back method_type in case Algorithm is restored from checkpoint
            method_config["type"] = method_type

        self.learner_group = None
        if self.config._enable_learner_api:
            # TODO (Kourosh): This is an interim solution where policies and modules
            # co-exist. In this world we have both policy_map and MARLModule that need
            # to be consistent with one another. To make a consistent parity between
            # the two we need to loop through the policy modules and create a simple
            # MARLModule from the RLModule within each policy.
            local_worker = self.workers.local_worker()
            module_spec = local_worker.marl_module_spec
            learner_group_config = self.config.get_learner_group_config(module_spec)
            self.learner_group = learner_group_config.build()

            # sync the weights from local rollout worker to trainers
            weights = local_worker.get_weights()
            self.learner_group.set_weights(weights)

        # Run `on_algorithm_init` callback after initialization is done.
        self.callbacks.on_algorithm_init(algorithm=self)

        # Create extra aggregation workers and assign each rollout worker to
        # one of them.
        self.batches_to_place_on_learner = []
        self.batch_being_built = []
        if self.config.num_aggregation_workers > 0:
            # This spawns `num_aggregation_workers` actors that aggregate
            # experiences coming from RolloutWorkers in parallel. We force
            # colocation on the same node (localhost) to maximize data bandwidth
            # between them and the learner.
            localhost = platform.node()
            assert localhost != "", (
                "ERROR: Cannot determine local node name! "
                "`platform.node()` returned empty string."
            )
            all_co_located = create_colocated_actors(
                actor_specs=[
                    # (class, args, kwargs={}, count=1)
                    (
                        AggregatorWorker,
                        [
                            self.config,
                        ],
                        {},
                        self.config.num_aggregation_workers,
                    )
                ],
                node=localhost,
            )
            aggregator_workers = [
                actor for actor_groups in all_co_located for actor in actor_groups
            ]
            self._aggregator_actor_manager = FaultTolerantActorManager(
                aggregator_workers,
                max_remote_requests_in_flight_per_actor=(
                    self.config.max_requests_in_flight_per_aggregator_worker
                ),
            )
            self._timeout_s_aggregator_manager = (
                self.config.timeout_s_aggregator_manager
            )
        else:
            # Create our local mixin buffer if the num of aggregation workers is 0.
            self.local_mixin_buffer = MixInMultiAgentReplayBuffer(
                capacity=(
                    self.config.replay_buffer_num_slots
                    if self.config.replay_buffer_num_slots > 0
                    else 1
                ),
                replay_ratio=self.config.get_replay_ratio(),
                replay_mode=ReplayMode.LOCKSTEP,
            )
            self._aggregator_actor_manager = None

        # This variable is used to keep track of the statistics from the most recent
        # update of the learner group
        self._results = {}
        self._timeout_s_sampler_manager = self.config.timeout_s_sampler_manager

        if not self.config._enable_learner_api:
            # Create and start the learner thread.
            self._learner_thread = make_learner_thread(
                self.workers.local_worker(), self.config
            )
            self._learner_thread.start()

        # TODO(avnishn):
        # this attribute isn't used anywhere else in the code. I think we can safely
        # delete it.
        if not self.config._enable_rl_module_api:
            self.update_kl = UpdateKL(self.workers)


class WorkerSetWithSpectator(WorkerSet):

    def __init__(
            self,
            *,
            env_creator: Optional[EnvCreator] = None,
            validate_env: Optional[Callable[[EnvType], None]] = None,
            default_policy_class: Optional[Type[Policy]] = None,
            config: Optional["AlgorithmConfig"] = None,
            num_workers: int = 0,
            local_worker: bool = True,
            logdir: Optional[str] = None,
            _setup: bool = True,
            # deprecated args.
            policy_class=DEPRECATED_VALUE,
            trainer_config=DEPRECATED_VALUE,
    ):
        super().__init__(env_creator=env_creator, validate_env=validate_env, default_policy_class=default_policy_class,
                         config=config, num_workers=num_workers, local_worker=local_worker, logdir=logdir,
                         _setup=_setup, policy_class=policy_class, trainer_config=trainer_config)

    def _make_worker(
        self,
        *,
        cls: Callable,
        env_creator: EnvCreator,
        validate_env: Optional[Callable[[EnvType], None]],
        worker_index: int,
        num_workers: int,
        recreated_worker: bool = False,
        config: "AlgorithmConfig",
        spaces: Optional[
            Dict[PolicyID, Tuple[gym.spaces.Space, gym.spaces.Space]]
        ] = None,
    ) -> Union[RolloutWorker, ActorHandle]:

        # TODO : add config entry for renderer idx and if we want to render
        if worker_index == 16:
            cls = SpectateWorker
            policy_cls = self._policy_class
        elif worker_index == 0:
            config = config.copy()
            config.policy_map_capacity = 100
            print("CHECK ME", config.policy_map_capacity)
            policy_cls = partial(self._policy_class, learner_bound=True)
        else:
            policy_cls = self._policy_class

        worker = cls(
            env_creator=env_creator,
            validate_env=validate_env,
            default_policy_class=policy_cls,
            config=config,
            worker_index=worker_index,
            num_workers=num_workers,
            recreated_worker=recreated_worker,
            log_dir=self._logdir,
            spaces=spaces,
            dataset_shards=self._ds_shards,
        )

        return worker




# TODO : we simply pass the custom class via config.worker_cls
class SpectateWorker(RolloutWorker):

    def __init__(
        self,
        *,
        env_creator: EnvCreator,
        validate_env: Optional[Callable[[EnvType, EnvContext], None]] = None,
        config: Optional["AlgorithmConfig"] = None,
        worker_index: int = 0,
        num_workers: Optional[int] = None,
        recreated_worker: bool = False,
        log_dir: Optional[str] = None,
        spaces: Optional[Dict[PolicyID, Tuple[Space, Space]]] = None,
        default_policy_class: Optional[Type[Policy]] = None,
        dataset_shards: Optional[List[ray.data.dataset.Dataset]] = None,
        # Deprecated: This is all specified in `config` anyways.
        policy_config=DEPRECATED_VALUE,
        input_creator=DEPRECATED_VALUE,
        output_creator=DEPRECATED_VALUE,
        rollout_fragment_length=DEPRECATED_VALUE,
        count_steps_by=DEPRECATED_VALUE,
        batch_mode=DEPRECATED_VALUE,
        episode_horizon=DEPRECATED_VALUE,
        preprocessor_pref=DEPRECATED_VALUE,
        sample_async=DEPRECATED_VALUE,
        compress_observations=DEPRECATED_VALUE,
        num_envs=DEPRECATED_VALUE,
        observation_fn=DEPRECATED_VALUE,
        clip_rewards=DEPRECATED_VALUE,
        normalize_actions=DEPRECATED_VALUE,
        clip_actions=DEPRECATED_VALUE,
        env_config=DEPRECATED_VALUE,
        model_config=DEPRECATED_VALUE,
        remote_worker_envs=DEPRECATED_VALUE,
        remote_env_batch_wait_ms=DEPRECATED_VALUE,
        soft_horizon=DEPRECATED_VALUE,
        no_done_at_end=DEPRECATED_VALUE,
        fake_sampler=DEPRECATED_VALUE,
        seed=DEPRECATED_VALUE,
        log_level=DEPRECATED_VALUE,
        callbacks=DEPRECATED_VALUE,
        disable_env_checking=DEPRECATED_VALUE,
        policy_spec=DEPRECATED_VALUE,
        policy_mapping_fn=DEPRECATED_VALUE,
        policies_to_train=DEPRECATED_VALUE,
        extra_python_environs=DEPRECATED_VALUE,
        policy=DEPRECATED_VALUE,
        tf_session_creator=DEPRECATED_VALUE,  # Use config.tf_session_options instead.
    ):
        """Initializes a RolloutWorker instance.

        Args:
            env_creator: Function that returns a gym.Env given an EnvContext
                wrapped configuration.
            validate_env: Optional callable to validate the generated
                environment (only on worker=0).
            worker_index: For remote workers, this should be set to a
                non-zero and unique value. This index is passed to created envs
                through EnvContext so that envs can be configured per worker.
            recreated_worker: Whether this worker is a recreated one. Workers are
                recreated by an Algorithm (via WorkerSet) in case
                `recreate_failed_workers=True` and one of the original workers (or an
                already recreated one) has failed. They don't differ from original
                workers other than the value of this flag (`self.recreated_worker`).
            log_dir: Directory where logs can be placed.
            spaces: An optional space dict mapping policy IDs
                to (obs_space, action_space)-tuples. This is used in case no
                Env is created on this RolloutWorker.
        """
        # Deprecated args.
        if policy != DEPRECATED_VALUE:
            deprecation_warning("policy", "policy_spec", error=True)
        if policy_spec != DEPRECATED_VALUE:
            deprecation_warning(
                "policy_spec",
                "RolloutWorker(default_policy_class=...)",
                error=True,
            )
        if policy_config != DEPRECATED_VALUE:
            deprecation_warning("policy_config", "config", error=True)
        if input_creator != DEPRECATED_VALUE:
            deprecation_warning(
                "input_creator",
                "config.offline_data(input_=..)",
                error=True,
            )
        if output_creator != DEPRECATED_VALUE:
            deprecation_warning(
                "output_creator",
                "config.offline_data(output=..)",
                error=True,
            )
        if rollout_fragment_length != DEPRECATED_VALUE:
            deprecation_warning(
                "rollout_fragment_length",
                "config.rollouts(rollout_fragment_length=..)",
                error=True,
            )
        if count_steps_by != DEPRECATED_VALUE:
            deprecation_warning(
                "count_steps_by", "config.multi_agent(count_steps_by=..)", error=True
            )
        if batch_mode != DEPRECATED_VALUE:
            deprecation_warning(
                "batch_mode", "config.rollouts(batch_mode=..)", error=True
            )
        if episode_horizon != DEPRECATED_VALUE:
            deprecation_warning("episode_horizon", error=True)
        if preprocessor_pref != DEPRECATED_VALUE:
            deprecation_warning(
                "preprocessor_pref", "config.rollouts(preprocessor_pref=..)", error=True
            )
        if sample_async != DEPRECATED_VALUE:
            deprecation_warning(
                "sample_async", "config.rollouts(sample_async=..)", error=True
            )
        if compress_observations != DEPRECATED_VALUE:
            deprecation_warning(
                "compress_observations",
                "config.rollouts(compress_observations=..)",
                error=True,
            )
        if num_envs != DEPRECATED_VALUE:
            deprecation_warning(
                "num_envs", "config.rollouts(num_envs_per_worker=..)", error=True
            )
        if observation_fn != DEPRECATED_VALUE:
            deprecation_warning(
                "observation_fn", "config.multi_agent(observation_fn=..)", error=True
            )
        if clip_rewards != DEPRECATED_VALUE:
            deprecation_warning(
                "clip_rewards", "config.environment(clip_rewards=..)", error=True
            )
        if normalize_actions != DEPRECATED_VALUE:
            deprecation_warning(
                "normalize_actions",
                "config.environment(normalize_actions=..)",
                error=True,
            )
        if clip_actions != DEPRECATED_VALUE:
            deprecation_warning(
                "clip_actions", "config.environment(clip_actions=..)", error=True
            )
        if env_config != DEPRECATED_VALUE:
            deprecation_warning(
                "env_config", "config.environment(env_config=..)", error=True
            )
        if model_config != DEPRECATED_VALUE:
            deprecation_warning("model_config", "config.training(model=..)", error=True)
        if remote_worker_envs != DEPRECATED_VALUE:
            deprecation_warning(
                "remote_worker_envs",
                "config.rollouts(remote_worker_envs=..)",
                error=True,
            )
        if remote_env_batch_wait_ms != DEPRECATED_VALUE:
            deprecation_warning(
                "remote_env_batch_wait_ms",
                "config.rollouts(remote_env_batch_wait_ms=..)",
                error=True,
            )
        if soft_horizon != DEPRECATED_VALUE:
            deprecation_warning("soft_horizon", error=True)
        if no_done_at_end != DEPRECATED_VALUE:
            deprecation_warning("no_done_at_end", error=True)
        if fake_sampler != DEPRECATED_VALUE:
            deprecation_warning(
                "fake_sampler", "config.rollouts(fake_sampler=..)", error=True
            )
        if seed != DEPRECATED_VALUE:
            deprecation_warning("seed", "config.debugging(seed=..)", error=True)
        if log_level != DEPRECATED_VALUE:
            deprecation_warning(
                "log_level", "config.debugging(log_level=..)", error=True
            )
        if callbacks != DEPRECATED_VALUE:
            deprecation_warning(
                "callbacks", "config.callbacks([DefaultCallbacks subclass])", error=True
            )
        if disable_env_checking != DEPRECATED_VALUE:
            deprecation_warning(
                "disable_env_checking",
                "config.environment(disable_env_checking=..)",
                error=True,
            )
        if policy_mapping_fn != DEPRECATED_VALUE:
            deprecation_warning(
                "policy_mapping_fn",
                "config.multi_agent(policy_mapping_fn=..)",
                error=True,
            )
        if policies_to_train != DEPRECATED_VALUE:
            deprecation_warning(
                "policies_to_train",
                "config.multi_agent(policies_to_train=..)",
                error=True,
            )
        if extra_python_environs != DEPRECATED_VALUE:
            deprecation_warning(
                "extra_python_environs",
                "config.python_environment(extra_python_environs_for_driver=.., "
                "extra_python_environs_for_worker=..)",
                error=True,
            )
        if tf_session_creator != DEPRECATED_VALUE:
            deprecation_warning(
                old="RolloutWorker(.., tf_session_creator=.., ..)",
                new="RolloutWorker(.., policy_config={tf_session_options=..}, ..)",
                error=False,
            )
        self.last_set_weights_params = None

        self._original_kwargs: dict = locals().copy()
        del self._original_kwargs["self"]

        global _global_worker
        _global_worker = self

        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

        # Default config needed?
        if config is None or isinstance(config, dict):
            config = AlgorithmConfig().update_from_dict(config or {})
        # Freeze config, so no one else can alter it from here on.
        config.freeze()

        # Set extra python env variables before calling super constructor.
        if config.extra_python_environs_for_driver and worker_index == 0:
            for key, value in config.extra_python_environs_for_driver.items():
                os.environ[key] = str(value)
        elif config.extra_python_environs_for_worker and worker_index > 0:
            for key, value in config.extra_python_environs_for_worker.items():
                os.environ[key] = str(value)

        def gen_rollouts():
            while True:
                yield self.sample()

        ParallelIteratorWorker.__init__(self, gen_rollouts, False)

        self.config = config

        self.config._is_frozen = False
        self.config.num_envs_per_worker = 1 if worker_index == 16 else self.config.num_envs_per_worker
        print('How many workers ?:', self.config.num_envs_per_worker)
        self.config.freeze()

        # TODO: Remove this backward compatibility.
        #  This property (old-style python config dict) should no longer be used!
        self.policy_config = config.to_dict()

        self.num_workers = (
            num_workers if num_workers is not None else self.config.num_rollout_workers
        )
        # In case we are reading from distributed datasets, store the shards here
        # and pick our shard by our worker-index.
        self._ds_shards = dataset_shards
        self.worker_index: int = worker_index

        # Lock to be able to lock this entire worker
        # (via `self.lock()` and `self.unlock()`).
        # This might be crucial to prevent a race condition in case
        # `config.policy_states_are_swappable=True` and you are using an Algorithm
        # with a learner thread. In this case, the thread might update a policy
        # that is being swapped (during the update) by the Algorithm's
        # training_step's `RolloutWorker.get_weights()` call (to sync back the
        # new weights to all remote workers).
        self._lock = threading.Lock()

        if (
            tf1
            and (config.framework_str == "tf2" or config.enable_tf1_exec_eagerly)
            # This eager check is necessary for certain all-framework tests
            # that use tf's eager_mode() context generator.
            and not tf1.executing_eagerly()
        ):
            tf1.enable_eager_execution()

        if self.config.log_level:
            logging.getLogger("ray.rllib").setLevel(self.config.log_level)

        if self.worker_index > 1:
            disable_log_once_globally()  # only need 1 worker to log
        elif self.config.log_level == "DEBUG":
            enable_periodic_logging()

        env_context = EnvContext(
            self.config.env_config,
            worker_index=self.worker_index,
            vector_index=0,
            num_workers=self.num_workers,
            remote=self.config.remote_worker_envs,
            recreated_worker=recreated_worker,
        )
        self.env_context = env_context
        self.config: AlgorithmConfig = config
        self.callbacks: DefaultCallbacks = self.config.callbacks_class()
        self.recreated_worker: bool = recreated_worker

        # Setup current policy_mapping_fn. Start with the one from the config, which
        # might be None in older checkpoints (nowadays AlgorithmConfig has a proper
        # default for this); Need to cover this situation via the backup lambda here.
        self.policy_mapping_fn = (
            lambda agent_id, episode, worker, **kw: DEFAULT_POLICY_ID
        )
        self.set_policy_mapping_fn(self.config.policy_mapping_fn)

        self.env_creator: EnvCreator = env_creator
        # Resolve possible auto-fragment length.
        configured_rollout_fragment_length = self.config.get_rollout_fragment_length(
            worker_index=self.worker_index
        )
        self.total_rollout_fragment_length: int = (
            configured_rollout_fragment_length * self.config.num_envs_per_worker
        )
        self.preprocessing_enabled: bool = not config._disable_preprocessor_api
        self.last_batch: Optional[SampleBatchType] = None
        self.global_vars: dict = {
            # TODO(sven): Make this per-policy!
            "timestep": 0,
            # Counter for performed gradient updates per policy in `self.policy_map`.
            # Allows for compiling metrics on the off-policy'ness of an update given
            # that the number of gradient updates of the sampling policies are known
            # to the learner (and can be compared to the learner version of the same
            # policy).
            "num_grad_updates_per_policy": defaultdict(int),
        }

        # If seed is provided, add worker index to it and 10k iff evaluation worker.
        self.seed = (
            None
            if self.config.seed is None
            else self.config.seed
            + self.worker_index
            + self.config.in_evaluation * 10000
        )

        # Update the global seed for numpy/random/tf-eager/torch if we are not
        # the local worker, otherwise, this was already done in the Algorithm
        # object itself.
        if self.worker_index > 0:
            update_global_seed_if_necessary(self.config.framework_str, self.seed)

        # A single environment provided by the user (via config.env). This may
        # also remain None.
        # 1) Create the env using the user provided env_creator. This may
        #    return a gym.Env (incl. MultiAgentEnv), an already vectorized
        #    VectorEnv, BaseEnv, ExternalEnv, or an ActorHandle (remote env).
        # 2) Wrap - if applicable - with Atari/rendering wrappers.
        # 3) Seed the env, if necessary.
        # 4) Vectorize the existing single env by creating more clones of
        #    this env and wrapping it with the RLlib BaseEnv class.
        self.env = self.make_sub_env_fn = None

        # Create a (single) env for this worker.
        if not (
            self.worker_index == 0
            and self.num_workers > 0
            and not self.config.create_env_on_local_worker
        ):
            # Run the `env_creator` function passing the EnvContext.
            self.env = env_creator(copy.deepcopy(self.env_context))

        clip_rewards = self.config.clip_rewards

        if self.env is not None:
            # Validate environment (general validation function).
            if not self.config.disable_env_checking:
                check_env(self.env)
            # Custom validation function given, typically a function attribute of the
            # algorithm trainer.
            if validate_env is not None:
                validate_env(self.env, self.env_context)

            # We can't auto-wrap a BaseEnv.
            if isinstance(self.env, (BaseEnv, ray.actor.ActorHandle)):

                def wrap(env):
                    return env

            # Atari type env and "deepmind" preprocessor pref.
            elif is_atari(self.env) and self.config.preprocessor_pref == "deepmind":
                # Deepmind wrappers already handle all preprocessing.
                self.preprocessing_enabled = False

                # If clip_rewards not explicitly set to False, switch it
                # on here (clip between -1.0 and 1.0).
                if self.config.clip_rewards is None:
                    clip_rewards = True

                # Framestacking is used.
                use_framestack = self.config.model.get("framestack") is True

                def wrap(env):
                    env = wrap_deepmind(
                        env,
                        dim=self.config.model.get("dim"),
                        framestack=use_framestack,
                        noframeskip=self.config.env_config.get("frameskip", 0) == 1,
                    )
                    return env

            elif self.config.preprocessor_pref is None:
                # Only turn off preprocessing
                self.preprocessing_enabled = False

                def wrap(env):
                    return env

            else:

                def wrap(env):
                    return env

            # Wrap env through the correct wrapper.
            self.env: EnvType = wrap(self.env)
            # Ideally, we would use the same make_sub_env() function below
            # to create self.env, but wrap(env) and self.env has a cyclic
            # dependency on each other right now, so we would settle on
            # duplicating the random seed setting logic for now.
            _update_env_seed_if_necessary(self.env, self.seed, self.worker_index, 0)
            # Call custom callback function `on_sub_environment_created`.
            self.callbacks.on_sub_environment_created(
                worker=self,
                sub_environment=self.env,
                env_context=self.env_context,
            )

            self.make_sub_env_fn = self._get_make_sub_env_fn(
                env_creator, env_context, validate_env, wrap, self.seed
            )

        self.spaces = spaces
        self.default_policy_class = default_policy_class
        self.policy_dict, self.is_policy_to_train = self.config.get_multi_agent_setup(
            env=self.env,
            spaces=self.spaces,
            default_policy_class=self.default_policy_class,
        )

        self.policy_map: Optional[PolicyMap] = None
        # TODO(jungong) : clean up after non-connector env_runner is fully deprecated.
        self.preprocessors: Dict[PolicyID, Preprocessor] = None

        # Check available number of GPUs.
        num_gpus = (
            self.config.num_gpus
            if self.worker_index == 0
            else self.config.num_gpus_per_worker
        )

        # This is only for the old API where local_worker was responsible for learning
        if not self.config._enable_learner_api:
            # Error if we don't find enough GPUs.
            if (
                ray.is_initialized()
                and ray._private.worker._mode() != ray._private.worker.LOCAL_MODE
                and not config._fake_gpus
            ):

                devices = []
                if self.config.framework_str in ["tf2", "tf"]:
                    devices = get_tf_gpu_devices()

                if len(devices) < num_gpus:
                    raise RuntimeError(
                        ERR_MSG_NO_GPUS.format(len(devices), devices)
                        + HOWTO_CHANGE_CONFIG
                    )
            # Warn, if running in local-mode and actual GPUs (not faked) are
            # requested.
            elif (
                ray.is_initialized()
                and ray._private.worker._mode() == ray._private.worker.LOCAL_MODE
                and num_gpus > 0
                and not self.config._fake_gpus
            ):
                logger.warning(
                    "You are running ray with `local_mode=True`, but have "
                    f"configured {num_gpus} GPUs to be used! In local mode, "
                    f"Policies are placed on the CPU and the `num_gpus` setting "
                    f"is ignored."
                )

        self.filters: Dict[PolicyID, Filter] = defaultdict(NoFilter)

        # if RLModule API is enabled, marl_module_spec holds the specs of the RLModules
        self.marl_module_spec = None
        self._update_policy_map(policy_dict=self.policy_dict)

        # Update Policy's view requirements from Model, only if Policy directly
        # inherited from base `Policy` class. At this point here, the Policy
        # must have it's Model (if any) defined and ready to output an initial
        # state.
        for pol in self.policy_map.values():
            if not pol._model_init_state_automatically_added:
                pol._update_model_view_requirements_from_init_state()

        self.multiagent: bool = set(self.policy_map.keys()) != {DEFAULT_POLICY_ID}
        if self.multiagent and self.env is not None:
            if not isinstance(
                self.env,
                (BaseEnv, ExternalMultiAgentEnv, MultiAgentEnv, ray.actor.ActorHandle),
            ):
                raise ValueError(
                    f"Have multiple policies {self.policy_map}, but the "
                    f"env {self.env} is not a subclass of BaseEnv, "
                    f"MultiAgentEnv, ActorHandle, or ExternalMultiAgentEnv!"
                )

        if self.worker_index == 0:
            logger.info("Built filter map: {}".format(self.filters))

        # This RolloutWorker has no env.
        if self.env is None:
            self.async_env = None
        # Use a custom env-vectorizer and call it providing self.env.
        elif "custom_vector_env" in self.config:
            self.async_env = self.config.custom_vector_env(self.env)
        # Default: Vectorize self.env via the make_sub_env function. This adds
        # further clones of self.env and creates a RLlib BaseEnv (which is
        # vectorized under the hood).
        else:
            # Always use vector env for consistency even if num_envs_per_worker=1.
            self.async_env: BaseEnv = convert_to_base_env(
                self.env,
                make_env=self.make_sub_env_fn,
                num_envs=self.config.num_envs_per_worker,
                remote_envs=self.config.remote_worker_envs,
                remote_env_batch_wait_ms=self.config.remote_env_batch_wait_ms,
                worker=self,
                restart_failed_sub_environments=(
                    self.config.restart_failed_sub_environments
                ),
            )

        # `truncate_episodes`: Allow a batch to contain more than one episode
        # (fragments) and always make the batch `rollout_fragment_length`
        # long.
        rollout_fragment_length_for_sampler = configured_rollout_fragment_length
        if self.config.batch_mode == "truncate_episodes":
            pack = True
        # `complete_episodes`: Never cut episodes and sampler will return
        # exactly one (complete) episode per poll.
        else:
            assert self.config.batch_mode == "complete_episodes"
            rollout_fragment_length_for_sampler = float("inf")
            pack = False

        # Create the IOContext for this worker.
        self.io_context: IOContext = IOContext(
            log_dir, self.config, self.worker_index, self
        )

        render = False
        if self.config.render_env is True and (
            self.num_workers == 0 or self.worker_index == 1
        ):
            render = True

        if self.env is None:
            self.sampler = None
        elif worker_index == 16:
            self.sampler = SpectateSampler(
                worker=self,
                env=self.async_env,
                clip_rewards=clip_rewards,
                rollout_fragment_length=rollout_fragment_length_for_sampler,
                count_steps_by=self.config.count_steps_by,
                callbacks=self.callbacks,
                multiple_episodes_in_batch=pack,
                normalize_actions=self.config.normalize_actions,
                clip_actions=self.config.clip_actions,
                observation_fn=self.config.observation_fn,
                sample_collector_class=self.config.sample_collector,
                render=render,
            )
            # Start the Sampler thread.
            self.sampler.start()

        elif self.config.sample_async:
            self.sampler = AsyncSampler(
                worker=self,
                env=self.async_env,
                clip_rewards=clip_rewards,
                rollout_fragment_length=rollout_fragment_length_for_sampler,
                count_steps_by=self.config.count_steps_by,
                callbacks=self.callbacks,
                multiple_episodes_in_batch=pack,
                normalize_actions=self.config.normalize_actions,
                clip_actions=self.config.clip_actions,
                observation_fn=self.config.observation_fn,
                sample_collector_class=self.config.sample_collector,
                render=render,
            )
            # Start the Sampler thread.
            self.sampler.start()
        else:
            self.sampler = SyncSampler(
                worker=self,
                env=self.async_env,
                clip_rewards=clip_rewards,
                rollout_fragment_length=rollout_fragment_length_for_sampler,
                count_steps_by=self.config.count_steps_by,
                callbacks=self.callbacks,
                multiple_episodes_in_batch=pack,
                normalize_actions=self.config.normalize_actions,
                clip_actions=self.config.clip_actions,
                observation_fn=self.config.observation_fn,
                sample_collector_class=self.config.sample_collector,
                render=render,
            )

        self.input_reader: InputReader = self._get_input_creator_from_config()(
            self.io_context
        )
        self.output_writer: OutputWriter = self._get_output_creator_from_config()(
            self.io_context
        )

        # The current weights sequence number (version). May remain None for when
        # not tracking weights versions.
        self.weights_seq_no: Optional[int] = None

        logger.debug(
            "Created rollout worker with env {} ({}), policies {}".format(
                self.async_env, self.env, self.policy_map
            )
        )

    def apply_weights(self):
        if self.last_set_weights_params is not None:
            args, kwargs = self.last_set_weights_params
            super().set_weights(*args, **kwargs)
            self.last_set_weights_params = None


    def set_weights(
            self,
            *args,
            **kwargs
    ) -> None:

        self.last_set_weights_params = (args, kwargs)


    # def sample(self):
    #     if self.worker_index == 16:
    #         all_batches = []
    #         while True:
    #             # Continue sampling until the episode ends
    #             # we should only have one env on this worker
    #             batch = super().sample()
    #             all_batches.append(batch)
    #             print("SAMPLE", batch["dones"], len(batch["dones"]))
    #             if np.any(batch["dones"]):
    #                 print("new episode")
    #                 break
    #
    #         return concat_samples(all_batches)
    #     else:
    #         return super().sample()


class SpectateSampler(AsyncSampler):

    def __init__(
        self,
        *,
        worker: "RolloutWorker",
        env: BaseEnv,
        clip_rewards: Union[bool, float],
        rollout_fragment_length: int,
        count_steps_by: str = "env_steps",
        callbacks: "DefaultCallbacks",
        multiple_episodes_in_batch: bool = False,
        normalize_actions: bool = True,
        clip_actions: bool = False,
        observation_fn: Optional["ObservationFunction"] = None,
        sample_collector_class: Optional[Type[SampleCollector]] = None,
        render: bool = False,
        blackhole_outputs: bool = False,
        # Obsolete.
        policies=None,
        policy_mapping_fn=None,
        preprocessors=None,
        obs_filters=None,
        tf_sess=None,
        no_done_at_end=DEPRECATED_VALUE,
        horizon=DEPRECATED_VALUE,
        soft_horizon=DEPRECATED_VALUE,
    ):
        """Initializes an AsyncSampler instance.

        Args:
            worker: The RolloutWorker that will use this Sampler for sampling.
            env: Any Env object. Will be converted into an RLlib BaseEnv.
            clip_rewards: True for +/-1.0 clipping,
                actual float value for +/- value clipping. False for no
                clipping.
            rollout_fragment_length: The length of a fragment to collect
                before building a SampleBatch from the data and resetting
                the SampleBatchBuilder object.
            count_steps_by: One of "env_steps" (default) or "agent_steps".
                Use "agent_steps", if you want rollout lengths to be counted
                by individual agent steps. In a multi-agent env,
                a single env_step contains one or more agent_steps, depending
                on how many agents are present at any given time in the
                ongoing episode.
            multiple_episodes_in_batch: Whether to pack multiple
                episodes into each batch. This guarantees batches will be
                exactly `rollout_fragment_length` in size.
            normalize_actions: Whether to normalize actions to the
                action space's bounds.
            clip_actions: Whether to clip actions according to the
                given action_space's bounds.
            blackhole_outputs: Whether to collect samples, but then
                not further process or store them (throw away all samples).
            observation_fn: Optional multi-agent observation func to use for
                preprocessing observations.
            sample_collector_class: An optional SampleCollector sub-class to
                use to collect, store, and retrieve environment-, model-,
                and sampler data.
            render: Whether to try to render the environment after each step.
        """
        # All of the following arguments are deprecated. They will instead be
        # provided via the passed in `worker` arg, e.g. `worker.policy_map`.
        if log_once("deprecated_async_sampler_args"):
            if policies is not None:
                deprecation_warning(old="policies")
            if policy_mapping_fn is not None:
                deprecation_warning(old="policy_mapping_fn")
            if preprocessors is not None:
                deprecation_warning(old="preprocessors")
            if obs_filters is not None:
                deprecation_warning(old="obs_filters")
            if tf_sess is not None:
                deprecation_warning(old="tf_sess")
            if horizon != DEPRECATED_VALUE:
                deprecation_warning(old="horizon", error=True)
            if soft_horizon != DEPRECATED_VALUE:
                deprecation_warning(old="soft_horizon", error=True)
            if no_done_at_end != DEPRECATED_VALUE:
                deprecation_warning(old="no_done_at_end", error=True)

        self.worker = worker

        for _, f in worker.filters.items():
            assert getattr(
                f, "is_concurrent", False
            ), "Observation Filter must support concurrent updates."

        self.base_env = convert_to_base_env(env)
        threading.Thread.__init__(self)
        self.queue = queue.Queue(64)
        self.extra_batches = queue.Queue()
        self.metrics_queue = queue.Queue()
        self.rollout_fragment_length = rollout_fragment_length
        self.clip_rewards = clip_rewards
        self.daemon = True
        self.multiple_episodes_in_batch = multiple_episodes_in_batch
        self.callbacks = callbacks
        self.normalize_actions = normalize_actions
        self.clip_actions = clip_actions
        self.blackhole_outputs = blackhole_outputs
        self.perf_stats = _PerfStats(
            ema_coef=worker.config.sampler_perf_stats_ema_coef,
        )
        self.shutdown = False
        self.observation_fn = observation_fn
        self.render = render
        if not sample_collector_class:
            sample_collector_class = SimpleListCollector
        self.sample_collector = sample_collector_class(
            self.worker.policy_map,
            self.clip_rewards,
            self.callbacks,
            self.multiple_episodes_in_batch,
            self.rollout_fragment_length,
            count_steps_by=count_steps_by,
        )
        self.count_steps_by = count_steps_by



    @override(AsyncSampler)
    def get_data(self) -> SampleBatchType:
        if not self.is_alive():
            raise RuntimeError("Sampling thread has died")
        try:
            # todo FIND OUT way for elo updates to be smooth without having this slow worker blocking everything.
            rollout = self.queue.get(timeout=10)
        except Exception:
            rollout = SampleBatch()

        # Propagate errors.
        if isinstance(rollout, BaseException):
            raise rollout

        return rollout

    @override(AsyncSampler)
    def get_metrics(self) -> List[RolloutMetrics]:
        completed = []
        while True:
            try:
                completed.append(
                    self.metrics_queue.get_nowait()._replace(
                        perf_stats=self.perf_stats.get()
                    )
                )
            except queue.Empty:
                break
        if len(completed) > 0:
            print("got some metrics [spectator]", len(completed))
        else:
            print("no metrics [spectator].")
        return completed











def combine_dicts(dict_list, keys):
    combined = {}

    for key in keys:
        gathered = [d[key] for d in dict_list if key in d]
        if len(gathered) > 0:
            combined[key] = np.sum(np.abs(gathered))
            #combined[key] = np.sum(np.maximum(gathered, 0.))
    return combined

def compute_winrate_per_char(dict_list):
    combined = {}
    char = "char"
    for d in dict_list:
        if char in d:
            combined[str(d[char]) + "_score"] = np.float32(d["win_rewards"] > 0.5)

    return combined

def fast_groupby(x):
    diff = np.diff(x)
    return np.concatenate(([x[0]], x[1:][diff != 0]))

def wrap_slice_set(arr, start, stop, values):
    length = arr.shape[0]
    num_values = len(values)
    diff = stop - start
    if diff > 0:
        arr[start:min(start + diff, length)] = values[:min(diff, num_values)]
    else:
        arr[start:] = values[:length-start]
        arr[:stop] = values[length-start:]

def compute_entropy_rewards(
        visited_states, # should be an array of floats because it is a box
        state_history: np.ndarray,
        full_counts,
        dicarded
):
    visited_states = np.int32(visited_states)
    data_amount = len(state_history)

    full_counts[:] = 0
    unique_elements, counts = np.unique(state_history, return_counts=True)
    full_counts[unique_elements] += counts
    probabilities = full_counts / data_amount

    #max_ = np.log(data_amount)
    lower_probablity_threshold = 1e-4
    probabilities_clipped = np.clip(probabilities + (1. - lower_probablity_threshold), 0., 1.)
    nll = - np.log(probabilities_clipped)

    action_state_entropy = - np.sum(probabilities * np.log(probabilities+1e-8))

    # WALK_SLOW = 0x0f
    # WALK_MIDDLE = 0x10
    # WALK_FAST = 0x11
    nll[dicarded] = 0.

    entropy_rewards = np.square(nll[visited_states] * 8000)

    return entropy_rewards, action_state_entropy



def filter_last_damage_before_death(damage_array, death_array):
    relevant_damage_idx = np.zeros(damage_array.shape, dtype=np.float32)
    last_damage_indices = np.where(damage_array)[0]  # Get indices where damage was taken

    if len(last_damage_indices) > 0:
        death_indices = np.where(death_array)[0]  # Get indices where death occurred

        if len(death_indices) > 0:
            last_damage_index = last_damage_indices[np.searchsorted(last_damage_indices, death_indices) - 1]
            relevant_damage_idx[last_damage_index] = 1.  # Set the last damage before death to 1

    return relevant_damage_idx


def retrieve_game_stats(infos):
    game_stats = []
    for info in infos:
        if "game_stats" in info:
            game_stats.append(info["game_stats"])
    if len(game_stats) > 0:
        gathered = {key: [d[key] for d in game_stats if key in d] for key in game_stats[0]}

        return gathered
    else:
        return None








class SSBMCallbacksPLUS(DefaultCallbacks):

    def on_episode_start(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        episode.user_data["button_dist"] = []
        episode.hist_data["button_dist"] = []
        episode.user_data["stick_m_dist"] = []
        episode.hist_data["stick_m_dist"] = []
        episode.user_data["stick_c_dist"] = []
        episode.hist_data["stick_c_dist"] = []

    def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )

        m, action_button, c = episode.last_action_for()
        episode.user_data["button_dist"].append(action_button)
        episode.user_data["stick_m_dist"].append(m)
        episode.user_data["stick_c_dist"].append(c)




    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        for d in ["button_dist", "stick_m_dist", 'stick_c_dist']:
            episode.hist_data[d] = episode.user_data[d]


class SSBMCallbacks(
    DefaultCallbacks
):
    def __init__(self, *args, negative_reward_scale=0.8,
                 killer_move_reward_scale=1.,
                 observation_manager: ObsBuilder, **kwargs):
        super().__init__(*args, **kwargs)
        self.om = observation_manager
        self.delay = self.om.config["obs"]["delay"]

        # We only care about the second player for now
        self.self_stock_idx = self.om.get_player_obs_idx("stock", 1)
        self.opp_stock_idx = self.om.get_player_obs_idx("stock", 2)
        self.percent_idx = self.om.get_player_obs_idx("percent", 2)
        self.action_state_idx = self.om.get_player_obs_idx("action", 1)

        self.self_pos_idx = self.om.get_player_obs_idx("position", 1)
        self.opp_pos_idx = self.om.get_player_obs_idx("position", 2)

        self.killer_move_reward_scale = killer_move_reward_scale
        self.negative_reward_scale = negative_reward_scale

        self.action_state_history_size = 9000*3
        self.action_state_history = defaultdict(lambda: np.zeros((self.action_state_history_size,), dtype=np.int32))
        self.idx = defaultdict(lambda: 0)
        self.counts = defaultdict(lambda: np.zeros((396,), dtype=np.int32))
        self.discarded_states = np.array([a.value for a in [
            Action.WALK_SLOW, Action.WALK_FAST, Action.WALK_MIDDLE,
            Action.TUMBLING, Action.TURNING, Action.TURNING_RUN, Action.GRABBED,
            Action.EDGE_TEETERING, Action.EDGE_TEETERING_START, Action.RUN_BRAKE,
            Action.EDGE_ATTACK_QUICK, Action.EDGE_HANGING, Action.EDGE_ATTACK_SLOW,
            Action.EDGE_GETUP_QUICK, Action.EDGE_JUMP_1_QUICK, Action.EDGE_JUMP_2_QUICK,
            Action.EDGE_JUMP_1_SLOW, Action.EDGE_JUMP_2_SLOW, Action.EDGE_GETUP_SLOW,
            Action.EDGE_ROLL_QUICK, Action.EDGE_ROLL_SLOW
        ]])
        self.wall_tech_states = np.array([a.value for a in [
            Action.WALL_TECH, Action.WALL_TECH_JUMP
        ]])

    @override(DefaultCallbacks)
    def on_postprocess_trajectory(
            self,
            *,
            worker: "RolloutWorker",
            episode: Episode,
            agent_id: AgentID,
            policy_id: PolicyID,
            policies: Dict[PolicyID, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
            **kwargs,
    ) -> None:

        # Compute stock rewards here

        # self_stock = postprocessed_batch[SampleBatch.OBS][:-self.delay, self.self_stock_idx]
        # opp_stock = postprocessed_batch[SampleBatch.OBS][self.delay:, self.opp_stock_idx]
        # opp_percent = postprocessed_batch[SampleBatch.OBS][self.delay:, self.percent_idx]
        # where_at_least_one_dead = np.where(np.logical_or(self_stock==0., opp_stock==0.))[0]
        # if len(where_at_least_one_dead) > 0:
        #     last_step = np.min(where_at_least_one_dead) + 1
        # else:
        #     last_step = len(self_stock)
        #
        # stock_deltas = np.diff(opp_stock[:last_step])
        # stock_deltas = np.logical_and(-0.26 < stock_deltas, stock_deltas < 0)
        #
        # # do not consider damages taken when near blastzones
        # percent_deltas = np.diff(opp_percent[:last_step]) > 1.5 * self.om.PERCENT_SCALE
        #
        # relevant_damage_idxs = filter_last_damage_before_death(percent_deltas, stock_deltas)
        #
        # episode.custom_metrics["irrelevant_deaths"] = np.sum(np.int8(stock_deltas)) - np.sum(relevant_damage_idxs)

        #print(np.sum(np.int8(stock_deltas)), np.sum(relevant_damage_idxs), list(stock_deltas), list(opp_stock),
        #      list(opp_percent))

        # give a reward if only this is not suicide and reward the corresponding attack (last damaging move)
        # TODO: This is not going to work for edge-hops
        #   -> keep rewarding for killing at the time of death
        # postprocessed_batch[SampleBatch.REWARDS][1:last_step] += relevant_damage_idxs * self.killer_move_reward_scale

        # TODO : investigate why sometimes the dict is empty even when no failure from the env.
        if "character" in postprocessed_batch[SampleBatch.INFOS][-1]:
            character = postprocessed_batch[SampleBatch.INFOS][-1]["character"]

            postprocessed_batch[SampleBatch.REWARDS][:] = \
                np.maximum(postprocessed_batch[SampleBatch.REWARDS], 0.) + \
                np.minimum(postprocessed_batch[SampleBatch.REWARDS], 0.) * self.negative_reward_scale

            action_states = postprocessed_batch[SampleBatch.OBS][:, self.action_state_idx]

            next_idx = self.idx[character] + len(action_states)
            wrap_slice_set(self.action_state_history[character], self.idx[character] % self.action_state_history_size,
                           next_idx % self.action_state_history_size, action_states)
            self.idx[character] = next_idx
            if self.idx[character] >= self.action_state_history_size:
                # More accurate to reward the transition preceding the observation, not the next one.
                # We are missing the last bit, but fine
                entropy_rewards, as_entropy = compute_entropy_rewards(action_states[1:], self.action_state_history[character],
                                                                      self.counts[character], self.discarded_states)
                episode.custom_metrics["action_state_rewards"] = np.sum(entropy_rewards)
                episode.custom_metrics["action_state_entropy"] = as_entropy

                postprocessed_batch[SampleBatch.REWARDS][:-1] += entropy_rewards
                # print(f"Episode total entropy rewards and entropy over {episode.length} timesteps: ",
                #       episode.custom_metrics["action_state_rewards"],
                #       as_entropy)

        super().on_postprocess_trajectory(
            worker=worker, episode=episode, agent_id=agent_id, policy_id=policy_id, policies=policies,
            postprocessed_batch=postprocessed_batch, original_batches=original_batches, **kwargs
        )

        all_infos = combine_dicts(postprocessed_batch[SampleBatch.INFOS], keys=[
            "distance_rewards", "energy_costs", "damage_rewards", "stock_rewards"
        ])
        for key in all_infos:
            episode.custom_metrics[key] = all_infos[key]

        episode.custom_metrics["Wall techs"] = np.sum(
            np.int16(np.isin(np.int32(postprocessed_batch[SampleBatch.OBS][:, self.action_state_idx]), self.wall_tech_states)
                    ))

        episode.custom_metrics["REWARDS"] = np.sum(postprocessed_batch[SampleBatch.REWARDS])

        game_stats = retrieve_game_stats(postprocessed_batch[SampleBatch.INFOS])
        if game_stats is not None:
            episode.hist_data[f"{policy_id}_outcome"] = game_stats["outcome"]
            episode.hist_data[f"{policy_id}_opponent"] = game_stats["opponent"]
            # TODO for relevant stats
            # ...

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:

        if isinstance(worker, SpectateWorker):
            # Utility to force weight updates at episode ends.
            worker.apply_weights()

    def on_episode_created(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        env_index: int,
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> None:
        # TODO : Could be used instead to pass options to the reset call.
        pass


class TraceMallocCallback(SSBMCallbacks):

    def __init__(self):
        super().__init__()

        tracemalloc.start(10)

    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode, env_index: Optional[int] = None, **kwargs) -> None:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        for stat in top_stats[:5]:
            count = stat.count
            size = stat.size

            trace = str(stat.traceback)

            episode.custom_metrics[f'tracemalloc/{trace}/size'] = size
            episode.custom_metrics[f'tracemalloc/{trace}/count'] = count

        process = psutil.Process(os.getpid())
        worker_rss = process.memory_info().rss
        worker_data = process.memory_info().data
        worker_vms = process.memory_info().vms
        episode.custom_metrics[f'tracemalloc/worker/rss'] = worker_rss
        episode.custom_metrics[f'tracemalloc/worker/data'] = worker_data
        episode.custom_metrics[f'tracemalloc/worker/vms'] = worker_vms
