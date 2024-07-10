import importlib
import logging
import platform
import queue
from collections import defaultdict
from pathlib import Path
from typing import Optional, Type, DefaultDict, Set, Dict, List

import ray
from ray.actor import ActorHandle
from ray.rllib import Policy, RolloutWorker
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.callbacks import MultiCallbacks
from ray.rllib.algorithms.impala import ImpalaConfig, Impala
from ray.rllib.algorithms.impala.impala import AggregatorWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution import LearnerThread
from ray.rllib.execution.buffers.mixin_replay_buffer import MixInMultiAgentReplayBuffer
from ray.rllib.offline import get_dataset_and_shards
from ray.rllib.offline.estimators import OffPolicyEstimator, ImportanceSampling, DirectMethod, \
    WeightedImportanceSampling, DoublyRobust
from ray.rllib.offline.offline_evaluation_utils import remove_time_dim
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
from ray.rllib.utils import override
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.actors import create_colocated_actors
from ray.rllib.utils.debug import update_global_seed_if_necessary

from ray.rllib.utils.deprecation import DEPRECATED_VALUE, deprecation_warning
from ray.rllib.utils.metrics import NUM_TARGET_UPDATES, LAST_TARGET_UPDATE_TS, NUM_AGENT_STEPS_SAMPLED, \
    NUM_ENV_STEPS_SAMPLED, SYNCH_WORKER_WEIGHTS_TIMER, NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS, \
    NUM_SYNCH_WORKER_WEIGHTS, NUM_AGENT_STEPS_TRAINED, NUM_ENV_STEPS_TRAINED, SAMPLE_TIMER
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.rllib.utils.replay_buffers import ReplayMode
from ray.rllib.utils.typing import AlgorithmConfigDict, PartialAlgorithmConfigDict, ResultDict
from ray.util.iter import _NextValueNotReady

from vmpo.NoCopyWorkerSet import LightWorkerSet
from vmpo.VmpoPolicy import VmpoPolicy

import tensorflow as tf
from PIL.PngImagePlugin import PngInfo

logger = logging.getLogger(__name__)


# Vmpo implementation with importance sampling
# discrete action space

class VmpoConfig(ImpalaConfig):

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or Vmpo)

        self.eta = 1.
        self.alpha = 5.
        self.eps_eta = 0.01
        self.eps_alpha = 5e-5
        self.statistics_lr = 1e-4
        self.target_network_update_freq = 1000
        # Override some hps
        self.lr = 1e-4
        self.replay_buffer_num_slots = 256
        self.learner_queue_timeout = 300


    def training(
            self,
            *,
            eta: Optional[float] = None,
            alpha: Optional[float] = None,
            eps_eta: Optional[float] = None,
            eps_alpha: Optional[float] = None,
            target_network_update_freq: Optional[int] = None,
            statistics_lr: Optional[float] = None,
            **kwargs,
    ) -> "VmpoConfig":

        super().training(**kwargs)

        if eta is not None:
            self.eta = eta

        if alpha is not None:
            self.alpha = alpha

        if eps_alpha is not None:
            self.eps_alpha = eps_alpha

        if eps_eta is not None:
            self.eps_eta = eps_eta

        if target_network_update_freq is not None:
            self.target_network_update_freq = target_network_update_freq

        if statistics_lr is not None:
            self.statistics_lr = statistics_lr

        return self


class Vmpo(Impala):
    @classmethod
    @override(Impala)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return VmpoConfig()

    @classmethod
    @override(Impala)
    def get_default_policy_class(
            cls, config: PartialAlgorithmConfigDict
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            raise NotImplementedError
        elif config["framework"] == "tf2":
            raise NotImplementedError
        elif config["framework"] == "tf":
            return VmpoPolicy

    @override(Impala)
    def setup(self, config: AlgorithmConfig) -> None:


        self.policy_update_idx = 0
        self.old_policy_weights = {}

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
            self.workers = LightWorkerSet(
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

                if not self.config._enable_learner_api:
                    # Create and start the learner thread.
                    self._learner_thread = VmpoLearnerThread(
                        self.workers.local_worker(),
                        minibatch_buffer_size=config["minibatch_buffer_size"],
                        num_sgd_iter=config["num_sgd_iter"],
                        learner_queue_size=config["learner_queue_size"],
                        learner_queue_timeout=config["learner_queue_timeout"],
                        target_network_update_freq=config["target_network_update_freq"],
                        policies=list(self.workers.local_worker().get_policies_to_train())
                    )
                    self._learner_thread.start()

                # TODO VICTOR
                # self.workers.sync_weights()
                self.update_workers_if_necessary(
                    workers_that_need_updates=set(self.workers.healthy_worker_ids())
                )

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
            "is" : ImportanceSampling,
            "wis": WeightedImportanceSampling,
            "dm" : DirectMethod,
            "dr" : DoublyRobust,
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

        # Impala setup

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

    def training_step(self) -> ResultDict:
        # First, check, whether our learner thread is still healthy.
        if not self.config._enable_learner_api and not self._learner_thread.is_alive():
            raise RuntimeError("The learner thread died while training!")

        use_tree_aggregation = (
                self._aggregator_actor_manager
                and self._aggregator_actor_manager.num_healthy_actors() > 0
        )

        # Get references to sampled SampleBatches from our workers.
        unprocessed_sample_batches = self.get_samples_from_workers(
            return_object_refs=use_tree_aggregation,
        )
        # Tag workers that actually produced ready sample batches this iteration.
        # Those workers will have to get updated at the end of the iteration.
        workers_that_need_updates = {
            worker_id for worker_id, _ in unprocessed_sample_batches
        }

        # Send the collected batches (still object refs) to our aggregation workers.
        if use_tree_aggregation:
            batches = self.process_experiences_tree_aggregation(
                unprocessed_sample_batches
            )
        # Resolve collected batches here on local process (using the mixin buffer).
        else:
            batches = self.process_experiences_directly(unprocessed_sample_batches)

        # Increase sampling counters now that we have the actual SampleBatches on
        # the local process (and can measure their sizes).
        for batch in batches:
            self._counters[NUM_ENV_STEPS_SAMPLED] += batch.count
            self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
        # Concatenate single batches into batches of size `train_batch_size`.
        self.concatenate_batches_and_pre_queue(batches)
        if self.config._enable_learner_api:
            train_results = self.learn_on_processed_samples()
        else:
            # Move train batches (of size `train_batch_size`) onto learner queue.
            self.place_processed_samples_on_learner_thread_queue()
            # Extract most recent train results from learner thread.
            train_results = self.process_trained_results()

        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:

            pids = list(train_results.keys())
            self.update_workers_if_necessary(
                workers_that_need_updates=workers_that_need_updates,
                policy_ids=pids,
            )

        # With a training step done, try to bring any aggregators back to life
        # if necessary.
        # Aggregation workers are stateless, so we do not need to restore any
        # state here.
        if self._aggregator_actor_manager:
            self._aggregator_actor_manager.probe_unhealthy_actors(
                timeout_seconds=self.config.worker_health_probe_timeout_s,
                mark_healthy=True,
            )

        # cur_ts = self._counters[
        #     NUM_AGENT_STEPS_SAMPLED
        #     if self.config.count_steps_by == "agent_steps"
        #     else NUM_ENV_STEPS_SAMPLED
        # ]

        # last_update = self._counters[LAST_TARGET_UPDATE_TS]
        # target_update_steps_freq = (
        #         self.config.train_batch_size
        #         * self.config.num_sgd_iter
        #         * self.config.target_network_update_freq
        # )
        # if cur_ts - last_update >= target_update_steps_freq:
        #
        #     local_worker = self.workers.local_worker()
        #     policies = list(local_worker.get_policies_to_train())
        #     next_in_line = policies[self.policy_update_idx]
        #     self.policy_update_idx += 1
        #     if self.policy_update_idx == len(policies):
        #         self.policy_update_idx = 0
        #
        #     if self.config.policy_states_are_swappable:
        #         local_worker.lock()
        #     weights = local_worker.get_weights([next_in_line])
        #     if self.config.policy_states_are_swappable:
        #         local_worker.unlock()
        #     self.old_policy_weights[next_in_line] = weights[next_in_line]
        #
        #     self.workers.local_worker().get_policy(next_in_line).update_target(weights[next_in_line])
        #
        #     self._counters[NUM_TARGET_UPDATES] += 1
        #     self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

        # if self.old_policy_weights:
        #    self.sync_weights(ray.put(self.old_policy_weights))

        return train_results

    def sync_weights(self, weights):

        self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS] = 0
        self._counters[NUM_SYNCH_WORKER_WEIGHTS] += 1

        self.workers.foreach_worker(
            func=lambda w: w.set_weights(ray.get(weights)),
            local_worker=False,
            timeout_seconds=0,  # Don't wait for the workers to finish.
        )

    def update_weights_and_policy_mapping_fn(
            self,
            workers_that_need_updates: Set[int],
            policy_ids = None,
            policy_mapping_fn = None
        ):
        local_worker = self.workers.local_worker()

        if self.config.policy_states_are_swappable:
            local_worker.lock()
        weights = local_worker.get_weights(policy_ids)
        if self.config.policy_states_are_swappable:
            local_worker.unlock()

        # We only care about policy
        for p_id in policy_ids:
            keys = list(weights[p_id].keys())
            for k in keys:
                if "value" in k:
                    weights[p_id].pop(k)
        weights = ray.put(weights)

        def setter(worker: RolloutWorker):
            worker.set_weights(ray.get(weights))
            worker.set_policy_mapping_fn(policy_mapping_fn)

        self._learner_thread.policy_ids_updated.clear()
        self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS] = 0
        self._counters[NUM_SYNCH_WORKER_WEIGHTS] += 1
        self.workers.foreach_worker(
            func=setter,
            local_worker=False,
            remote_worker_ids=list(workers_that_need_updates),
            timeout_seconds=0,  # Don't wait for the workers to finish.
        )


    def update_workers_if_necessary(
            self,
            workers_that_need_updates: Set[int],
            policy_ids=None,
    ) -> None:
        """Updates all RolloutWorkers that require updating.

        Updates only if NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS has been
        reached and the worker has sent samples in this iteration. Also only updates
        those policies, whose IDs are given via `policies` (if None, update all
        policies).

        Args:
            workers_that_need_updates: Set of worker IDs that need to be updated.
            policy_ids: Optional list of Policy IDs to update. If None, will update all
                policies on the to-be-updated workers.
        """
        local_worker = self.workers.local_worker()
        # Update global vars of the local worker.
        if self.config.policy_states_are_swappable:
            local_worker.lock()
        global_vars = {
            "timestep"                   : self._counters[NUM_AGENT_STEPS_TRAINED],
            "num_grad_updates_per_policy": {
                pid: local_worker.policy_map[pid].num_grad_updates
                for pid in policy_ids or []
            },
        }
        local_worker.set_global_vars(global_vars, policy_ids=policy_ids)
        if self.config.policy_states_are_swappable:
            local_worker.unlock()

        # Only need to update workers if there are remote workers.
        self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS] += 1
        if (
                self.workers.num_remote_workers() > 0
                and self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS]
                >= self.config.broadcast_interval
                and workers_that_need_updates
        ):
            if self.config.policy_states_are_swappable:
                local_worker.lock()
            weights = local_worker.get_weights(policy_ids)
            if self.config.policy_states_are_swappable:
                local_worker.unlock()

            # We only care about policy
            for p_id in weights:
                keys = list(weights[p_id].keys())
                for k in keys:
                    if "value" in k or "ICM" in k:
                        weights[p_id].pop(k)

            weights = ray.put(weights)


            self._learner_thread.policy_ids_updated.clear()
            self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS] = 0
            self._counters[NUM_SYNCH_WORKER_WEIGHTS] += 1
            self.workers.foreach_worker(
                func=lambda w: w.set_weights(ray.get(weights), global_vars),
                local_worker=False,
                remote_worker_ids=list(workers_that_need_updates),
                timeout_seconds=0,  # Don't wait for the workers to finish.
            )


class VmpoLearnerThread(LearnerThread):

    def __init__(self, *args, policies=[], target_network_update_freq=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_network_update_freq = target_network_update_freq
        self.policies = policies
        self.target_step_counters = 0
        self.policy_target_update_idx = 0
        self.batch_counts = defaultdict(lambda: 0)

        # self.target_step_counters = defaultdict(lambda: 0)

    def step(self) -> Optional[_NextValueNotReady]:
        with self.queue_timer:
            try:
                batch, _ = self.minibatch_buffer.get()
            except queue.Empty:
                return _NextValueNotReady()
        with self.grad_timer:
            # Use LearnerInfoBuilder as a unified way to build the final
            # results dict from `learn_on_loaded_batch` call(s).
            # This makes sure results dicts always have the same structure
            # no matter the setup (multi-GPU, multi-agent, minibatch SGD,
            # tf vs torch).
            learner_info_builder = LearnerInfoBuilder(num_devices=1)
            if self.local_worker.config.policy_states_are_swappable:
                self.local_worker.lock()


            multi_agent_results = self.local_worker.learn_on_batch(batch)

            self.local_worker.foreach_policy_to_train(
                lambda p, p_id: p.update_statistics(p_id, multi_agent_results[p_id])
            )
            self.target_step_counters += 1
            # if self.target_step_counters % self.target_network_update_freq == 0:
            #        self.local_worker.get_policy(self.policies[self.policy_target_update_idx % len(self.policies)]).update_target()
            #        self.policy_target_update_idx += 1

            if self.local_worker.config.policy_states_are_swappable:
                self.local_worker.unlock()
            self.policy_ids_updated.extend(list(multi_agent_results.keys()))
            for pid, results in multi_agent_results.items():
                self.batch_counts[pid] += 1
                if self.batch_counts[pid] % self.target_network_update_freq == 0:
                    self.local_worker.get_policy(pid).update_target()

                stats = results["learner_stats"]

                results["batch_count"] = self.batch_counts[pid]
                learner_info_builder.add_learn_on_batch_results(results, pid)



            self.learner_info = learner_info_builder.finalize()

        self.num_steps += 1

        # Put tuple: env-steps, agent-steps, and learner info into the queue.
        self.outqueue.put((batch.count, batch.agent_steps(), self.learner_info))
        self.learner_queue_size.push(self.inqueue.qsize())