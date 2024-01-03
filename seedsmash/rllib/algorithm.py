import functools
from typing import Optional, Dict

from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.alpha_star.league_builder import LeagueBuilder
from ray.rllib.algorithms.appo import APPO, APPOConfig
from ray.rllib.algorithms.impala import Impala
from ray.rllib.evaluation.metrics import collect_episodes
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils import override
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.utils.metrics import SYNCH_WORKER_WEIGHTS_TIMER, NUM_AGENT_STEPS_SAMPLED, NUM_ENV_STEPS_SAMPLED
from ray.rllib.utils.typing import ResultDict

from seedsmash.rllib.league import GeneticLeague


class SeedSmashConfig(APPOConfig):


    def multi_agent(
        self,
        *,
        population_size=30,
        hyperparameter,
        **kwargs

    ) -> AlgorithmConfig:
        pass









class SeedSmash(APPO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.population: LeagueBuilder

    @override(APPO)
    def setup(self, config: AlgorithmConfig):
        super().setup(config)

        self.population = GeneticLeague(algo=self, algo_config=self.config)

    @override(APPO)
    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.update(
            {
                "population": self.population.__getstate__(),
            }
        )
        return state

    @override(APPO)
    def __setstate__(self, state: dict) -> None:
        state_copy = state.copy()
        super().__setstate__(state_copy)
        self.population.__setstate__(state.pop("population", {}))

    @override(APPO)
    def step(self) -> ResultDict:
        # Perform a full step (including evaluation).:
        """Implements the main `Trainer.train()` logic.

        Takes n attempts to perform a single training step. Thereby
        catches RayErrors resulting from worker failures. After n attempts,
        fails gracefully.

        Override this method in your Trainer sub-classes if you would like to
        handle worker failures yourself.
        Otherwise, override only `training_step()` to implement the core
        algorithm logic.

        Returns:
            The results dict with stats/infos on sampling, training,
            and - if required - evaluation.
        """
        # Do we have to run `self.evaluate()` this iteration?
        # `self.iteration` gets incremented after this function returns,
        # meaning that e. g. the first time this function is called,
        # self.iteration will be 0.
        evaluate_this_iter = (
                self.config.evaluation_interval is not None
                and (self.iteration + 1) % self.config.evaluation_interval == 0
        )

        # Results dict for training (and if appolicable: evaluation).
        results: ResultDict = {}

        # Parallel eval + training: Kick off evaluation-loop and parallel train() call.
        if evaluate_this_iter and self.config.evaluation_parallel_to_training:
            (
                results,
                train_iter_ctx,
            ) = self._run_one_training_iteration_and_evaluation_in_parallel()
        # - No evaluation necessary, just run the next training iteration.
        # - We have to evaluate in this training iteration, but no parallelism ->
        #   evaluate after the training iteration is entirely done.
        else:
            results, train_iter_ctx = self._run_one_training_iteration()

        # Sequential: Train (already done above), then evaluate.
        if evaluate_this_iter and not self.config.evaluation_parallel_to_training:
            results.update(self._run_one_evaluation(train_future=None))

        # Attach latest available evaluation results to train results,
        # if necessary.
        if not evaluate_this_iter and self.config.always_attach_evaluation_results:
            assert isinstance(
                self.evaluation_metrics, dict
            ), "Trainer.evaluate() needs to return a dict."
            results.update(self.evaluation_metrics)

        if hasattr(self, "workers") and isinstance(self.workers, WorkerSet):
            # Sync filters on workers.
            self._sync_filters_if_needed(
                from_worker=self.workers.local_worker(),
                workers=self.workers,
                timeout_seconds=self.config.sync_filters_on_rollout_workers_timeout_s,
            )
            # TODO (avnishn): Remove the execution plan API by q1 2023
            # Collect worker metrics and add combine them with `results`.
            if self.config._disable_execution_plan_api:
                # VICTOR: collect metrics every 8 mins:
                if ((self.iteration + 1) % (8 * 60 // self.config.min_time_s_per_iteration)) == 0:
                    print("Collecting metrics...")
                    episodes_this_iter = collect_episodes(
                        self.workers,
                        self._remote_worker_ids_for_metrics(),
                        timeout_seconds=self.config.metrics_episode_collection_timeout_s,
                    )
                    print("done collecting.")
                    results = self._compile_iteration_results(
                        episodes_this_iter=episodes_this_iter,
                        step_ctx=train_iter_ctx,
                        iteration_results=results,
                    )
                else :
                    print("Collecting metrics normally...")
                    workers = self._remote_worker_ids_for_metrics().copy()
                    workers.remove(16)
                    episodes_this_iter = collect_episodes(
                        self.workers,
                        workers,
                        timeout_seconds=self.config.metrics_episode_collection_timeout_s,
                    )
                    print("done collecting normally.")
                    results = self._compile_iteration_results(
                        episodes_this_iter=episodes_this_iter,
                        step_ctx=train_iter_ctx,
                        iteration_results=results,
                    )

        # Check `env_task_fn` for possible update of the env's task.
        if self.config.env_task_fn is not None:
            if not callable(self.config.env_task_fn):
                raise ValueError(
                    "`env_task_fn` must be None or a callable taking "
                    "[train_results, env, env_ctx] as args!"
                )

            def fn(env, env_context, task_fn):
                new_task = task_fn(results, env, env_context)
                cur_task = env.get_task()
                if cur_task != new_task:
                    env.set_task(new_task)

            fn = functools.partial(fn, task_fn=self.config.env_task_fn)
            self.workers.foreach_env_with_context(fn)

        # Based on the (train + evaluate) results, perform a step of
        # league building.
        self.population.build_league(result=results)

        return results


    @override(APPO)
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

        # Sync worker weights (only those policies that were actually updated).
        # TODO : Victor
        #   removed pid 16 for spectator, we update the weights upon game end.

        # TODO : FIND A WAY TO UPDATE ONLY THE CURRENTLY USED POLICIES

        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            pids = list(train_results.keys())
            self.update_workers_if_necessary(
                workers_that_need_updates=workers_that_need_updates,
                policy_ids=pids,
            )
            # for worker_idx, sample_batches in unprocessed_sample_batches:
            #     if worker_idx == 16:
            #         spectator_data = sample_batches
            #         currently_used_policies = list(spectator_data.policy_batches.keys())
            #         print("spectator uses the following policies", currently_used_policies)
            #         self.update_workers_if_necessary(
            #             workers_that_need_updates={16},
            #             policy_ids=currently_used_policies,
            #         )
            #         break


        # With a training step done, try to bring any aggregators back to life
        # if necessary.
        # Aggregation workers are stateless, so we do not need to restore any
        # state here.
        if self._aggregator_actor_manager:
            self._aggregator_actor_manager.probe_unhealthy_actors(
                timeout_seconds=self.config.worker_health_probe_timeout_s,
                mark_healthy=True,
            )

        # Specific to APPO
        self.after_train_step(train_results)

        return train_results


    @override(APPO)
    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.update(
            {
                "population": self.population.__getstate__(),
            }
        )
        return state

    @override(APPO)
    def __setstate__(self, state: dict) -> None:
        state_copy = state.copy()
        self.population.__setstate__(state.pop("population", {}))
        super().__setstate__(state_copy)