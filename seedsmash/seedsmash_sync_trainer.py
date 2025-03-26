import copy
import importlib
import os
import queue
import threading
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import polaris.experience.matchmaking
import tree
from melee import Action
from ml_collections import ConfigDict

from polaris.checkpointing.checkpointable import Checkpointable
from polaris.experience.episode import EpisodeMetrics, NamedPolicyMetrics
from polaris.experience.worker_set import SyncWorkerSet
from polaris.experience.matchmaking import MatchMaking
from polaris.environments.polaris_env import PolarisEnv
from polaris.policies.policy import Policy, PolicyParams, ParamsMap
from polaris.experience.sampling import ExperienceQueue, SampleBatch
from polaris.utils.metrics import MetricBank, GlobalCounter, GlobalTimer

import psutil

from seedsmash.api import ApiInterface, SeedSmashDataBag, Game, jsonify_game
from seedsmash.bot import Bot
from seedsmash.bots.bot_config import BotConfig
from seedsmash.bots.generate_form import load_filled_form
from seedsmash.elo_matchmaking import SeedSmashMatchmaking
from seedsmash.utils import ActionStateCounts, inject_botconfig, ActionStateHitCounts


class SeedSmashTrainer(Checkpointable):
    def __init__(
            self,
            config: ConfigDict,
            restore=False,
    ):

        self.api_interface = ApiInterface(config["db_address"])

        self.config = config
        self.worker_set = SyncWorkerSet(
            config,
            with_spectator=False,
        )

        # Init environment
        self.env = PolarisEnv.make(self.config.env, env_index=-1, **self.config.env_config)

        self.PolicylCls = getattr(importlib.import_module(self.config.policy_path), self.config.policy_class)
        self.policy_map: Dict[str, Policy] = {}

        self.params_map = ParamsMap()

        self.experience_queue: Dict[str, ExperienceQueue] = {}

        self.matchmaking = SeedSmashMatchmaking(agent_ids=self.env.get_agent_ids())

        self.running_experience_jobs = []
        self.running_spectate_jobs = []

        self.metricbank = MetricBank(
            report_freq=self.config.report_freq
        )

        self.metrics = self.metricbank.metrics

        super().__init__(
            checkpoint_config = config.checkpoint_config,

            components={
                "config": self.config,
                "params_map": self.params_map,
                "metrics": self.metrics,
            }
        )

        if restore:
            if isinstance(restore, str):
                self.restore(restore_path=restore)
            else:
                self.restore()

            # override user config
            self.config = config

            # Need to pass the restored references afterward
            self.metricbank.metrics = self.metrics

            env_step_counter = "counters/" + GlobalCounter.ENV_STEPS
            if env_step_counter in self.metrics:
                GlobalCounter[GlobalCounter.ENV_STEPS] = self.metrics["counters/" + GlobalCounter.ENV_STEPS].get()

            for policy_name, params in self.params_map.items():
                self.policy_map[policy_name] = self.PolicylCls(
                    name=policy_name,
                    action_space=self.env.action_space,
                    observation_space=self.env.observation_space,
                    config=self.config,
                    policy_config=params.config,
                    options=params.options,
                    stats={"rank": 100, "rating": 1000, "games_played": 0, "winrate": 0},
                    # For any algo that needs to track either we have the online model
                    is_online=True,
                )
                self.policy_map[policy_name].setup(params)
                self.experience_queue[policy_name] = ExperienceQueue(self.config)

        self.games_outcome_queue: List[Game] = []
        self.last_database_state_update_time = self.last_database_game_update_time = self.startup_time = time.time()
        self.update_live_bots(self.api_interface.read_db_bots())
        self.agent_frames_since_startup = 0

    def update_live_bots(self, db_bots: List[Bot]):
        if len(db_bots) == 0:
            return
        db_bot_tags = {bot.tag for bot in db_bots}  # Set of current bot tags in the database

        # Add missing bots from db_bots to params_map
        for bot in db_bots:
            if bot.tag not in self.params_map:
                self.inject_bot(bot)

        # Remove bots that are no longer in db_bots
        coaches = set()
        for pid, params in self.params_map.items():
            bot: Bot = params.options
            if bot.is_coached():
                coaches.add(bot.coach_tag)
        to_remove = set(self.params_map) - (db_bot_tags | coaches)
        for tag in to_remove:
            # TODO:
            # handle the case where this gets deleted but some games with this bot are still ongoing.
            del self.params_map[tag]
            del self.policy_map[tag]
            del self.experience_queue[tag]


    def communicate_with_db_if_needed(self):
        t = time.time()
        if t - self.last_database_game_update_time > self.config["database_game_update_freq_s"]:
            self.last_database_game_update_time = t
            # TODO: communicate elo/rank every minute
            # every 10 mins elo rank but for metrics
            params = list(self.params_map.values())

            def rank_value(p):
                bot = p.options
                if bot.is_out:
                    return 1e8
                else:
                    return -bot.elo

            params = sorted(params, key=rank_value)
            if t - self.last_database_state_update_time > self.config["database_state_update_freq_s"]:
                self.last_database_state_update_time = t

                bot_states = [
                    p.options.get_state(rank+1)
                    for rank, p in enumerate(params)
                ]

            else:
                bot_states = None

            data = SeedSmashDataBag(
                games=self.games_outcome_queue,
                bot_states=bot_states,
                bot_rankings=[{
                    "tag": p.options.tag,
                    "elo": p.options.elo,
                    "rank": rank+1}
                    for rank, p in enumerate(params)
                ]
            )

            db_bots: List[Bot] = self.api_interface.communicate(data)
            self.games_outcome_queue = []

            # Do we have new bots ?
            self.update_live_bots(db_bots)



    def inject_bot(self, bot: Bot):

        if len(self.params_map)> 0:
            # init new policies with same amount of samples, so that they do not get sampled all the time!
            mean_sample_generated = np.mean([
                p.options.offset_samples_generated for p in self.params_map.values()
            ])
        else:
            mean_sample_generated = 0

        bot.mean_samples_at_creation = mean_sample_generated

        policy_config = copy.deepcopy(self.config.default_policy_config)
        # This alters the weights for action state rewards and discount factor
        # TODO:
        # for now, keep those fixed for all bots.
        # We may add patience (or reflexion) and creativity later.
        #inject_botconfig(policy_config, bot_config)
        pid = bot.tag
        self.policy_map[pid] = self.PolicylCls(
            name=pid,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            config=self.config,
            policy_config=policy_config,
            options=bot,
            is_online=True,
        )

        self.params_map[pid] = self.policy_map[pid].get_params()
        self.experience_queue[pid] = ExperienceQueue(self.config)
        print("New bot ! :", bot)


    def training_step(self):
        """
        Executes one iteration of the trainer.
        :return: Training iteration results
        """
        GlobalTimer[GlobalTimer.PREV_ITERATION] = time.time()
        self.communicate_with_db_if_needed()
        if len(self.policy_map) < 2:
            return

        experience = self.recv()
        experience_metrics = self.process_experience(experience)
        training_metrics = self.train()

        self.process_metrics(experience_metrics, training_metrics)


    def recv(self) -> List[EpisodeMetrics | SampleBatch]:
        experience_jobs = [self.matchmaking.next(self.params_map, wid) for wid in self.worker_set.available_workers]

        self.running_experience_jobs += self.worker_set.push_jobs(self.params_map, experience_jobs)
        experience, self.running_experience_jobs = self.worker_set.wait(self.params_map, self.running_experience_jobs, timeout=1e-2)
        # if len(experience)>0:
        #     print("collected ", len(experience), "experiences")

        return experience


    def process_experience(self, experience):
        experience_metrics = []
        for exp_batch in experience:
            if isinstance(exp_batch, EpisodeMetrics):
                try:
                    # If this fails, it means the episode exited early
                    pid1, pid2 = exp_batch.policy_metrics.keys()
                    game_info = exp_batch.custom_metrics.pop("game_info")

                    if pid1 not in self.policy_map or pid2 not in self.policy_map:
                        # This game is outdated (at least one bot was removed)
                        continue

                    bot_a: Bot = self.policy_map[pid1].options
                    bot_b: Bot = self.policy_map[pid2].options

                    winner = game_info["winner"]
                    outcome = 0.5 if winner is None else float(winner == bot_a.tag)
                    self.matchmaking.update(
                        self.policy_map[pid1].options,
                        self.policy_map[pid2].options,
                        outcome
                    )
                    self.games_outcome_queue.append(
                        jsonify_game(
                            bot_a,
                            bot_b,
                            winner,
                            game_info["stage"],
                            game_info["length"],
                            game_info["replay"],
                        )
                    )
                    bot_a.push_metrics(game_info["metrics"]["bot_a"], registry="progression")
                    bot_b.push_metrics(game_info["metrics"]["bot_b"], registry="progression")
                    # disable metrics here
                    # experience_metrics.append(exp_batch)
                    GlobalCounter[GlobalCounter.ENV_STEPS] += exp_batch.length
                    self.agent_frames_since_startup += exp_batch.length * 2
                    GlobalCounter[GlobalCounter.NUM_EPISODES] += 1

                except Exception as e:
                    print(e, exp_batch)

            else:  # Experience batch
                batch_pid = exp_batch.get_owner()
                if batch_pid not in self.policy_map:
                    continue

                owner = self.policy_map[batch_pid]

                if (not self.experience_queue[owner.name].is_ready()) and owner.version == \
                        exp_batch[SampleBatch.VERSION][0]:

                    exp_batch = exp_batch.pad_sequences()
                    exp_batch[SampleBatch.SEQ_LENS] = np.array(exp_batch[SampleBatch.SEQ_LENS])
                    self.experience_queue[owner.name].push([exp_batch])
                    self.policy_map[owner.name].options.num_samples_generated += exp_batch.size()
                    GlobalCounter.incr("batch_count")

                elif owner.version != exp_batch[SampleBatch.VERSION][0]:
                    # pass
                    # toss the batch...
                    pass
                else:
                    # toss the batch...
                    pass

        return experience_metrics


    def train(self) -> dict:
        training_metrics = {}
        for policy_name, policy_queue in self.experience_queue.items():
            if not policy_queue.is_ready():
                continue
            bot: Bot = self.policy_map[policy_name].options
            coach_policy = None if not bot.is_coached() \
                else self.policy_map.get(bot.coach_tag, None)

            coach_model = None if coach_policy is None else coach_policy.model

            pulled_batch = policy_queue.pull(self.config.train_batch_size)
            if np.any(pulled_batch[SampleBatch.VERSION] != self.policy_map[policy_name].version):
                print(
                    f"Had older samples in the batch for policy {policy_name} version {self.policy_map[policy_name].version}!"
                    f" {pulled_batch[SampleBatch.VERSION]}")

            # TODO: we use the batch sampled by the policy to ensure it is different each time and to not have to compute
            #       over two batches.
            train_results = self.policy_map[policy_name].train(
                pulled_batch,
                coach_model=coach_model,
            )

            bot.push_metrics(train_results, registry="rl")
            bot.update_coaching_progression()

            training_metrics[f"{policy_name}"] = train_results
            GlobalCounter.incr(GlobalCounter.STEP)

            params = self.policy_map[policy_name].get_params()
            self.params_map[policy_name] = params

        return training_metrics

    def process_metrics(self, experience_metrics, training_metrics):


        def mean_metric_batch(b):
            return tree.flatten_with_path(tree.map_structure(
                lambda *samples: np.mean(samples),
                *b
            ))

        # Make it policy specific, thus extract metrics of policies.

        if len(training_metrics)> 0:
            for policy_name, policy_training_metrics in training_metrics.items():
                policy_training_metrics = mean_metric_batch([policy_training_metrics])
                self.metricbank.update(policy_training_metrics, prefix=f"training/{policy_name}/",
                                       smoothing=self.config.training_metrics_smoothing)
        if len(experience_metrics) > 0:
            for metrics in experience_metrics:

                self.metricbank.update(tree.flatten_with_path(metrics), prefix=f"experience/",
                                       smoothing=self.config.episode_metrics_smoothing)

        self.metricbank.update(
            [
                ("FPS", self.agent_frames_since_startup / (time.time()-self.startup_time)),
            ]
            , prefix="misc/", smoothing=0.0
        )

        # We should call those only at the report freq...
        self.metricbank.update(
            tree.flatten_with_path(GlobalCounter.get()), prefix="counters/"
        )

    def run(self):
        try:
            while not self.is_done(self.metricbank):
                self.training_step()
                #self.metricbank.report(print_metrics=False)
                self.checkpoint_if_needed()
        except KeyboardInterrupt:
            print("Caught C^.")
            #self.save()






