import os
import time
from collections import defaultdict
from typing import Dict, List, Union, Tuple, Any
from abc import ABC
import numpy as np
import ray
from polaris.experience import MatchMaking
from polaris.policies import PolicyParams

from seedsmash2.submissions.bot_config import BotConfig
from seedsmash2.visualisation.ranking import RankingWindow

import pyglet
import zmq

from seedsmash2.window_worker import WindowWorker


@ray.remote(num_cpus=1, num_gpus=0)
class RankingWindowWorker(WindowWorker):
    def __init__(self, update_interval_s=5, pipe_name="pipe"):
        super().__init__(window=RankingWindow(), update_interval_s=update_interval_s, pipe_name=pipe_name)

    def update_window(self, dt, **k):
        data = super().update_window(dt, **k)
        if data is not None:
            self.window.update_ratings(data)



class SeedSmashMatchmaking(MatchMaking):

    def __reduce__(self):

        return (self.__class__, (self.agent_ids,self.initial_elo, self.initial_lr, self.annealing, self.final_lr,
                                 self.win_rate_lr, self.match_count, self.elo_scores, self.lr))


    def __init__(
            self,
            agent_ids,
            initial_elo=1000,
            initial_lr=40,
            annealing=0.99,
            final_lr=20,
            win_rate_lr=2e-2,
            match_count=None,
            elo_scores=None,
            lr=None,

    ):
        super().__init__(agent_ids=agent_ids)

        self.initial_elo = initial_elo
        self.initial_lr = initial_lr
        self.annealing = annealing
        self.final_lr = final_lr

        self.match_count = defaultdict(int) if match_count is None else match_count

        self.elo_scores = {} if elo_scores is None else elo_scores
        self.lr = {} if lr is None else lr

        # We keep track for some more stats
        self.win_rate_lr = win_rate_lr
        self.win_rates = defaultdict(lambda: 0.5)

        pipe_name = "ratingdata_pipe"
        try:
            os.remove(pipe_name)
        except Exception as e:
            print(e)

        self.window = RankingWindowWorker.remote(update_interval_s=30, pipe_name=pipe_name)
        context = zmq.Context()
        self.push_pipe = context.socket(zmq.PUSH)
        self.push_pipe.bind(f"ipc://{pipe_name}")

    def next(
            self,
            params_map: Dict[str, "PolicyParams"],
            **kwargs,
    ) -> Dict[str, "PolicyParams"]:

        # Do not make players play against themselves
        pcopy = params_map.copy()
        policies = []
        sampled_policy = np.random.choice(list(params_map.keys()))
        policies.append(pcopy.pop(sampled_policy))

        ratings = np.array([
            self.elo_scores.get(p, self.initial_elo)
            for p in pcopy
        ])
        sampled_policy_rating = self.elo_scores.get(sampled_policy, self.initial_elo)
        rating_gaps = sampled_policy_rating - ratings

        winning_probs = self.expected_outcome(rating_gaps)
        sigma_squared = (1/5)**2 #(1/6)**2 #
        probabilities = np.exp(-(winning_probs-0.5)**2/(2*sigma_squared)) / np.sqrt(2*np.pi*sigma_squared)
        probabilities /= probabilities.sum()

        sampled_opponent = np.random.choice(list(pcopy.keys()), p=probabilities)

        return {
            1: params_map[sampled_policy],
            2: params_map[sampled_opponent]
        }


    def expected_outcome(self, delta_elo):
        # 400 is just a score used for human normalisation
        return 1 / (1 + np.power(10, -delta_elo / 400.))

    def update(
            self,
            pid1: str,
            pid2: str,
            outcome: float
    ):
        for pid in (pid1, pid2):
            if pid not in self.elo_scores:
                self.elo_scores[pid] = self.initial_elo
                self.lr[pid] = self.initial_lr

        delta_elo = self.elo_scores[pid1] - self.elo_scores[pid2]

        win_prob = self.expected_outcome(delta_elo)
        update = outcome - win_prob

        self.elo_scores[pid1] = self.elo_scores[pid1] + self.lr[pid1] * update
        self.elo_scores[pid2] = self.elo_scores[pid2] + self.lr[pid2] * (-update)

        self.match_count[pid1] += 1
        self.match_count[pid2] += 1

        self.lr[pid1] = np.maximum(self.final_lr, self.annealing * self.lr[pid1])
        self.lr[pid2] = np.maximum(self.final_lr, self.annealing * self.lr[pid2])

        self.win_rates[pid1] = self.win_rates[pid1] * (1 - self.win_rate_lr) + outcome * self.win_rate_lr
        self.win_rates[pid2] = self.win_rates[pid2] * (1 - self.win_rate_lr) + (1-outcome) * self.win_rate_lr

    def metrics(self):

        metrics = []
        for m_name, values in zip(["rating", "games_played", "winrate", "elo_learning_rate"],
                                  [self.elo_scores, self.match_count, self.win_rates, self.lr]):
            for pid, value in values.items():
                metrics.append((f"{pid}/" + m_name, value))

        return metrics

    def update_policy_stats(self, policy_map):

        sorted_pids = sorted(self.elo_scores.keys(), key=lambda k: -self.elo_scores[k])
        for rank, pid in enumerate(sorted_pids, 1):
            for m_name, values in zip(["rating", "games_played", "winrate"],
                                      [self.elo_scores, self.match_count, self.win_rates]):
                if pid in policy_map:
                    policy_map[pid].stats[m_name] = values.get(pid, 0)

            policy_map[pid].stats["rank"] = rank

        self.push_pipe.send_pyobj({
            pid: PolicyParams(name=policy.name, stats=policy.stats, options=policy.options)
            for pid, policy in policy_map.items()
        })


if __name__ == '__main__':


    matchmaking = SeedSmashMatchmaking(
        agent_ids={1, 2}
    )

    policies = {f"player_{i}": PolicyParams(
        name=f"player_{i}",
        options=BotConfig(tag=f"player_{i}", character="MARTH", costume=i%3),
        stats={"rating": i, 'rank': 20}
    ) for i in range(20)}

    while True:
        match_making = matchmaking.next(policies)
        outcome = np.random.choice([0, 1])
        matchmaking.update(
            match_making[1].name, match_making[2].name, outcome
        )
        matchmaking.update_policy_stats(policies)
        time.sleep(0.5)







