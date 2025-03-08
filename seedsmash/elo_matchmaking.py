import time
from typing import Dict
import numpy as np
import ray
from polaris.experience import MatchMaking
from polaris.policies import PolicyParams

from seedsmash.bot import Bot
from seedsmash.bots.bot_config import BotConfig

#from seedsmash.window_worker import WindowWorker
# @ray.remote(num_cpus=1, num_gpus=0)
# class RankingWindowWorker(WindowWorker):
#     def __init__(self, update_interval_s=5, pipe_name="pipe"):
#         super().__init__(window=RankingWindow(), update_interval_s=update_interval_s, pipe_name=pipe_name)
#
#     def update_window(self, dt, **k):
#         data = super().update_window(dt, **k)
#         if data is not None:
#             self.window.update_ratings(data)



class SeedSmashMatchmaking(MatchMaking):

    def __init__(
            self,
            agent_ids,
            lr=12,

    ):
        super().__init__(agent_ids=agent_ids)
        self.lr = lr

    def next(
            self,
            params_map: Dict[str, "PolicyParams"],
            wid: int,
            **kwargs,
    ) -> Dict[str, "PolicyParams"]:

        if wid == 0:
            # TODO: read matchup from bot requests (read from database)
            # pick uniformly for stream
            p = None
        else:

            total_samples = np.array([
                p.options.offset_samples_generated for p in params_map.values()
            ])
            total_samples -= np.min(total_samples)
            delta = np.maximum(1e-8, np.max(total_samples) - total_samples)
            p = delta / delta.sum()

        policies = list(params_map.keys())


        pid_a = np.random.choice(list(params_map.keys()), p=p)
        policies.remove(pid_a)

        ratings = np.array([
            params_map[pid].options.elo for pid in policies
        ])

        rating_gaps = params_map[pid_a].options.elo - ratings

        winning_probs = self.expected_outcome(rating_gaps)
        sigma_squared = (1/6)**2 #(1/6)**2 #
        probabilities = np.exp(-(winning_probs-0.5)**2/(2*sigma_squared)) / np.sqrt(2*np.pi*sigma_squared)
        probabilities /= probabilities.sum()

        pid_b = np.random.choice(policies, p=probabilities)

        return {
            1: params_map[pid_a],
            2: params_map[pid_b]
        }

    @staticmethod
    def expected_outcome(delta_elo):
        # 400 is just a score used for human normalisation
        return 1 / (1 + np.power(10, -delta_elo / 400.))

    def update(
            self,
            bot_a: Bot,
            bot_b: Bot,
            outcome: float
    ):

        elo_a = bot_a.elo
        elo_b = bot_b.elo

        delta_elo = elo_a - elo_b

        win_prob = self.expected_outcome(delta_elo)
        update = outcome - win_prob

        elo_a = elo_a + self.lr * update
        elo_b = elo_b + self.lr * (-update)

        bot_a.elo = elo_a
        bot_b.elo = elo_b



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







