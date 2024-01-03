import time
from typing import Optional, Dict
import numpy as np
import pyglet.clock
import multiprocessing as mp

from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.alpha_star.league_builder import LeagueBuilder
from ray.rllib.algorithms.appo import APPO, APPOConfig
from ray.rllib.algorithms.impala import Impala
from ray.rllib.utils import override
from ray.rllib.utils.typing import ResultDict

from seedsmash.ga.elo import Elo
from seedsmash.visualization.pyglet_ranking import RankingWindow


class GeneticLeague(LeagueBuilder):

    def __init__(self, algo: Algorithm, algo_config):
        super().__init__(algo=algo, algo_config=algo_config)

        self.step_counter = 1

        # TODO BUILD the multiagent population from there, not config.

        # policy key 3
        self.population = {
            policy_id: {
                "elo": Elo(start=np.random.normal(1000, 1)),
                "name": algo_config["multiagent"]["policies"][policy_id][3]["name"],
                "last_elo": np.random.normal(1000, 1),
                "last_age": 0,
                "char": algo_config["multiagent"]["policies"][policy_id][3]["character"],
                "rank": 1,


            } for policy_id in algo_config["multiagent"]["policies"]
        }

        self.ranking_window = None

        # pyglet.app.platform_event_loop.step()
        # self.ranking_window.switch_to()
        # self.ranking_window.dispatch_events()

        # self._buffer, worker_buffer = mp.Pipe(False)
        # self._shutdown = mp.Event()
        # self._worker = mp.Process(
        #     target=_run_worker,
        #     kwargs=dict(
        #         buffer=worker_buffer,
        #         population=self.population,
        #         shutdown_flag=self._shutdown
        #     )
        # )

    def build_league(self, result: ResultDict) -> None:

        if self.ranking_window is None:
            self.ranking_window = RankingWindow(self.population)
            time.sleep(1.5)

        if "hist_stats" in result:
            for player in self.population:
                if f"{player}_outcome" in result["hist_stats"]:
                    for game_idx in range(len(result["hist_stats"][f"{player}_outcome"])):
                        outcome = result["hist_stats"][f"{player}_outcome"][game_idx]
                        opponent = result["hist_stats"][f"{player}_opponent"][game_idx]
                        if opponent in self.population:
                            print("retrieved game:", outcome, opponent)
                            self.population[player]["elo"].update(self.population[opponent]["elo"], (outcome+1)/2)

        self.ranking_window.update_elos(self.population, frames=60)
        # pyglet.app.platform_event_loop.step()
        # self.ranking_window.switch_to()
        self.ranking_window.dispatch_events()

        print("UPDATED")

        self.step_counter += 1

    def __getstate__(self) -> Optional[Dict]:
        pyglet.app.platform_event_loop.step()
        self.ranking_window.switch_to()
        self.ranking_window.dispatch_events()

        # todo : save population settings ?
        return {"population": self.population}

    def __setstate__(self, state: Dict):
        self.population = state.pop("population", None)