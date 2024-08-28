from typing import List, Union, Dict
import os

from matplotlib.style.core import available
from ml_collections import ConfigDict
from polaris.experience import SampleBatch
from polaris.experience.episode import EpisodeMetrics
from polaris.experience.worker_set import WorkerSet, SyncWorkerSet
from polaris.experience.environment_worker import EnvWorker, SyncEnvWorker
from polaris.policies.policy import PolicyParams
import zmq
import ray

from seedsmash2.twitch_bot import TwitchBotPULL
from seedsmash2.visualisation.matchmaking import MatchMakingWindow
from seedsmash2.window_worker import WindowWorker


@ray.remote(num_cpus=1, num_gpus=0)
class MatchMakingWindowWorker(WindowWorker):
    def __init__(self, update_interval_s=5, pipe_name="pipe"):
        super().__init__(window=MatchMakingWindow(), update_interval_s=update_interval_s, pipe_name=pipe_name)

    def update_window(self, dt, **k):
        data = super().update_window(dt, **k)
        if data is not None:
            self.window.animate_matchmaking(**data)


class SpotlightWorkerSet(WorkerSet):
    def __init__(
            self,
            config: ConfigDict,
    ):
        super().__init__(config=config)
        self.pipe_name = "matchmaking_pipe"

    def init_window(self):
        self.window = MatchMakingWindowWorker.remote(update_interval_s=1, pipe_name=self.pipe_name)
        context = zmq.Context()
        try:
            os.unlink(self.pipe_name)
        except:
            pass
        self.push_pipe = context.socket(zmq.PUSH)
        self.push_pipe.bind(f"ipc://{self.pipe_name}")

        self.twitch_requested_matchup_socket = TwitchBotPULL()


    def push_jobs(
            self,
            jobs: List,
            params_map,
    ):
        job_refs = []
        hired_workers = set()
        for wid, job in zip(self.available_workers, jobs):
            if wid == 0:
                # send matchmaking info to window:

                requested_matchups = self.twitch_requested_matchup_socket.pull()
                if len(requested_matchups)> 0:
                    mu = requested_matchups[0]["requested_matchup"]
                    if all([pid in params_map for pid in mu]):
                        print("Playing requested matchup:", mu)
                        job = {
                            i: params_map[pid] for i, pid in enumerate(mu, 1)
                        }
                    else:
                        print("Tried to play requested matchup:", mu)

                self.push_pipe.send_pyobj(
                    {
                        "selected": [
                            p.name for p in job.values()
                        ],
                        "policy_params": {pid: PolicyParams(
                            name=p.name,
                            options=p.options,
                            stats=p.stats,
                        )
                            for pid, p in params_map.items()
                        }
                    }
                )

            hired_workers.add(wid)
            job_refs.append(self.workers[wid].run_episode_for.remote(
                job
            ))

        self.available_workers -= hired_workers

        return job_refs


class SyncSpotlightWorkerSet(SyncWorkerSet):
    def __init__(
            self,
            config: ConfigDict,
    ):
        self.workers = {
            wid: SyncEnvWorker.remote(
                worker_id=wid,
                config=config,
                spectator=wid==0
            )
            for wid in range(config.num_workers)
        }

        self.databag = []
        self.waiting = False
        self.num_workers = len(self.workers) - 1
        self.batch_count = 0

        self.available_workers = set(list(self.workers.keys()))
        self.running_jobs = {}
        self.pipe_name = "matchmaking_pipe"


    def init_window(self):
        self.window = MatchMakingWindowWorker.remote(update_interval_s=1, pipe_name=self.pipe_name)
        context = zmq.Context()
        try:
            os.unlink(self.pipe_name)
        except:
            pass
        self.push_pipe = context.socket(zmq.PUSH)
        self.push_pipe.bind(f"ipc://{self.pipe_name}")

        self.twitch_requested_matchup_socket = TwitchBotPULL()


    def push_jobs(
            self,
            params_map: Dict[str, PolicyParams],
            jobs: List[Dict[str, PolicyParams]],
            push_back=False
    ):
        job_refs = []
        if not push_back and 0 in self.available_workers:
            # send matchmaking info to window:
            job = jobs.pop()
            requested_matchups = self.twitch_requested_matchup_socket.pull()
            if len(requested_matchups) > 0:
                mu = requested_matchups[0]["requested_matchup"]
                if all([pid in params_map for pid in mu]):
                    print("Playing requested matchup:", mu)
                    job = {
                        i: params_map[pid] for i, pid in enumerate(mu, 1)
                    }
                else:
                    print("Tried to play requested matchup:", mu)

            self.push_pipe.send_pyobj(
                {
                    "selected": [
                        p.name for p in job.values()
                    ],
                    "policy_params": {pid: PolicyParams(
                        name=p.name,
                        options=p.options,
                        stats=p.stats,
                    )
                        for pid, p in params_map.items()
                    }
                }
            )

            job_refs.append(self.workers[0].get_next_batch_for.remote(
                job
            ))
            self.available_workers -= {0}


        if not push_back and self.waiting:
            return job_refs

        available_workers = self.available_workers - {0}
        print(available_workers, self.running_jobs.keys())
        job_refs = []

        for wid, job in self.running_jobs.items():
            # wid == 0 should not return mid episode !
            if wid != 0:
                for aid, params in job.items():
                    job[aid] = params_map[params.name]
                job_refs.append(self.workers[wid].get_next_batch_for.remote(
                    job
                ))

        hired_workers = set()
        for wid, job in zip(available_workers, jobs):
            hired_workers.add(wid)
            self.running_jobs[wid] = job
            for job in jo
            job_refs.append(self.workers[wid].get_next_batch_for.remote(
                job
            ))
        self.available_workers -= hired_workers

        return job_refs