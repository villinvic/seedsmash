from typing import List
import os

from ml_collections import ConfigDict
from polaris.experience.worker_set import WorkerSet
from polaris.experience.environment_worker import EnvWorker
from polaris.policies.policy import PolicyParams
import zmq
import ray

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

        pipe_name = "matchmaking_pipe"
        self.window = MatchMakingWindowWorker.remote(update_interval_s=1, pipe_name=pipe_name)
        context = zmq.Context()
        try:
            os.unlink(pipe_name)
        except:
            pass
        self.push_pipe = context.socket(zmq.PUSH)
        self.push_pipe.bind(f"ipc://{pipe_name}")

    def push_jobs(
            self,
            jobs: List,
            policy_map,
    ):
        job_refs = []
        hired_workers = set()
        for wid, job in zip(self.available_workers, jobs):
            if wid == 0:
                # send matchmaking info to window:

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
                            for pid, p in policy_map.items()
                        }
                    }
                )


            hired_workers.add(wid)
            job_refs.append(self.workers[wid].run_episode_for.remote(
                job
            ))

        self.available_workers -= hired_workers

        return job_refs