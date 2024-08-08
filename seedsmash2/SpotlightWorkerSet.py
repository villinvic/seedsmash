from typing import List
import os

from ml_collections import ConfigDict
from polaris.experience.worker_set import WorkerSet
from polaris.experience.environment_worker import EnvWorker
from polaris.policies.policy import Policy
import zmq

class SpotlightWorkerSet(WorkerSet):
    def __init__(
            self,
            config: ConfigDict,
    ):
        super().__init__(config=config)

        pipe_name = "matchmaking_pipe"
        self.matchmaking_window = ...
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
            params_map,
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
                        "params_map": {pid: Policy(
                            name = p.name,
                            options = p.options,
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