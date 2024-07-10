from functools import partial
from typing import Callable, Optional, Dict, Tuple, Union

from ray.actor import ActorHandle
from ray.rllib import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.typing import EnvCreator, EnvType, PolicyID

import gymnasium as gym

class LightWorkerSet(WorkerSet):

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

        if worker_index == 0:
            # Local worker
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
