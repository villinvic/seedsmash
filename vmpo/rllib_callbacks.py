from collections import defaultdict
from typing import Dict, Tuple, Union, Optional

import cv2
import annoy
from ray.rllib import Policy, SampleBatch, BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import AgentID, PolicyID
from pathlib import Path
import matplotlib.pyplot as plt


import numpy as np

from pkmn_env.red import PkmnRedEnv


class PokemonCallbacks(
    DefaultCallbacks
):
    """
    Log important game stats
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.knn_index = None
        self.width = None
        self.height = None
        self.height_cut = None
        self.similar_frame_dist = 1700.
        self.path = Path("sessions/novelty_frames")
        self.path.mkdir(parents=True, exist_ok=True)
        self.num_distinct_frames = 0
        self.novelty_table = defaultdict(float)
        self.beta = 0.15
        self.multipler = 2-0.997**2

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:

        sub_env: PkmnRedEnv = base_env.get_sub_environments()[env_index]

        if hasattr(episode, "custom_metrics"):

            episode.custom_metrics.update(
                **{metric: sub_env.game_stats[metric][-1] for metric in sub_env.LOGGABLE_VALUES}
            )

            episode.custom_metrics.update(
                **{metric: sum(sub_env.game_stats[metric]) for metric in sub_env.game_stats if "reward" in metric}
            )

    # def on_learn_on_batch(
    #     self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    # ) -> None:
    #
    #         # screen_data_batch = train_batch[SampleBatch.OBS]["screen"]
    #         # total_novelty = 0
    #         # if self.knn_index is None:
    #         #     self.height_cut = 22
    #         #     self.width = screen_data_batch[0].shape[1]//2
    #         #     self.height = (screen_data_batch[0].shape[0]-self.height_cut)//2
    #         #     self.knn_index = annoy.AnnoyIndex(self.width*self.height, "euclidean")
    #         #     self.knn_index.build(n_trees=200, n_jobs=1)
    #         #
    #         #
    #         # idx_delta = 8
    #         # last_added_idx = -idx_delta
    #         #
    #         # for idx, screen in enumerate(screen_data_batch):
    #         #
    #         #     if idx - last_added_idx >= idx_delta:
    #         #         screen = cv2.resize(
    #         #             screen[:-self.height_cut], (self.width, self.height), interpolation=cv2.INTER_AREA)
    #         #
    #         #         screen_flat = np.uint8(screen.flatten())
    #         #
    #         #         if self.num_distinct_frames == 0:
    #         #             # if index is empty add current frame
    #         #             self.knn_index.add_item(self.num_distinct_frames, screen_flat)
    #         #             self.num_distinct_frames += 1
    #         #         else:
    #         #
    #         #             labels, distances = self.knn_index.get_nns_by_vector(
    #         #                 screen_flat, n=1, search_k=200, include_distances=True
    #         #             )
    #         #             distance = distances[0]
    #         #
    #         #             if distance > self.similar_frame_dist:
    #         #                 d = distance - self.similar_frame_dist
    #         #                 self.knn_index.add_item(self.num_distinct_frames, screen_flat)
    #         #                 last_added_idx = idx
    #         #                 self.num_distinct_frames += 1
    #         #                 print(self.num_distinct_frames, d)
    #         #
    #         #                 if self.num_distinct_frames > 200:
    #         #                     screenshot_path = self.path / Path(f"{self.num_distinct_frames}_{int(d)}.jpeg")
    #         #                     cv2.imwrite(screenshot_path.as_posix(), screen[:, :])
    #         #                     train_batch[SampleBatch.REWARDS][idx] += 30.
    #         #                     total_novelty += 1
    #
    #         coordinates = train_batch[SampleBatch.OBS]["coordinates"]
    #
    #         total_novelty = 0
    #         novelty_count_min = np.inf
    #         novelty_count_max = -np.inf
    #         map_grids_visited = len(self.novelty_table)
    #         prev_coords = (-1, -1, -1)
    #         for idx, coordinate in enumerate(coordinates):
    #             coords = self.coord_bins(coordinate)
    #
    #             #if prev_coords != coords:
    #             self.novelty_table[coords] += 1.
    #
    #             count = self.novelty_table[coords]
    #             if novelty_count_min > count:
    #                 novelty_count_min = count
    #             if novelty_count_max < count:
    #                 novelty_count_max = count
    #
    #             if count > 1_000:
    #                 score = 0
    #             else:
    #                 score = 1 / np.sqrt(self.novelty_table[coords])
    #
    #             score *= self.beta * ( self.multipler ** map_grids_visited )
    #             train_batch[SampleBatch.REWARDS][idx] += score
    #             total_novelty += score
    #
    #             prev_coords = coords
    #
    #
    #
    #         result["novelty/batch_novelty"] = total_novelty
    #         result["novelty/novelty_count_max"] = novelty_count_max
    #         result["novelty/novelty_count_min"] = novelty_count_min
    #         result["novelty/map_grids_visited"] = map_grids_visited
    #
    #
    #         #result["novelty/distinct_frames"] = self.num_distinct_frames
    #
    #
    # def coord_bins(self, coords):
    #
    #     x, y, map_id = coords
    #     bx = x - x % 2
    #     by = y - y % 2
    #
    #     return (bx, by, map_id)







