from collections import defaultdict
from typing import Dict, List

from melee import Action
from polaris.experience.episode_callbacks import EpisodeCallbacks
from polaris.experience import SampleBatch
from tensorflow.python.keras.backend import dtype

from polaris_melee.observation_space import ObsBuilder, idx_to_action
import numpy as np

from seedsmash.utils import ActionStateCounts, ActionStateValues


class SSBMCallbacks(
    EpisodeCallbacks
):
    def __init__(self, config):
        super().__init__(config)
        self.negative_reward_scale = config.negative_reward_scale
        # self.action_state_counts = defaultdict(lambda: np.zeros((len(Action),), dtype=np.int32))
        # self.action_state_hit_counts = defaultdict(lambda: np.zeros((len(Action),), dtype=np.int32))
        #
        # self.landed_hits = defaultdict(lambda: {
        #     action.name: 0. for action in Action
        # })

    def process_agent(
            self,
            policy,
            batch,
            metrics,
    ):
        defensivity = 1. - policy.options.stats.aggressivity / 100

        neg_scale_min = 0.93
        neg_scale_max = 0.99
        neg_scale = defensivity * (neg_scale_max - neg_scale_min) + neg_scale_min

        batch[SampleBatch.REWARD][:] = \
            np.maximum(batch[SampleBatch.REWARD], 0.) + \
            np.minimum(batch[SampleBatch.REWARD], 0.) * neg_scale

    #     tech_metric_name = f"{policy.name}/Wall techs"
    #     num_techs = np.sum(
    #         np.int16(
    #             np.isin(np.int32(batch[SampleBatch.OBS]["categorical"]["action1"]), ActionStateValues.wall_tech_states)
    #             ))
    #     if tech_metric_name not in metrics:
    #         metrics[f"{policy.name}/Wall techs"] = num_techs
    #     else:
    #         metrics[f"{policy.name}/Wall techs"] += num_techs
    #
    #
    #     action_states = batch[SampleBatch.OBS]["categorical"]["action1"][:, 0]
    #
    #     dist = np.sqrt(
    #         np.sum(np.square((batch[SampleBatch.OBS]["continuous"]["position1"][:-1]
    #                          - batch[SampleBatch.OBS]["continuous"]["position2"][:-1])/ObsBuilder.POS_SCALE), axis=-1)
    #     )
    #     dealt_damage_steps = batch[SampleBatch.OBS]["continuous"]["percent2"][1:, 0] - \
    #                          batch[SampleBatch.OBS]["continuous"]["percent2"][:-1, 0] > 1e-3
    #     valid_hit_timesteps = np.float32(np.logical_and(dist < 50, dealt_damage_steps))
    #
    #     as_bonus = 0.
    #
    #     if "action_state_values" in policy.stats:
    #         action_state_rewards = policy.stats["action_state_values"].get_rewards(
    #             action_states[1:]
    #         )
    #         as_bonus += action_state_rewards
    #         total_action_state_rewards = np.sum(action_state_rewards)
    #
    #         as_bonus_metric_name = f"{policy.name}/action_state_bonus"
    #         if as_bonus_metric_name not in metrics:
    #             metrics[as_bonus_metric_name] = total_action_state_rewards
    #         else:
    #             metrics[as_bonus_metric_name] += total_action_state_rewards
    #
    #     if "action_state_hit_values" in policy.stats:
    #
    #         action_state_hit_rewards = policy.stats["action_state_hit_values"].get_rewards(
    #             action_states[:-1], mask=valid_hit_timesteps
    #         )
    #
    #         as_bonus += action_state_hit_rewards
    #
    #         total_action_state_hit_rewards = np.sum(action_state_hit_rewards)
    #
    #         as_hit_bonus_metric_name = f"{policy.name}/action_state_hit_bonus"
    #         if as_hit_bonus_metric_name not in metrics:
    #
    #             metrics[as_hit_bonus_metric_name] = total_action_state_hit_rewards
    #         else:
    #             metrics[as_hit_bonus_metric_name] += total_action_state_hit_rewards
    #
    #
    #     batch[SampleBatch.REWARD][:-1] += as_bonus * policy.policy_config["action_state_reward_scale"]
    #
    #     not_dones = np.logical_not(batch[SampleBatch.DONE])
    #     uniques, counts = np.unique(batch[SampleBatch.OBS]["categorical"]["action1"][not_dones, 0], return_counts=True)
    #     self.action_state_counts[policy.name][uniques] += counts
    #
    #     hit_uniques, hit_counts = np.unique(batch[SampleBatch.OBS]["categorical"]["action1"][:-1][
    #                                             np.logical_and(not_dones[:-1], valid_hit_timesteps)
    #                                             , 0], return_counts=True)
    #
    #     self.action_state_hit_counts[policy.name][hit_uniques] += hit_counts
    #
    #
    # def on_step(
    #         self,
    #         agents_to_policies: Dict,
    #         actions: Dict,
    #         observations: Dict,
    #         next_observations: Dict,
    #         rewards: Dict,
    #         dones: Dict,
    #         infos: Dict,
    #         metrics: Dict,
    # ):
    #     for aid, policy in agents_to_policies.items():
    #         other_aid = aid % 2 + 1
    #         inflicted = next_observations[other_aid]["continuous"]["percent1"][0] - observations[other_aid]["continuous"]["percent1"][0]
    #         if inflicted > 0:
    #             curr_action = idx_to_action[observations[aid]["categorical"]["action1"][0]]
    #             self.landed_hits[policy.name][curr_action.name] += inflicted * 100


    def on_trajectory_end(
            self,
            agents_to_policies: Dict,
            sample_batches: List,
            metrics: Dict,
    ):

        for batch in sample_batches:
            policy = agents_to_policies[batch[SampleBatch.AGENT_ID][0]]
            self.process_agent(
                policy,
                batch,
                metrics,
            )

    # def on_episode_end(
    #     self,
    #     agents_to_policies,
    #     env_metrics,
    #     metrics,
    # ):
    #     for metric_name, metric in env_metrics.items():
    #         policy_metric = False
    #         for aid, policy in agents_to_policies.items():
    #             # TODO: aid may be in other metrics...
    #             str_aid = str(aid)
    #             if str_aid in metric_name:
    #                 policy_metric = True
    #                 metrics[metric_name.replace(str_aid, policy.name)] = metric
    #                 break
    #         if not policy_metric:
    #             metrics[metric_name] = metric
    #
    #     for policy in agents_to_policies.values():
    #         if f"{policy.name}/to_pop/action_states_counts" not in metrics:
    #             metrics[f"{policy.name}/to_pop/action_states_counts"] = self.action_state_counts[policy.name].copy()
    #             metrics[f"{policy.name}/to_pop/action_states_hit_counts"] = self.action_state_hit_counts[policy.name].copy()
    #             # TODO: del this ?
    #             self.action_state_counts[policy.name][:] = 0
    #             self.action_state_hit_counts[policy.name][:] = 0
    #
    #
    #         # if ditto, sends one with double count
    #         landed_hits_metric = f"{policy.name}/landed_hits"
    #         if landed_hits_metric in metrics:
    #             metrics[landed_hits_metric] = np.array({
    #                 a: v/2 for a, v in metrics[landed_hits_metric].item().items()
    #             })
    #         else:
    #             metrics[f"{policy.name}/landed_hits"] = np.array(self.landed_hits[policy.name])
    #             del self.landed_hits[policy.name]

