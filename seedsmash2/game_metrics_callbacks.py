from collections import defaultdict
from typing import Dict, List

from melee import Action
from polaris.experience.episode_callbacks import EpisodeCallbacks
from polaris.experience import SampleBatch
from melee_env.observation_space_v2 import ObsBuilder
import numpy as np

from seedsmash2.utils import ActionStateValues


def wrap_slice_set(arr, start, stop, values):
    length = arr.shape[0]
    num_values = len(values)
    diff = stop - start
    if diff > 0:
        arr[start:min(start + diff, length)] = values[:min(diff, num_values)]
    else:
        arr[start:] = values[:length-start]
        arr[:stop] = values[length-start:]

def compute_entropy_rewards(
        visited_states, # should be an array of floats because it is a box
        state_history: np.ndarray,
        full_counts,
        dicarded
):
    visited_states = np.int32(visited_states)
    data_amount = len(state_history)

    full_counts[:] = 0
    unique_elements, counts = np.unique(state_history, return_counts=True)
    full_counts[unique_elements] += counts
    probabilities = full_counts / data_amount

    #max_ = np.log(data_amount)
    lower_probablity_threshold = 1e-4
    probabilities_clipped = np.clip(probabilities + (1. - lower_probablity_threshold), 0., 1.)
    nll = - np.log(probabilities_clipped)

    action_state_entropy = - np.sum(probabilities * np.log(probabilities+1e-8))

    # WALK_SLOW = 0x0f
    # WALK_MIDDLE = 0x10
    # WALK_FAST = 0x11
    nll[dicarded] = 0.

    entropy_rewards = np.square(nll[visited_states] * 8000)

    return entropy_rewards, action_state_entropy

class SSBMCallbacks(
    EpisodeCallbacks
):
    def __init__(self, config):
        super().__init__(config)
        self.negative_reward_scale = config.negative_reward_scale
        self.action_state_values = {}

    def process_agent(
            self,
            policy,
            batch,
            metrics
    ):

        batch[SampleBatch.REWARD][:] = \
            np.maximum(batch[SampleBatch.REWARD], 0.) + \
            np.minimum(batch[SampleBatch.REWARD], 0.) * self.negative_reward_scale

        metrics[f"{policy.name}/Wall techs"] = np.sum(
            np.int16(
                np.isin(np.int32(batch[SampleBatch.OBS]["categorical"]["action1"]), ActionStateValues.wall_tech_states)
                ))

        if policy.name not in self.action_state_values:
            self.action_state_values[policy.name] = ActionStateValues(config=self.config)

        action_states = batch[SampleBatch.OBS]["categorical"]["action1"][:, 0]
        self.action_state_values[policy.name].push_samples(
            action_states
        )

        action_state_rewards = self.action_state_values[policy.name].get_rewards(
            action_states
        )
        # TODO this may be wrong if the model uses prev reward !
        batch[SampleBatch.REWARD][:] = action_state_rewards + batch[SampleBatch.REWARD] * policy.policy_config["action_state_reward_scale"]

        for m, v in self.action_state_values[policy.name].get_metrics().items():
            metrics[f"{policy.name}/{m}"] = v


    def on_step(
            self,
            agents_to_policies: Dict,
            actions: Dict,
            observations: Dict,
            next_observations: Dict,
            rewards: Dict,
            dones: Dict,
            infos: Dict,
            metrics: Dict,
    ):
        pass

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
                metrics
            )

    def on_episode_end(
        self,
        agents_to_policies,
        env_metrics,
        metrics,
    ):
        for metric_name, metric in env_metrics.items():
            policy_metric = False
            for aid, policy in agents_to_policies.items():
                # TODO: aid may be in other metrics...
                str_aid = str(aid)
                if str_aid in metric_name:
                    policy_metric = True
                    metrics[metric_name.replace(str_aid, policy.name)] = metric
                    break
            if not policy_metric:
                metrics[metric_name] = metric
