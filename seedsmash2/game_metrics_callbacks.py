from typing import Dict, List

from melee import Action
from polaris.experience.episode_callbacks import EpisodeCallbacks
from polaris.experience import SampleBatch
from melee_env.observation_space_v2 import ObsBuilder
import numpy as np

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
    def __init__(self, *args, negative_reward_scale=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        #self.om = observation_manager
        #self.delay = self.om.config["obs"]["delay"]

        # We only care about the second player for now
        # self.self_stock_idx = self.om.get_player_obs_idx("stock", 1)
        # self.opp_stock_idx = self.om.get_player_obs_idx("stock", 2)
        # self.percent_idx = self.om.get_player_obs_idx("percent", 2)
        # self.action_state_idx = self.om.get_player_obs_idx("action", 1)
        #
        # self.self_pos_idx = self.om.get_player_obs_idx("position", 1)
        # self.opp_pos_idx = self.om.get_player_obs_idx("position", 2)

        self.negative_reward_scale = negative_reward_scale

        # self.action_state_history_size = 9000*3
        # self.action_state_history = np.zeros((self.action_state_history_size,), dtype=np.int32)
        # self.idx = 0
        # self.counts = np.zeros((396,), dtype=np.int32)
        # self.discarded_states = np.array([a.value for a in [
        #     Action.WALK_SLOW, Action.WALK_FAST, Action.WALK_MIDDLE,
        #     Action.TUMBLING, Action.TURNING, Action.TURNING_RUN, Action.GRABBED,
        #     Action.EDGE_TEETERING, Action.EDGE_TEETERING_START, Action.RUN_BRAKE,
        #     Action.EDGE_ATTACK_QUICK, Action.EDGE_HANGING, Action.EDGE_ATTACK_SLOW,
        #     Action.EDGE_GETUP_QUICK, Action.EDGE_JUMP_1_QUICK, Action.EDGE_JUMP_2_QUICK,
        #     Action.EDGE_JUMP_1_SLOW, Action.EDGE_JUMP_2_SLOW, Action.EDGE_GETUP_SLOW,
        #     Action.EDGE_ROLL_QUICK, Action.EDGE_ROLL_SLOW
        # ]])
        self.wall_tech_states = np.array([a.value for a in [
            Action.WALL_TECH, Action.WALL_TECH_JUMP
        ]])


    def process_agent(
            self,
            policy,
            batch,
            metrics
    ):

        batch[SampleBatch.REWARD][:] = \
            np.maximum(batch[SampleBatch.REWARD], 0.) + \
            np.minimum(batch[SampleBatch.REWARD], 0.) * self.negative_reward_scale


        # action_states = batch[SampleBatch.OBS][:, self.action_state_idx]
        # next_idx = self.idx + len(action_states)
        #
        # wrap_slice_set(self.action_state_history, self.idx % self.action_state_history_size,
        #                next_idx % self.action_state_history_size, action_states)
        # self.idx = next_idx
        #
        # if self.idx >= self.action_state_history_size:
        #     # More accurate to reward the transition preceding the observation, not the next one.
        #     # We are missing the last bit, but fine
        #     entropy_rewards, as_entropy = compute_entropy_rewards(action_states[1:],
        #                                                           self.action_state_history[character],
        #                                                           self.counts[character], self.discarded_states)
        #     episode.custom_metrics["action_state_rewards"] = np.sum(entropy_rewards)
        #     episode.custom_metrics["action_state_entropy"] = as_entropy
        #
        #     postprocessed_batch[SampleBatch.REWARDS][:-1] += entropy_rewards
        #     # print(f"Episode total entropy rewards and entropy over {episode.length} timesteps: ",
        #     #       episode.custom_metrics["action_state_rewards"],
        #     #       as_entropy)

        metrics[f"{policy.name}/Wall techs"] = np.sum(
            np.int16(
                np.isin(np.int32(batch[SampleBatch.OBS]["categorical"]["action1"]), self.wall_tech_states)
                ))



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
