import numpy as np
from melee import Action

from seedsmash2.submissions.bot_config import BotConfig


def inject_botconfig(policy_config, botconfig: BotConfig):

    # patience of 0: half-life of 3 seconds
    # patience of 100: half-life of 15 seconds
    halflife = botconfig.reflexion / 100. * (15-3) + 3
    policy_config["discount"] = np.exp(-np.log(2)/(halflife*20))

    if botconfig.creativity == 0:
        creativity_coeff = 0.
    else:
        creativity_coeff = np.exp((botconfig.creativity-50)/20)
    policy_config["action_state_reward_scale"] = creativity_coeff * 6e-3


class ActionStateValues:
    discarded_states = np.array([a.value for a in [
        Action.WALK_SLOW, Action.WALK_FAST, Action.WALK_MIDDLE,
        Action.TUMBLING, Action.TURNING, Action.TURNING_RUN, Action.GRABBED,
        Action.EDGE_TEETERING, Action.EDGE_TEETERING_START, Action.RUN_BRAKE,
        Action.EDGE_ATTACK_QUICK, Action.EDGE_HANGING, Action.EDGE_ATTACK_SLOW,
        Action.EDGE_GETUP_QUICK, Action.EDGE_JUMP_1_QUICK, Action.EDGE_JUMP_2_QUICK,
        Action.EDGE_JUMP_1_SLOW, Action.EDGE_JUMP_2_SLOW, Action.EDGE_GETUP_SLOW,
        Action.EDGE_ROLL_QUICK, Action.EDGE_ROLL_SLOW, Action.DEAD_UP, Action.DEAD_FLY, Action.DEAD_FLY_SPLATTER,
        Action.DEAD_DOWN, Action.DEAD_LEFT, Action.DEAD_RIGHT
    ]])
    wall_tech_states = np.array([a.value for a in [
        Action.WALL_TECH, Action.WALL_TECH_JUMP
    ]])

    def __init__(self, config):
        self.config = config
        self.n_action_states = len(Action)
        self.probs = np.full((self.n_action_states,), dtype=np.float32, fill_value=1/self.n_action_states)

        self.discarded_mask = np.ones((self.n_action_states,), dtype=np.float32)
        self.discarded_mask[self.discarded_states] = 0.

        self.prob_treshold = 1 / (8*self.n_action_states)

    def push_samples(self, action_states):
        u, counts = np.unique(action_states, return_counts=True)
        lr = 5e-2
        count_probs = counts / counts.sum()
        self.probs[:] *= (1-lr)
        self.probs[u] += count_probs * lr
        np.clip(self.probs, 1e-6, 1., out=self.probs)

    def get_rewards(self, action_states):
        mask = self.probs < self.prob_treshold
        logprobs = np.log(self.probs) * np.float32(mask) * self.discarded_mask
        rewards = - logprobs[action_states]
        return rewards * self.config.action_state_reward_scale

    def get_metrics(self):
        return {
            "entropy": - np.sum(np.log(self.probs) * self.probs),
            "min_prob": np.min(self.probs),
            "max_prob": np.max(self.probs)
        }
