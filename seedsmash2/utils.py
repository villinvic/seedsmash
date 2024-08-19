import numpy as np
from melee import Action

from seedsmash2.submissions.bot_config import BotConfig


def inject_botconfig(policy_config, botconfig: BotConfig):

    # patience of 0: half-life of 5 seconds
    # patience of 100: half-life of 18 seconds
    halflife = botconfig.reflexion / 100. * (18-5) + 5
    policy_config["discount"] = np.exp(-np.log(2)/(halflife*20))

    creativity_coeff = np.exp((botconfig.creativity-50)/25)
    policy_config["action_state_reward_scale"] = creativity_coeff


class ActionStateValues:
    discarded_states = np.array([a.value for a in [
        Action.WALK_SLOW, Action.WALK_FAST, Action.WALK_MIDDLE,
        Action.TUMBLING, Action.TURNING, Action.TURNING_RUN, Action.GRABBED,
        Action.EDGE_TEETERING, Action.EDGE_TEETERING_START, Action.RUN_BRAKE,
        Action.EDGE_ATTACK_QUICK, Action.EDGE_HANGING, Action.EDGE_ATTACK_SLOW,
        Action.EDGE_GETUP_QUICK, Action.EDGE_JUMP_1_QUICK, Action.EDGE_JUMP_2_QUICK,
        Action.EDGE_JUMP_1_SLOW, Action.EDGE_JUMP_2_SLOW, Action.EDGE_GETUP_SLOW,
        Action.EDGE_ROLL_QUICK, Action.EDGE_ROLL_SLOW, Action.DEAD_UP, Action.DEAD_FLY, Action.DEAD_FLY_SPLATTER,
        Action.DEAD_DOWN, Action.DEAD_LEFT, Action.DEAD_RIGHT, Action.STANDING, Action.CROUCHING
    ]])
    wall_tech_states = np.array([a.value for a in [
        Action.WALL_TECH, Action.WALL_TECH_JUMP
    ]])

    def __init__(self, config):
        self.config = config
        self.n_action_states = len(Action)
        self.probs = np.full((self.n_action_states,), dtype=np.float32, fill_value=1/self.n_action_states)

        self.discarded_action_states = np.ones((self.n_action_states,), dtype=np.float32)
        self.discarded_action_states[self.discarded_states] = 0.

        self.underused_prob = 1 / (60*self.n_action_states)
        self.overused_prob = 0.33

    def push_samples(self, action_states):
        u, counts = np.unique(action_states, return_counts=True)
        lr = 6e-2
        count_probs = counts / counts.sum()
        self.probs[:] *= (1-lr)
        self.probs[u] += count_probs * lr
        np.clip(self.probs, 1e-6, 1., out=self.probs)

    def get_rewards(self, action_states):
        underused_mask =  (self.probs < self.underused_prob)[action_states]
        overused_mask = (self.probs > self.overused_prob)[action_states]
        discarded_mask = self.discarded_action_states[action_states]
        logprobs = np.log(self.probs)
        rewards = (np.log(self.underused_prob) - logprobs[action_states]) * np.float32(underused_mask) * discarded_mask
        penalty = (logprobs[action_states]-np.log(self.overused_prob)) * np.float32(overused_mask) * discarded_mask
        return rewards-penalty

    def get_metrics(self):
        return {
            "entropy": - np.sum(np.log(self.probs) * self.probs),
            "min_prob": np.min(self.probs[self.probs > 1e-6]),
            "max_prob": np.max(self.probs)
        }
