import numpy as np
from melee import Action

from seedsmash2.submissions.bot_config import BotConfig


def inject_botconfig(policy_config, botconfig: BotConfig):

    # patience of 0: half-life of 6 seconds
    # patience of 100: half-life of 18 seconds

    halflife = botconfig.reflexion / 100. * (18-5) + 5
    policy_config["discount"] = np.exp(-np.log(2)/(halflife*20))

    creativity_coeff = np.exp((botconfig.creativity-50)/30)
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
        Action.DEAD_DOWN, Action.DEAD_LEFT, Action.DEAD_RIGHT, Action.STANDING, Action.CROUCHING,
        Action.LYING_GROUND_DOWN, Action.LYING_GROUND_UP, Action.LYING_GROUND_UP_HIT,
        Action.SHIELD_BREAK_FLY, Action.SHIELD_BREAK_FALL, Action.SHIELD_BREAK_TEETER, Action.JUMPING_FORWARD,
        Action.JUMPING_BACKWARD, Action.GRAB_BREAK, Action.SHIELD_BREAK_DOWN_U, Action.SHIELD_BREAK_DOWN_D,
        Action.SHIELD_BREAK_STAND_U, Action.SHIELD_BREAK_STAND_D, Action.GRAB, Action.GRAB_RUNNING,
        Action.GRAB_PULL, Action.GRAB_PUMMELED
    ]])
    wall_tech_states = np.array([a.value for a in [
        Action.WALL_TECH, Action.WALL_TECH_JUMP, Action.CEILING_TECH
    ]])

    def __init__(self, config):
        self.config = config
        self.n_action_states = len(Action)
        self.probs = np.full((self.n_action_states,), dtype=np.float32, fill_value=1/self.n_action_states)

        self.discarded_action_states = np.ones((self.n_action_states,), dtype=np.float32)
        self.discarded_action_states[self.discarded_states] = 0.

        self.underused_prob = 1e-4
        self.overused_prob = 0.3
        self.last_rewards = 0.
        self.last_penalties = 0.
        self.min_taken_prob = 0.

    def push_samples(self, action_states):
        u, counts = np.unique(action_states, return_counts=True)
        lr = 6e-2
        count_probs = counts / counts.sum()
        self.probs[:] *= (1-lr)
        self.probs[u] += count_probs * lr
        np.clip(self.probs, 1e-6, 1., out=self.probs)

    def get_rewards(self, action_states):
        # underused_mask =  (self.probs < self.underused_prob)[action_states]
        # overused_mask = (self.probs > self.overused_prob)[action_states]
        discarded_mask = self.discarded_action_states[action_states]
        #logprobs = np.log(self.probs)
        # rewards = (np.log(self.underused_prob) - logprobs[action_states]) * np.float32(underused_mask) * discarded_mask
        # penalty = (logprobs[action_states]-np.log(self.overused_prob)) * np.float32(overused_mask) * discarded_mask

        taken_action_state_probs = self.probs[action_states]
        rewards = np.log(np.clip(taken_action_state_probs + (1 - self.underused_prob), 1e-8, 1)) * discarded_mask
        penalty = np.log(np.clip(-taken_action_state_probs + (1 + self.overused_prob), 1e-8, 1)) * discarded_mask

        rewards = np.square(rewards*8000.)
        penalty = np.square(penalty)

        self.last_rewards = np.mean(rewards)
        self.last_penalties = np.mean(penalty)
        self.min_taken_prob = np.min(taken_action_state_probs)

        return rewards - penalty

    def get_metrics(self):
        return {
            "entropy": - np.sum(np.log(self.probs) * self.probs),
            "min_taken_prob": self.min_taken_prob,
            "min_prob": np.min(self.probs),
            "max_prob": np.max(self.probs),
            "bonus": self.last_rewards,
            "penatly": self.last_penalties
        }
