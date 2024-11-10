import numpy as np
from melee import Action
from tensorflow_probability.python.internal.backend.jax import argmax

from seedsmash.bots.bot_config import BotConfig

action_idx = {
    s: i for i, s in enumerate(Action)
}
idx_to_action = {
    i: s for i, s in enumerate(Action)
}

def inject_botconfig(policy_config, botconfig: BotConfig):

    # patience of 0: half-life of 6 seconds
    # patience of 100: half-life of 18 seconds

    halflife = botconfig.reflexion / 100. * (18-5) + 5
    policy_config["discount"] = np.exp(-np.log(2)/(halflife*20))

    creativity_coeff = np.exp((botconfig.creativity-50)/30)
    policy_config["action_state_reward_scale"] = creativity_coeff


class ActionStateCounts:

    discarded_states = np.array([action_idx[a] for a in [
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
        Action.GRAB_PULL, Action.GRAB_PUMMELED, Action.BUMP_WALL, Action.BUMP_CIELING, Action.BOUNCE_WALL,
        Action.BOUNCE_CEILING,
        Action.DEAD_FLY_STAR, Action.THROWN_COPY_STAR, Action.THROWN_KIRBY_STAR, Action.THROWN_KIRBY, Action.THROWN_UP,
        Action.THROWN_BACK, Action.THROWN_DOWN, Action.THROWN_FORWARD, Action.DAMAGE_FLY_HIGH, Action.DAMAGE_HIGH_1,
        Action.DAMAGE_HIGH_2, Action.DAMAGE_HIGH_3, Action.DAMAGE_NEUTRAL_1, Action.DAMAGE_NEUTRAL_2,
        Action.DAMAGE_NEUTRAL_3,
        Action.DAMAGE_LOW_1, Action.DAMAGE_LOW_2, Action.DAMAGE_LOW_3, Action.DAMAGE_AIR_1, Action.DAMAGE_AIR_2,
        Action.DAMAGE_AIR_3,
        Action.DAMAGE_FLY_HIGH, Action.DAMAGE_FLY_NEUTRAL, Action.DAMAGE_FLY_LOW, Action.DAMAGE_FLY_TOP,
        Action.DAMAGE_FLY_ROLL,
        Action.DAMAGE_GROUND, Action.PUMMELED_HIGH, Action.GRABBED_WAIT_HIGH, Action.YOSHI_EGG, Action.KIRBY_YOSHI_EGG,
        Action.THROWN_F_HIGH, Action.THROWN_DOWN_2, Action.THROWN_F_LOW, Action.THROWN_MEWTWO, Action.THROWN_MEWTWO_AIR,
        Action.THROWN_FB, Action.THROWN_FF, Action.THROWN_KIRBY_DRINK_S_SHOT, Action.THROWN_KIRBY_SPIT_S_SHOT,
        Action.THROWN_KOOPA_B, Action.THROWN_KOOPA_F, Action.THROWN_KOOPA_AIR_B, Action.THROWN_KOOPA_END_F,
        Action.THROWN_KOOPA_AIR_F, Action.THROWN_KOOPA_AIR_END_B, Action.THROWN_KOOPA_END_B,
        Action.THROWN_KOOPA_AIR_END_F,
        Action.GRAB_ESCAPE,
        Action.GROUND_GETUP
    ]])


    def __init__(self, config, underused_prob=5e-4, overused_prob=0.16):
        self.config = config
        self.n_action_states = len(action_idx)
        self.probs = np.full((self.n_action_states,), dtype=np.float32, fill_value=1/self.n_action_states)

        self.discarded_action_states = np.ones((self.n_action_states,), dtype=np.float32)
        self.discarded_action_states[self.discarded_states] = 0.

        self.underused_prob = underused_prob
        self.overused_prob = overused_prob

        self.count_max = 100 / self.underused_prob
        self.count_min = 15 / self.underused_prob
        self.curr_count = 0
        self.count_sizes = []
        self.queue = []


    def push_samples(self, action_state_counts):
        # We want to punish moves that are used to often
        # Reward rare states.
        # if we take the unique actions, we can't punish the overused actions
        # same if we only take the first frame where the action appears (example: DK down-b)
        # if we take all actions, we repeatdly reward states that can't be stayed on (e.g. techs).
        # but for those states we can't stay on, we do not reward a lot of frames in consequence, so this should be
        # fine.

        self.queue.append(action_state_counts)
        size = np.sum(action_state_counts)
        self.curr_count += size
        self.count_sizes.append(size)

        if self.curr_count < self.count_min:
            return

        while self.curr_count > self.count_max:
            popped_size = self.count_sizes.pop(0)
            self.queue.pop(0)
            self.curr_count -= popped_size


    def get_values(self):

        if self.curr_count > self.count_min:
            self.probs[:] = np.maximum(np.sum(self.queue, dtype=np.float32, axis=0), 1e-8)
            self.probs /= self.probs.sum()
            self.probs = np.clip(self.probs, 1e-5, 1.)


        # underused_mask =  (self.probs < self.underused_prob)[action_states]
        # overused_mask = (self.probs > self.overused_prob)[action_states]
        logprobs = np.log(self.probs)
        rewards = np.clip((np.log(self.underused_prob) - logprobs) * self.discarded_action_states, 0., 1e2)
        penalty = np.clip((logprobs-np.log(self.overused_prob)) * self.discarded_action_states,0, 1e2)


        # rewards = np.log(np.clip(self.probs + (1 - self.underused_prob), 1e-8, 1)) * self.discarded_action_states
        # penalty = np.log(np.clip(-self.probs + (1 + self.overused_prob), 1e-8, 1)) * self.discarded_action_states
        #
        # rewards = np.square(rewards*8_000.) * 0.05
        # penalty = np.square(penalty)

        scores = rewards ** 3 / 300 - penalty * 0.005

        return ActionStateValues(scores)

    def debug(self):

        logprobs = np.log(self.probs)
        rewards = np.clip((np.log(self.underused_prob) - logprobs) , 0., 1e2)
        penalty = np.clip((logprobs-np.log(self.overused_prob)) ,0, 1e2)

        rewards =rewards ** 3 / 300
        penalty = penalty * 0.005

        return (rewards, penalty)


    def get_metrics(self):
        return {
            "entropy": - np.sum(np.log(self.probs) * self.probs),
            "min_prob": np.min(self.probs),
            "max_prob": np.max(self.probs),
            "count": self.curr_count,
        }


class ActionStateHitCounts(ActionStateCounts):

    # dont want to encourage certain ways to get up.
    discarded_states = np.concatenate([ActionStateCounts.discarded_states, [action_idx[Action.GETUP_ATTACK],
                                                                            action_idx[Action.GROUND_ATTACK_UP]]])

    def __init__(self, config):
        super().__init__(
            config,
            underused_prob=1/25,
            overused_prob=15/25,
        )
        self.count_max = 25 * 128 * 15
        self.count_min = 25 * 128
        self.probs[:] = 3/25

    def get_values(self):

        if self.curr_count > self.count_min:
            self.probs[:] = np.maximum(np.sum(self.queue, dtype=np.float32, axis=0), 1e-8)
            self.probs /= self.probs.sum()
            self.probs = np.clip(self.probs, 1e-4, 1.)


        logprobs = np.log(self.probs)
        rewards = np.clip((np.log(self.underused_prob) - logprobs) * self.discarded_action_states, 0., 1e2)
        penalty = np.clip((logprobs-np.log(self.overused_prob)) * self.discarded_action_states,0, 1e2)

        scores = rewards ** 2 / 200 - penalty * 0.1

        return ActionStateValues(scores, self.__class__.__name__)

    def debug(self):

        logprobs = np.log(self.probs)
        rewards = np.clip((np.log(self.underused_prob) - logprobs) , 0., 1e2)
        penalty = np.clip((logprobs-np.log(self.overused_prob)) ,0, 1e2)

        rewards = rewards ** 2 / 75
        penalty = penalty * 0.1

        return (rewards, penalty)


class ActionStateValues:

    wall_tech_states = np.array([action_idx[a] for a in [
        Action.WALL_TECH, Action.WALL_TECH_JUMP, Action.CEILING_TECH
    ]])

    def __init__(self, values, name=None):
        self.values = values
        self.last_penalty = 0.
        self.last_bonus = 0.
        self.name = self.__class__.__name__ if name is None else name

    def get_rewards(self, action_states, mask=None):

        rewards = self.values[action_states]
        if mask is not None:
            rewards = rewards * np.float32(mask)


        self.last_penalty = np.minimum(np.min(rewards), 0)
        self.last_bonus = np.maximum(np.max(rewards), 0)

        arg = np.argmax(rewards)

        if rewards[arg] > 0:
            if hasattr(self, "name"):
                print(self.name, idx_to_action[action_states[arg]], rewards[arg])
            else:
                # TODO: cleanup
                print(idx_to_action[action_states[arg]], rewards[arg])


        return rewards


    def get_metrics(self):
        return {
            "penalty": self.last_penalty,
            "bonus": self.last_bonus
        }


if __name__ == '__main__':


    asc = ActionStateCounts(None)

    asc.probs = np.logspace(
        -5, 0, len(action_idx)
    )

    rs, pens = asc.debug()
    for p, r, pen in zip(asc.probs, rs, pens):

        #print(f"{p:.5f} ({p*50:.3f}):\t r{r:.4f}, {pen:.4f}")
        print(f"{p:.5f} ({p*(4000):.3f}):\t r{r:.4f}, {pen:.4f}")
