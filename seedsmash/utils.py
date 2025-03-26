import numpy as np
from melee import Action, Character

from polaris_melee.observation_space import ObsBuilder
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


    def __init__(
            self,
            preferred_move: Action | None,
            underused_prob=8e-4,
            overused_prob=0.16,
            min_prob=1e-6,
            reward_scale=0.015,
            penalty_scale=0.005
    ):
        preferred_move = preferred_move if preferred_move is not None else Action.DEAD_FLY_STAR # death move, does not count
        self.preferred_move_idx = action_idx[preferred_move]
        self.n_action_states = len(action_idx)
        self.probs = np.full((self.n_action_states,), dtype=np.float32, fill_value=1/self.n_action_states)

        self.action_state_weights = np.ones((self.n_action_states,), dtype=np.float32)
        self.action_state_weights[self.discarded_states] = 0.

        self.action_state_weights[self.preferred_move_idx] = 5

        self.preferred_move_min_logp = np.log(underused_prob * 3)
        self.underused_logp = np.log(underused_prob)
        self.overused_logp = np.log(overused_prob)

        self.min_prob = min_prob
        self.count_min = 1 / min_prob
        self.count_max = self.count_min * 10

        self.reward_scale = reward_scale
        self.penalty_scale = penalty_scale

        self.curr_count = 0
        self.count_sizes = []
        self.queue = []


    def push_samples(self, action_state_counts):

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
            probs = np.maximum(np.sum(self.queue, dtype=np.float32, axis=0), 1e-8)
            probs /= probs.sum()
            probs = np.maximum(probs, self.min_prob)

            logprobs = np.log(probs)
            penalty = np.maximum((logprobs - self.overused_logp) * np.maximum(self.action_state_weights, 1.), 0.)
            # offset the prob of the preferred move
            logprobs[self.preferred_move_idx] += (self.underused_logp - self.preferred_move_min_logp)
            rewards = np.maximum((self.underused_logp - logprobs) * self.action_state_weights, 0.)
            self.probs = probs
        else:
            rewards = penalty = np.zeros_like(self.probs)

        self.last_rewards = rewards * self.reward_scale
        self.last_penalty = penalty * self.penalty_scale


        return ActionStateValues(
            self.last_rewards - self.last_penalty,
            self.__class__.__name__
        )

    def get_top_k_probs(self, k: int):
        argsorted = np.argsort(-self.probs)[:k]

        return {
            "moves": [idx_to_action[idx] for idx in argsorted],
            "probs": self.probs[argsorted]
        }

    def get_metrics(self):
        return {
            "entropy": - np.sum(np.log(self.probs) * self.probs),
            "min_prob": np.min(self.probs),
            "max_prob": np.max(self.probs),
            "count": self.curr_count,
        }


class ActionStateHitCounts(ActionStateCounts):

    def __init__(self, preferred_move: Action, character: Character):

        self.discarded_states = np.array([action_idx[a] for a in [Action.EDGE_ATTACK_SLOW, Action.EDGE_ATTACK_QUICK,
                                                                  Action.GETUP_ATTACK, Action.GROUND_ATTACK_UP] +
                        [attack for attack in Action if (ObsBuilder.FD.has_projectile(character, attack) or
                                                         not ObsBuilder.FD.is_attack(character, attack))]

                        ])

        super().__init__(
            preferred_move,
            underused_prob=1/25,
            overused_prob=10/25,
            min_prob=1e-4,
            reward_scale=0.02,
            penalty_scale=0.12
        )

        self.character = character
        self.action_state_weights[self.preferred_move_idx] = 2
        self.probs[:] = 3/25



class ActionStateValues:

    def __init__(self, values, name=None):
        self.values = values
        self.last_penalty = 0.
        self.last_bonus = 0.
        self.name = self.__class__.__name__ if name is None else name

    def __call__(self, action_state: Action):
        return self.values[action_idx[action_state]]

    def get_rewards(self, action_states, mask=None):

        rewards = self.values[action_states]
        if mask is not None:
            rewards = rewards * np.float32(mask)


        self.last_penalty = np.minimum(np.min(rewards), 0)
        self.last_bonus = np.maximum(np.max(rewards), 0)

        arg = np.argmax(rewards)

        if rewards[arg] > 0:
            print(self.name, idx_to_action[action_states[arg]], rewards[arg])

        return rewards

    def get_metrics(self):
        return {
            "penalty": self.last_penalty,
            "bonus": self.last_bonus
        }


if __name__ == '__main__':


    asc = ActionStateHitCounts(None, Character.CPTFALCON)

    asc.probs = np.logspace(
        -6, 0, len(action_idx)
    )

    rs, pens = asc.debug()
    for p, r, pen in zip(asc.probs, rs, pens):

        #print(f"{p:.5f} ({p*50:.3f}):\t r{r:.4f}, {pen:.4f}")
        print(f"{p:.5f} ({p*(4000):.3f}):\t r{r:.4f}, {pen:.4f}")
