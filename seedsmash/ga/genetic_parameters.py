from melee_env.rewards import SSBMRewards
from ray.tune import uniform, loguniform, choice, randint, qrandint, lograndint
import numpy as np


# TODO :
#   For neural networks:    -parent distillation (copy parent A, distill toward B with some weight factor)
#                           - mutation ? Probably useless


class Evolvable:
    def __init__(self, sampler, *sampler_args):
        self.sampler = sampler
        self.args = sampler_args
        self._value = None
        self.sample()


    def sample(self):
        self.set(self.sampler(*self.args).sample())
        return self.get()

    def get(self):
        return self._value

    def set(self, v):
        self._value = v

    def __repr__(self):
        return f"Evolvable({self._value}, {self.sampler.__name__}{self.args})"


class Mutation:
    def __init__(self, strength=0.2, chance=0.1, resample_chance=0.1):
        self.strength = strength
        self.chance = chance
        self.resample_chance = resample_chance

    def __call__(self, param: Evolvable):
        if np.random.random() < self.resample_chance:
            param.sample()
        else:
            if param.sampler in [uniform, loguniform]:
                self.mutate_float(param)
            elif param.sampler in [randint, qrandint, lograndint]:
                self.mutate_integer(param)
            else:
                pass

    def mutate_float(self, evolvable):
        new_value = (
            evolvable.get()
            if np.random.random() > self.chance
            else (
                    evolvable.get() * np.random.choice([1 - self.strength, 1 + self.strength])
            ))
        evolvable.set(np.clip(new_value, *evolvable.args[:2]))

    def mutate_integer(self, evolvable):
        new_value = (
            evolvable.get()
            if np.random.random() > self.chance
            else (
                    evolvable.get() + np.random.choice([-1, +1])
            ))
        evolvable.set(np.clip(new_value, *evolvable.args[:2]))


class Crossover:
    def __init__(self, strength=0.2, chance=0.1, resample_chance=0.1):
        self.strength = strength
        self.chance = chance
        self.resample_chance = resample_chance

    def __call__(self, param1: Evolvable, param2: Evolvable):
        assert (param2.sampler == param1.sampler and param1.args == param2.args), (param1, param2)
        if param1.sampler in [uniform, loguniform]:
            self.crossover_float(param)
        elif param1.sampler in [randint, qrandint, lograndint]:
            self.crossover_(param)
        else:
            param.sample()

    def crossover_float(self, evolvable1, evolvable2):
        new_value = (
            evolvable.get()
            if np.random.random() > self.chance
            else (
                    evolvable.get() * np.random.choice([1 - self.strength, 1 + self.strength])
            ))
        evolvable.set(np.clip(new_value, *evolvable.args[:2]))

    def mutate_integer(self, evolvable):
        new_value = (
            evolvable.get()
            if np.random.random() > self.chance
            else (
                    evolvable.get() + np.random.choice([-1, +1])
            ))
        evolvable.set(np.clip(new_value, *evolvable.args[:2]))




class SSBMRewardShaping:


    def __init__(
            self,
            win_reward_scale=0,
            damage_reward_scale=uniform(0.002, 0.02),
            off_stage_multiplier=uniform(1, 2.5),
            distance_reward_scale=loguniform(1e-3, 0.001)
    ):
        pass



    def generate(self):
        return SSBMRewards(**self.sample())


if __name__ == '__main__':

    import pickle

    x = Evolvable(randint, 0, 5)

    print(x)

    with open("tmp.txt", "wb") as f:

        pickle.dump(x , f)
