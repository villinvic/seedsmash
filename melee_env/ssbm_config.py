import numpy as np
from melee import Stage, Character, ControllerType
from copy import copy
from melee_env.enums import PlayerType
#from seedsmash.ga.naming import Name


# env_config = dict(
#         full_reset=False,
#         n_players=2,
#         blocking_input=True,
#         save_replays=False,
#         render=True,
#         obs=dict(
#             stage=False,
#             ecb=True,
#             character=False,
#             controller_state=True,
#             max_projectiles=3,
#         )
# )

# Does not provide help but it is automated
class FuncConf(dict):

    def __init__(self, **default_args):
        super().__init__(**default_args)

        for arg, value in default_args.items():
            if isinstance(value, bool):
                def func(self):
                    self[arg] = not value
                    return self
            else:
                def func(self, new_value):
                    self[arg] = new_value
                    return self

            setattr(self, arg, func)



class SSBMConfig(dict):

    def __init__(self):
        super().__init__()

        self.update(
            stages=[Stage.FINAL_DESTINATION],
            chars=[Character.FOX],
            full_reset=False,
            players=[PlayerType.BOT, PlayerType.HUMAN],
            blocking_input=True,
            save_replays=False,
            render=False,
            render_idx=0,
            debug=False,
            n_eval=1,
            obs= dict(SSBM_OBS_Config())
        )



    # def get_ma_config(self, version="v3"):
    #
    #     pop_size = 3
    #     pop = [Name() for _ in range(pop_size)]
    #     policy_names = [n.get_safe() for n in pop]
    #
    #     return {"v1": dict(),
    #                       "v2": dict(
    #                           policies={
    #                               "TRAINED": PolicySpec(),
    #                           },
    #                           policy_mapping_fn=lambda p_id, ep, worker, **kwargs: "TRAINED",
    #                           policies_to_train=["TRAINED"],
    #                           policy_states_are_swappable=True,
    #                           capacity=6,
    #                         ),
    #                       "v3": dict(
    #                           policies={
    #                               name.get_safe(): PolicySpec(config={
    #                                   "character": self["chars"][i % len(self["chars"])],
    #                                   "name": name.get()
    #                                   # nametag set in __policy_id
    #                               }) for i, name in enumerate(pop)
    #                           },
    #                           policy_mapping_fn=lambda p_id, ep, worker, **kwargs: np.random.choice(
    #                               policy_names, replace=False),
    #                           policy_states_are_swappable=True,
    #                           policy_map_capacity=6,
    #                       )
    #                       }[version]

    def stages(self, stages):
        cself = copy(self)
        assert isinstance(stages, list) and len(stages) > 0
        cself["stages"] = stages
        return cself

    def chars(self, chars):
        cself = copy(self)
        assert isinstance(chars, list) and len(chars) > 0
        cself["chars"] = chars
        return cself

    def full_reset(self):
        cself = copy(self)
        cself["full_reset"] = True
        return cself

    def players(self, players):
        cself = copy(self)
        assert len(players) > 1
        cself["players"] = players
        return cself

    def n_eval(self, n):
        cself = copy(self)
        cself["n_eval"] = n
        return cself

    def noblock(self):
        cself = copy(self)
        cself["blocking_input"] = False
        return cself

    def save_replays(self):
        cself = copy(self)
        cself["save_replays"] = True
        return cself

    def render(self, render=True, idx=None):
        cself = copy(self)
        cself["render"] = render
        if idx is not None:
            cself["render_idx"] = idx

        return cself

    def set_obs_conf(self, obs_conf: "SSBM_OBS_Config"):
        cself = copy(self)
        cself["obs"] = dict(obs_conf)
        return cself

    def debug(self):
        cself = copy(self)
        cself["debug"] = True
        return cself


class SSBM_OBS_Config(dict):

    def __init__(self):
        super().__init__()

        self.update(
            stage=False,
            ecb=False,
            character=False,
            controller_state=False,
            delay=0,
            max_projectiles=3,
        )

    def stage(self):
        cself = copy(self)
        cself["stage"] = True
        return cself

    def ecb(self):
        cself = copy(self)
        cself["ecb"] = True
        return cself

    def character(self):
        cself = copy(self)
        cself["character"] = True
        return cself

    def controller_state(self):
        cself = copy(self)
        cself["controller_state"] = True
        return cself

    def max_projectiles(self, n_projectiles):
        cself = copy(self)
        assert n_projectiles >= 0
        cself["max_projectiles"] = n_projectiles
        return cself

    def delay(self, n_agent_steps):
        cself = copy(self)
        assert n_agent_steps >= 0

        cself["delay"] = n_agent_steps
        return cself

