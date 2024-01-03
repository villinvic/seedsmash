from time import sleep
from typing import Optional

from ray.rllib.env.base_env import convert_to_base_env

import melee
import numpy as np
import os
import gymnasium
from gymnasium.spaces import Discrete, Box, Tuple
import atexit

from melee.enums import ControllerType
from melee_env.enums import PlayerType
from melee_env.action_space import ActionSpace, ComboPad, ActionSpaceStick, ActionSpaceCStick
from melee_env.observation_space_v2 import ObsBuilder
from melee_env.rewards import RewardFunction


# TODO multiagent
class SSBM(gymnasium.Env):
    def __init__(self, config):
        self.config = config

        home = os.path.expanduser("~")
        path = home + "/SlippiOnline/%s/squashfs-root/usr/bin"

        self.path = path % "gui" if config["render"] \
            else path % "headless"

        self.iso = os.path.dirname(os.path.abspath(__file__)) + "/melee.iso"

        self.console = None
        self.controllers = None
        # TODO human

        # TODO short_hop for other chars
        self.discrete_pad = ActionSpace()
        self.discrete_stick = ActionSpaceStick()
        self.discrete_cstick = ActionSpaceCStick()

        self.action_queue = []

        self.action_space = Tuple([
            Discrete(self.discrete_pad.dim),
            # Box(low=-1.05, high=1.05, shape=(2,))
            Discrete(self.discrete_stick.dim),
            Discrete(self.discrete_cstick.dim),
        ])
        self.om = ObsBuilder(config)
        self.observation_space = self.om.gym_obs_spec()[1]
        self.state = {1: (np.zeros(self.observation_space.spaces[0].shape, dtype=np.float32),
                      np.zeros(self.observation_space.spaces[1].shape, dtype=np.int8))}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        if self.config["full_reset"]:
            if self.console is not None:
                self.close()
                sleep(5)

        if self.console is None:
            self.console = melee.Console(
                path=self.path,
                blocking_input=self.config["blocking_input"],
                save_replays=self.config["save_replays"],
                setup_gecko_codes=True,
                slippi_port=51441 + self.config.worker_index,
                use_exi_inputs=not self.config["render"],
                enable_ffw=not self.config["render"],
                gfx_backend="" if self.config["render"] else "Null",
                disable_audio=not self.config["render"]
            )
            self.controllers = [ComboPad(console=self.console, port=i + 1, type=(
                ControllerType.GCN_ADAPTER if p == PlayerType.HUMAN else ControllerType.STANDARD
            )) for i, p in enumerate(self.config["players"])]

            self.console.run(iso_path=self.iso)
            self.console.connect()
            for c in self.controllers:
                c.connect()

        state = self.console.step()
        self.reward_function = RewardFunction()

        stage = np.random.choice(self.config["stages"])
        chars = np.random.choice(self.config["chars"], len(self.config["players"]))

        n_start = 70
        counter = 0
        while state.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            for i, c in enumerate(self.controllers):
                if c != PlayerType.HUMAN:
                    if i == 1:
                        cpu_level = 9
                        auto_start = counter > n_start
                    else:
                        cpu_level = 0
                        auto_start = False

                    melee.MenuHelper.menu_helper_simple(
                        state,
                        c,
                        chars[i],
                        stage,
                        autostart=auto_start,
                        cpu_level=cpu_level
                    )
            state = self.console.step()
            counter += 1

        init_frames = 60
        for _ in range(init_frames):
            state = self.console.step()

        return self.om.build_obs(self.state, state)[1]

    def step(self, action):
        r = 0.
        buttons, m_id, c_id = action
        # action_id, stick_id = action

        m = self.discrete_stick[m_id]
        c = self.discrete_cstick[c_id]
        # if m_x > 1.05:
        #     m_x = 0.
        #     r -= 0.1
        # if m_y > 1.05:
        #     m_y = 0.
        #     r -= 0.1

        # print(m, c)

        done = False
        action_chain = self.discrete_pad[buttons]
        if isinstance(action_chain, list):
            for a in action_chain:
                a.send_controller(self.controllers[0], m, c)
                for _ in range(a["duration"]):
                    state = self.console.step()
                    r += self.reward_function.compute(state)

        else:
            action_chain.send_controller(self.controllers[0], m, c)
            for _ in range(action_chain["duration"]):
                state = self.console.step()
                r += self.reward_function.compute(state)

        if state is None:
            print("Stuck ?")

        elif state.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            done = True
        else:
            self.state = self.om.build_obs(self.state, state)

        return self.state[1], r, done, {}
        # return <obs>, <reward: float>, <done: bool>, <info: dict>

    @atexit.register
    def close(self):
        print(self.config.worker_index,
              "STOPING=================================================================================================")

        if self.config.worker_index == 1:
            # Flush updated frame data to csv
            self.om.FFD.override()

        self.to_close = True
        if self.console is not None:
            try:
                self.console.stop()
                for c in self.controllers:
                    c.disconnect()
                del self.controllers
                del self.console
                self.controllers = None
                self.console = None
            except AttributeError as e:
                print()
                print(e)
                print()
            except Exception as e:
                print()
                print('other')
                print(e)
                print()


