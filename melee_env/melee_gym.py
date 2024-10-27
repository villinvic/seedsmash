import atexit
import time
from collections import defaultdict
from socket import socket
from time import sleep
from typing import Optional, List, Union
from typing import Dict as Dict_T
from typing import Tuple as Tuple_T
from copy import copy, deepcopy

from gymnasium.error import ResetNeeded
from melee import GameState, Console

import melee
import numpy as np
import os
import gymnasium
from gymnasium.spaces import Discrete, Box, Tuple, Dict, MultiDiscrete

from melee.enums import ControllerType, Character, Stage
from melee.slippstream import EnetDisconnected

from melee_env.enums import PlayerType
from melee_env.action_space import ActionSpace, ComboPad, ActionSpaceStick, ActionSpaceCStick, ControllerStateCombo, \
    SimpleActionSpace, InputQueue, ActionControllerInterface, SSBMActionSpace
from melee_env.make_data import BadCombinations
from melee_env.observation_space_v2 import ObsBuilder
from melee_env.rewards import RewardFunction, DeltaFrame, StepRewards
import gc

from polaris.environments import PolarisEnv

from seedsmash2.submissions.bot_config import BotConfig


class SSBM(PolarisEnv):
    env_id = "SSBM"

    PAD_INTERFACE = ActionControllerInterface

    def __init__(self, env_index=-1, **config):

        super().__init__(env_index=env_index, **config)

        self.render = (
            self.config["render"] and self.env_index == self.config["render_idx"]
        )

        sock = socket()
        sock.bind(('', 0))

        self.slippi_port =51441 + self.env_index * 100  #int(sock.getsockname()[1])

        home = os.path.expanduser("~")
        path = home + "/SlippiOnline/%s/dolphin"

        self.path = path % "gui" if self.render else path % "headless"

        self.iso = home + "/isos/melee.iso"

        self.console = None
        self.controllers = None
        self.previously_crashed = False
        self.bad_combinations: BadCombinations = None
        self.ingame_stuck_frozen_counter = 0

        self.pad_at_end = self.config["obs"]["delay"]

        self.reached_end = False

        self.players = copy(self.config["players"])
        if not (PlayerType.HUMAN in self.config["players"] or PlayerType.HUMAN_DEBUG in self.config["players"]):
            if self.env_index <= self.config["n_eval"]:
                self.players[1] = PlayerType.CPU

        self.om = ObsBuilder(self.config, self.players)
        self._agent_ids = set(self.om.bot_ports)
        self._debug_port = set([i+1 for i, player_type in enumerate(self.players)
                                if player_type == PlayerType.HUMAN_DEBUG])

        self.discrete_controllers: Dict_T[int, SSBMActionSpace] = {
            p: SSBMActionSpace() for p in self._agent_ids | self._debug_port
        }
        self.observation_space = self.om.gym_specs

        self.action_space = self.discrete_controllers[list(self.get_agent_ids())[0]].gym_spec

        self.episode_reward = 0
        self.episode_length = 1
        self.current_matchup = None
        self.action_queues: Dict_T[int, InputQueue] = None
        self.reward_functions : Dict_T[int, RewardFunction] = None
        self.delta_frame: DeltaFrame = None

        atexit.register(self.close)

    def step_nones(self) -> Union[GameState, None]:
        max_w = 2000000
        c = 0
        state = None
        while state is None:
            try:
                state = self.console.step()
            except BrokenPipeError as e:
                print(e, self.env_index)
                break
            except EnetDisconnected:
                raise ResetNeeded("EnetDisconnected")
            c += 1
            if c > max_w:
                break
        if state is None:
            self.dump_bad_combination_and_raise_error(1)

        return state


    def make_console_and_controllers(self):

        self.console = melee.Console(
            path=self.path,
            blocking_input=self.config["blocking_input"],
            save_replays=self.config["save_replays"],
            setup_gecko_codes=True,
            slippi_port=self.slippi_port,
            use_exi_inputs=not self.render,
            enable_ffw=not self.render,
            gfx_backend="" if self.render else "Null",
            disable_audio=not self.render,
            polling_mode=True,
            #dual_core=self.render
        )
        self.controllers = {i + 1: ComboPad(console=self.console, port=i + 1, type=(
            ControllerType.GCN_ADAPTER if p_type == PlayerType.HUMAN else ControllerType.STANDARD
        )) for i, p_type in enumerate(self.players)}


    def setup(self):

        connected = False
        tries = 10
        while not connected:
            self.make_console_and_controllers()
            self.console.run(iso_path=self.iso,
                             #exe_name="dolphin-emu" if self.render else "dolphin-emu-headless"
                             #platform=None if self.render else "headless"
                             )

            connected = self.console.connect()
            tries -= 1

            if not connected:

                print("failed to connect with port", self.slippi_port)

                self.console.stop()
                for controller in self.controllers.values():
                    #controller.disconnect()
                    del controller


                del self.console
                del self.controllers

                self.slippi_port += 2000

            if tries == 0:
                break

        for c in self.controllers.values():
            c.connect()

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: dict[int, BotConfig],
    ):
        gc.collect()

        if self.render:
            print(options)

        if self.config["full_reset"] or self.previously_crashed:
            if self.console is not None:
                self.close()
                sleep(2)
                self.previously_crashed = False

        if self.console is None:
            self.setup()

        state = self.step_nones()

        # Reset some attributes
        self.delta_frame = DeltaFrame()
        self.reward_functions = {
            port: RewardFunction(port, options[port])
            for port in self.get_agent_ids() | self._debug_port
        }
        self.episode_metrics = defaultdict(float)
        self.pad_at_end = self.config["obs"]["delay"] + 1
        self.reached_end = False
        self.ingame_stuck_frozen_counter = 0
        self.episode_length = 1
        # For now, with this corny implementation, we need to re-instantiate to reset the pads reliably
        self.discrete_controllers = {
            p: SSBMActionSpace() for p in self._agent_ids | self._debug_port
        }
        self.action_queues = {port: InputQueue() for port in self._agent_ids | self._debug_port}
        self.episode_reward = 0.

        if options is None:
            stage = np.random.choice(
                self.config["stages"])  # if (options is None or "stage" not in options) else options["stage"]
            chars = np.random.choice(self.config["chars"], len(self.config["players"]))
        else:
            chars = ()
            p = {s: 1. for s in self.config["stages"]}
            for port in range(len(self.players)):
                port += 1
                if port in options:
                    chars = chars + (options[port].character,)
                    #p[options[port].preferred_stage] += 1.
                else:
                    chars = chars + (np.random.choice(self.config["chars"]),)

            p = np.array(list(p.values()))
            p /= p.sum()

            stage = np.random.choice(self.config["stages"], p=p)
            tries = 0

        counter = 0
        self.current_matchup = tuple(chars) + (stage,)
        # print(self.current_matchup, reversed_ports)

        while state.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH, None]:
            state = self.step_nones()

        lvl = 9  # TODO: # np.random.randint(1, 10)
        ready = [True for _ in range(4)]
        press_start = False
        if self.env_index == 0:
            # Wait for the matchmaking animation to end
            time.sleep(7)

        costumes = [options[p].costume for p in options]
        if len(costumes) > 1:
            if chars[0] == chars[1] and costumes[0] == costumes[1]:
                l = [0,1,2,3]
                if costumes[0] in l:
                    l.remove(costumes[0])
                costumes[1] = np.random.choice(l)

        while state.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            for port, controller in self.controllers.items():
                if controller._type != PlayerType.HUMAN:
                    cpu_level = lvl if self.players[port-1] == PlayerType.CPU else 0
                    ready[port-1] = melee.MenuHelper.menu_helper_simple(
                        state,
                        controller,
                        chars[port-1], # we reverse chars when needed
                        stage,
                        costume=0 if port not in options else costumes[port-1],
                        autostart=press_start,
                        cpu_level=cpu_level
                    )
            press_start = all(ready)

            state = self.step_nones()

            counter += 1
            if counter > 3800:
                if counter > 3900:
                    self.previously_crashed = True
                    return self.reset(options=options)
                print("STUCK (todo)", state.menu_state, state.frame, state.players[1].cursor.x, state.players[1].cursor.y, self.current_matchup)

        init_frames = 77
        for _ in range(init_frames):
            state = self.step_nones()

        self.game_frame = 0

        self.om.reset()
        self.om.update(state)

        return self.om.build(), {i: {} for i in self.om.bot_ports}

    def get_next_state_reward(self, every=3) \
            -> Tuple_T[Union[GameState, None], Dict_T[int, StepRewards], bool]:

        collected_rewards = {p: StepRewards() for p in self._agent_ids | self._debug_port}
        active_actions = {p: None for p in self._agent_ids | self._debug_port}

        # Are we done ? check if one of the teams are out
        game_is_finished = (self.delta_frame.episode_finished or
                            any([np.int32(p.stock)==0 for p in self.get_gamestate().players.values()]))

        next_state = None

        if not game_is_finished:
            for frame in range(every):
                players = self.get_gamestate().players
                for port in self._agent_ids | self._debug_port:
                    if port in players:
                        next_input = self.action_queues[port].pull(
                            frame == 0,
                            players[port]
                        )
                        if next_input:
                            active_actions[port], curr_sequence = next_input
                            gs = self.get_gamestate()
                            ps = gs.players[port]
                            SSBM.PAD_INTERFACE.send_controller(active_actions[port], self.controllers[port],
                                                               # Additionally pass some info to filter out dumb actions
                                                               gs, ps,
                                                               # If the action is dumb, we want to terminate the current
                                                               # sequence
                                                               curr_sequence)
                            if self.config["debug"]:
                                print(f"Sent input {active_actions[port]} of sequence {curr_sequence} on port {port}.")
                                print(f"curr action_state:{gs.players[port].action}")
                        elif self.config["debug"]:
                            print(f"Waiting a frame before new input on port {port}.")

                old_state = self.get_gamestate()
                next_state = self.step_nones()

                if next_state is not None:
                    for port in self._agent_ids:
                        if port in old_state.players and port in next_state.players:
                            if (old_state.players[port].action_frame > 1
                                    and
                            old_state.players[port].action_frame - next_state.players[port].action_frame == 0):
                                self.ingame_stuck_frozen_counter += 1
                                if self.ingame_stuck_frozen_counter > 200:
                                    self.dump_bad_combination_and_raise_error(2)
                            else:
                                self.ingame_stuck_frozen_counter = 0

                if self.config["debug"]:
                    input("Press [Enter] to step a frame")

                if next_state is None:
                    # Weird state
                    print(f"Stuck here?, crashed with {self.current_matchup}")
                    # force full reset here
                    self.dump_bad_combination_and_raise_error(3)
                else:
                    # Put in our custom frame counter
                    self.game_frame += 1
                    next_state.custom["game_frame"] = self.game_frame

                    # process rewards
                    self.delta_frame.update(next_state)

                    for port in self.reward_functions:
                        self.reward_functions[port].get_frame_rewards(
                            self.delta_frame, active_actions[port], collected_rewards[port]
                        )
                    game_is_finished = self.delta_frame.episode_finished
                # check if we are done

                if game_is_finished:
                    break
        else:
            # The game is done, what do we do ?
            # Pad for delay
            next_state = self.get_gamestate()

        return next_state, collected_rewards, game_is_finished

    def step(self, action_dict):

        for port, chosen_input_sequence_index in action_dict.items():
            action = self.discrete_controllers[port][chosen_input_sequence_index]
            self.action_queues[port].push(action)
        for port in self._debug_port:
            action = None
            while action is None:
                try:
                    print(self.discrete_controllers[port])
                    action = input(f"Choose an action for port {port}: ")
                    if not action:
                        action = self.discrete_controllers[port].RESET_CONTROLLER
                    else:
                        action = self.discrete_controllers[port][int(action)]
                except Exception as e:
                    print(e)

            self.action_queues[port].push(action)

        state, rewards, game_finished = self.get_next_state_reward()

        done = self.handle_state_termination(state, game_finished)
        obs = self.om.build()

        dones = {
            i: done for i in self._agent_ids
        }
        dones["__all__"] = done

        self.episode_length += 1

        total_rewards = self.compute_rewards(rewards)

        if done:
            for port in self.get_agent_ids():
                self.episode_metrics[f"agent_{port}"] = self.reward_functions[port].get_metrics(self.episode_length)
            if self.config["debug"]:
                print(f"Collected rewards: {total_rewards}")

        return obs, total_rewards, dones, dones, {}

    def close(self):
        if self.env_index == 1:
            # Flush updated frame data to csv
            self.om.FFD.save()

        self.to_close = True
        if self.console is None:
            return
        try:
            self.console.stop()
            for c in self.controllers.values():
                c.disconnect()
            del self.controllers
            del self.console
        except Exception as e:
            print(e)
            self.controllers = None
            self.console = None

    def get_gamestate(self) -> GameState:
        if self.console is None:
            return None
        else:
            return self.console._prev_gamestate

    def handle_state_termination(self, state, game_finished):
        if state.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            # TODO: sudden death is bugged, it appears.
            print("We are here, so this happens sometimes ? This appears to happen with a timeout")

            # Pad if we have delay
            done = self.pad_at_end <= 0
            self.pad_at_end -= 1
            self.reached_end = True
            # Do not give the new state, because it is weird looking
            # Just update the game frame
            self.om.update(state, game_frame=True)
        elif game_finished:
            done = self.pad_at_end <= 0
            self.pad_at_end -= 1
            if self.reached_end:
                self.om.update(state, game_frame=True)
            else:
                self.om.update(state)
            self.reached_end = True
        else:
            done = False
            self.om.update(state)

        return done

    def compute_rewards(self, collected_rewards):
        total_rewards = {}
        for port in self.get_agent_ids():
            other_port = 1 + (port % 2)
            opponent_combo_counter = 0 if other_port not in self.reward_functions\
                else self.reward_functions[other_port].combo_counter
            total_rewards[port] = self.reward_functions[port].compute(collected_rewards[port], opponent_combo_counter)
        return  total_rewards

    def get_episode_metrics(self):
        return self.episode_metrics

    def dump_bad_combination_and_raise_error(self, errnum):
        self.bad_combinations.dump_on_error(*self.current_matchup, errnum)
        raise ResetNeeded(f"Dolphin {self.env_index} crashed with {self.current_matchup} [error:{errnum}]")

