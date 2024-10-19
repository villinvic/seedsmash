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

    # before_game_stuck_combs = [
    #         (Character.FOX, Character.ROY, Stage.POKEMON_STADIUM),
    #         (Character.FALCO, Character.CPTFALCON, Stage.YOSHIS_STORY),
    #         (Character.FALCO, Character.GANONDORF, Stage.YOSHIS_STORY),
    #         (Character.FALCO, Character.JIGGLYPUFF, Stage.YOSHIS_STORY),
    #         (Character.MARIO, Character.FALCO, Stage.YOSHIS_STORY),
    #         (Character.MARIO, Character.MARTH, Stage.YOSHIS_STORY),
    #         (Character.MARTH, Character.LUIGI, Stage.POKEMON_STADIUM),
    #         (Character.MARTH, Character.DOC, Stage.POKEMON_STADIUM), # ?
    #         (Character.MARTH, Character.FOX, Stage.POKEMON_STADIUM), # ?
    #         (Character.MARTH, Character.FALCO,Stage.POKEMON_STADIUM), # ?
    #         (Character.MARTH, Character.MARIO, Stage.POKEMON_STADIUM), # ?
    #         (Character.MARTH, Character.MARTH, Stage.POKEMON_STADIUM), # ?
    #         (Character.MARTH, Character.ROY, Stage.POKEMON_STADIUM), # ?
    #         (Character.MARTH, Character.JIGGLYPUFF, Stage.POKEMON_STADIUM), # ?
    #         (Character.MARTH, Character.LINK, Stage.POKEMON_STADIUM), # ?
    #         (Character.MARTH, Character.YLINK, Stage.POKEMON_STADIUM), # ?
    #         (Character.ROY, Character.FOX, Stage.BATTLEFIELD), # ?
    #         (Character.ROY, Character.FALCO, Stage.BATTLEFIELD), # ?
    #         (Character.ROY, Character.MARIO, Stage.BATTLEFIELD), # ?
    #         (Character.ROY,Character.DOC, Stage.BATTLEFIELD), # ?
    #         (Character.ROY, Character.MARTH, Stage.BATTLEFIELD), # ?
    #         (Character.ROY, Character.ROY, Stage.BATTLEFIELD),
    #         (Character.ROY, Character.CPTFALCON, Stage.BATTLEFIELD),
    #         (Character.ROY, Character.GANONDORF, Stage.BATTLEFIELD),
    #         (Character.ROY,Character.JIGGLYPUFF, Stage.BATTLEFIELD),
    #         (Character.ROY, Character.LINK, Stage.BATTLEFIELD),
    #         (Character.ROY, Character.LUIGI, Stage.BATTLEFIELD),
    #         (Character.ROY, Character.YLINK, Stage.BATTLEFIELD),
    #         (Character.CPTFALCON, Character.LUIGI, Stage.YOSHIS_STORY),
    #         (Character.CPTFALCON, Character.ROY, Stage.POKEMON_STADIUM),
    #         (Character.CPTFALCON, Character.JIGGLYPUFF, Stage.FINAL_DESTINATION),
    #         (Character.CPTFALCON, Character.LINK, Stage.DREAMLAND),
    #         (Character.JIGGLYPUFF, Character.CPTFALCON, Stage.BATTLEFIELD),
    #         (Character.JIGGLYPUFF, Character.LINK, Stage.DREAMLAND),
    #         (Character.LINK, Character.FOX, Stage.YOSHIS_STORY),
    #         (Character.LINK, Character.FALCO, Stage.YOSHIS_STORY),
    #         (Character.LINK, Character.DOC, Stage.YOSHIS_STORY),
    #         (Character.LINK, Character.MARIO, Stage.YOSHIS_STORY),
    #         (Character.LINK, Character.MARTH, Stage.YOSHIS_STORY),
    #         (Character.LINK, Character.ROY, Stage.YOSHIS_STORY),
    #         (Character.LINK, Character.CPTFALCON, Stage.YOSHIS_STORY),
    #         (Character.LINK, Character.GANONDORF, Stage.YOSHIS_STORY),
    #         (Character.LINK, Character.JIGGLYPUFF, Stage.YOSHIS_STORY),
    #         (Character.LINK, Character.LINK, Stage.YOSHIS_STORY),
    #         (Character.LINK, Character.LUIGI, Stage.YOSHIS_STORY),
    #         (Character.LINK, Character.YLINK, Stage.YOSHIS_STORY),
    #         (Character.LINK, Character.YLINK, Stage.POKEMON_STADIUM),
    #         (Character.LUIGI, Character.CPTFALCON, Stage.YOSHIS_STORY),
    #         (Character.YLINK, Character.FOX, Stage.YOSHIS_STORY),
    #         (Character.YLINK, Character.FOX, Stage.POKEMON_STADIUM),
    #         (Character.YLINK, Character.FALCO, Stage.YOSHIS_STORY),
    #         (Character.YLINK, Character.MARIO, Stage.YOSHIS_STORY),
    #         (Character.YLINK, Character.DOC, Stage.YOSHIS_STORY),
    #         (Character.YLINK, Character.MARTH, Stage.YOSHIS_STORY),
    #         (Character.YLINK, Character.ROY, Stage.YOSHIS_STORY),
    #         (Character.YLINK, Character.CPTFALCON, Stage.YOSHIS_STORY),
    #         (Character.YLINK, Character.CPTFALCON, Stage.POKEMON_STADIUM),
    #         (Character.YLINK, Character.GANONDORF, Stage.YOSHIS_STORY),
    #         (Character.YLINK, Character.YLINK, Stage.YOSHIS_STORY),
    #         (Character.YLINK, Character.LINK, Stage.YOSHIS_STORY),
    #         (Character.YLINK, Character.JIGGLYPUFF, Stage.YOSHIS_STORY),
    #         (Character.YLINK, Character.CPTFALCON, Stage.BATTLEFIELD),
    #         (Character.YLINK, Character.CPTFALCON, Stage.DREAMLAND),
    #         (Character.YLINK, Character.LINK, Stage.POKEMON_STADIUM),
    #         (Character.YLINK, Character.LUIGI, Stage.POKEMON_STADIUM),
    #         (Character.YLINK, Character.LUIGI, Stage.YOSHIS_STORY),
    #         (Character.MARIO, Character.ROY, Stage.FINAL_DESTINATION),
    #         (Character.JIGGLYPUFF, Character.ROY, Stage.YOSHIS_STORY),
    #         (Character.CPTFALCON, Character.GANONDORF, Stage.YOSHIS_STORY),
    #         (Character.CPTFALCON, Character.JIGGLYPUFF, Stage.YOSHIS_STORY),
    #         (Character.CPTFALCON, Character.MARIO, Stage.YOSHIS_STORY),
    #         (Character.DOC, Character.CPTFALCON, Stage.POKEMON_STADIUM),
    #         (Character.LINK, Character.LINK, Stage.DREAMLAND),
    #         (Character.DOC, Character.GANONDORF, Stage.BATTLEFIELD),
    #         (Character.CPTFALCON, Character.YLINK, Stage.YOSHIS_STORY),
    #         (Character.CPTFALCON, Character.FOX, Stage.YOSHIS_STORY),
    #         (Character.DOC, Character.FOX, Stage.POKEMON_STADIUM),
    #         (Character.FALCO, Character.ROY, Stage.DREAMLAND),
    #         (Character.JIGGLYPUFF, Character.GANONDORF, Stage.POKEMON_STADIUM),
    #         (Character.CPTFALCON, Character.LINK, Stage.YOSHIS_STORY),
    #         (Character.YLINK, Character.LINK, Stage.FINAL_DESTINATION),
    #         (Character.CPTFALCON, Character.MARTH, Stage.YOSHIS_STORY),
    #         (Character.CPTFALCON, Character.ROY, Stage.YOSHIS_STORY),
    #         (Character.JIGGLYPUFF, Character.MARIO, Stage.YOSHIS_STORY),
    #         (Character.CPTFALCON, Character.FALCO, Stage.YOSHIS_STORY),
    #         (Character.JIGGLYPUFF, Character.GANONDORF, Stage.FINAL_DESTINATION),
    #         (Character.MARTH, Character.FALCO, Stage.BATTLEFIELD),
    #         (Character.CPTFALCON, Character.DOC, Stage.YOSHIS_STORY),
    #         (Character.JIGGLYPUFF, Character.LINK, Stage.POKEMON_STADIUM),
    #         (Character.MARTH, Character.LINK, Stage.BATTLEFIELD),
    #         (Character.CPTFALCON, Character.FALCO, Stage.FINAL_DESTINATION),
    #         (Character.FOX, Character.MARTH, Stage.BATTLEFIELD),
    #         (Character.CPTFALCON, Character.MARTH, Stage.FINAL_DESTINATION),
    #         (Character.YLINK, Character.JIGGLYPUFF, Stage.FINAL_DESTINATION),
    #         (Character.YLINK, Character.LUIGI, Stage.FINAL_DESTINATION),
    #
    #     # TODO : fix those combinations where p2 is stuck (apparently ?)
    #         #       the game still goes on, but p2 cannot die, I see controller states and position updating though
    #         (Character.CPTFALCON, Character.CPTFALCON, Stage.FINAL_DESTINATION),
    #         (Character.CPTFALCON, Character.CPTFALCON, Stage.YOSHIS_STORY),
    #         (Character.CPTFALCON, Character.FOX, Stage.FINAL_DESTINATION)
    #
    #     #(Character.CPTFALCON,  Character.FOX, Stage.YOSHIS_STORY),
    #
    # ]
    #
    # ingame_stuck_combs = [
    #     (Character.DOC, Character.JIGGLYPUFF, Stage.FINAL_DESTINATION),
    #     (Character.FALCO, Character.LINK, Stage.YOSHIS_STORY),
    #     (Character.FALCO, Character.YLINK, Stage.YOSHIS_STORY),
    #     (Character.JIGGLYPUFF, Character.FOX, Stage.POKEMON_STADIUM),
    #     (Character.DOC, Character.ROY, Stage.POKEMON_STADIUM), # ?
    #     (Character.DOC, Character.LINK, Stage.POKEMON_STADIUM), # ?
    # ]


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
            #print(self.env_index, self.config["n_eval"])
            if self.env_index <= self.config["n_eval"]:
                self.players[1] = PlayerType.CPU
            else:
                pass
                #print(self.players)

        self.om = ObsBuilder(self.config, self.players)
        self._agent_ids = set(self.om.bot_ports)
        self._debug_port = set([i+1 for i, player_type in enumerate(self.players)
                                if player_type == PlayerType.HUMAN_DEBUG])

        self.discrete_controllers: Dict_T[int, SSBMActionSpace] = {
            p: SSBMActionSpace() for p in self._agent_ids | self._debug_port
        }
        self.observation_space = self.om.gym_specs
        # update for embeddings

        if self.render:
            print(self.om.bot_ports, self.players)

        # self.action_space = Dict({i: Tuple([
        #     Discrete(self.discrete_stick.dim),
        #     Discrete(self.discrete_pad.dim + self.discrete_cstick.dim),
        #     # Box(low=-1.05, high=1.05, shape=(2,))
        #     # Discrete(self.discrete_cstick.dim),
        # ]) for i in self.om.bot_ports})
        # self.action_space = Dict({i:
        #                               MultiDiscrete([self.discrete_stick.dim,
        #                                              self.discrete_pad.dim + self.discrete_cstick.dim])
        #                          for i in self.om.bot_ports})
        self.action_space = self.discrete_controllers[list(self.get_agent_ids())[0]].gym_spec

        self.episode_reward = 0
        self.episode_length = 1
        self.current_matchup = None
        self.current_aids = {
            1: 1,
            2: 2
        }
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

        self.bad_combinations = BadCombinations()
        self.bad_combinations.load_combinations()

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
            for port in self.get_agent_ids()
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

        bad_combination_checking = not self.render

        self.current_aids = {
            1: 1,
            2: 2
        }
        reversed_ports = False

        if options is None:
            stage = np.random.choice(
                self.config["stages"])  # if (options is None or "stage" not in options) else options["stage"]
            chars = np.random.choice(self.config["chars"], len(self.config["players"]))
            while bad_combination_checking and (*chars, stage) in self.bad_combinations:
                stage = np.random.choice(self.config["stages"])
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
            while bad_combination_checking and (*chars, stage) in self.bad_combinations:
                if (*reversed(chars), stage) not in self.bad_combinations:
                    self.current_aids = {
                        1: 2,
                        2: 1
                    }
                    chars = tuple(reversed(chars))
                    reversed_ports = True
                    break
                stage = np.random.choice(self.config["stages"], p=p)
                tries += 1
                if tries > 100:
                    print(f"Can't find fitting stage for {chars}!")
                    return


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

        costumes = [options[p].costume for p in self.get_agent_ids()]
        if len(costumes) > 1:
            if chars[0] == chars[1] and costumes[0] == costumes[1]:
                l = [0,1,2,3]
                if costumes[0] in l:
                    l.remove(costumes[0])
                costumes[1] = np.random.choice(l)

        print(chars)

        while state.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            for port, controller in self.controllers.items():
                curr_port = self.current_aids[port]
                if controller._type != PlayerType.HUMAN:
                    cpu_level = lvl if self.players[curr_port-1] == PlayerType.CPU else 0
                    ready[curr_port-1] = melee.MenuHelper.menu_helper_simple(
                        state,
                        controller,
                        chars[port-1], # we reverse chars when needed
                        stage,
                        costume=0 if curr_port not in options else costumes[port-1],
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

        return self.om.build(self.current_aids), {i: {} for i in self.om.bot_ports}

    def get_next_state_reward(self, every=3) \
            -> Tuple_T[Union[GameState, None], Dict_T[int, StepRewards], bool]:

        collected_rewards = {p: StepRewards() for p in self._agent_ids}
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

                    for port in self.get_agent_ids():
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
            curr_port = self.current_aids[port]
            action = self.discrete_controllers[curr_port][chosen_input_sequence_index]
            self.action_queues[curr_port].push(action)
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


        done = False
        obs = None
        if state is None:
            # We crashed, return dummy stuff
            # Should not be going there
            self.dump_bad_combination_and_raise_error(4)

        elif state.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            # Might have happened if we reached time limit ?
            print("We are here, so this happens sometimes ? This appears to happen with a timeout")

            # Pad if we have delay
            done = self.pad_at_end <= 0
            self.pad_at_end -= 1
            self.reached_end = True
            # Do not give the new state, because it looks weird right now
            # Just update the game frame
            self.om.update(state, game_frame=True)
            obs = self.om.build(self.current_aids)

        elif game_finished:
            done = self.pad_at_end <= 0
            self.pad_at_end -= 1

            if self.reached_end:
                self.om.update(state, game_frame=True)
            else:
                self.om.update(state)
            self.reached_end = True
            obs = self.om.build(self.current_aids)
        else:
            done = False
            self.om.update(state)
            obs = self.om.build(self.current_aids)

        dones = {
            i: done for i in self._agent_ids
        }
        dones["__all__"] = done

        if done and self.config["debug"]:
            print(f"Collected rewards: {rewards}")

        total_rewards = {}
        self.episode_length += 1

        for port in self.get_agent_ids():
            curr_port = self.current_aids[port]
            other_port = 1 + (port % 2)
            opponent_combo_counter = 0 if other_port not in self.reward_functions\
                else self.reward_functions[other_port].combo_counter
            total_rewards[curr_port] = self.reward_functions[port].compute(rewards[port], opponent_combo_counter)
            if done:
                self.episode_metrics[f"agent_{port}"] = self.reward_functions[port].get_metrics(self.episode_length)

        return obs, total_rewards, dones, dones, {}

    def close(self):

        if self.env_index == 1:
            # Flush updated frame data to csv
            self.om.FFD.save()

        self.to_close = True
        if self.console is not None:
            try:
                self.console.stop()
                for c in self.controllers.values():
                    c.disconnect()

                del self.controllers
                del self.console

            except AttributeError as e:
                print()
                print(e)
                print()
            except Exception as e:
                print()
                print('other')
                print(e)
                print()
            self.controllers = None
            self.console = None

    def get_gamestate(self) -> GameState:
        if self.console is None:
            return None
        else:
            return self.console._prev_gamestate

    def get_episode_metrics(self):
        return self.episode_metrics

    def dump_bad_combination_and_raise_error(self, errnum):
        self.bad_combinations.dump_on_error(*self.current_matchup, errnum)
        raise ResetNeeded(f"Dolphin {self.env_index} crashed with {self.current_matchup} [error:{errnum}]")

