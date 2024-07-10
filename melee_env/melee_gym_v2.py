import atexit
from collections import defaultdict
from time import sleep
from typing import Optional, List, Union
from typing import Dict as Dict_T
from typing import Tuple as Tuple_T
from copy import copy

from melee import GameState, Console

import melee
import numpy as np
import os
import gymnasium
from gymnasium.spaces import Discrete, Box, Tuple, Dict, MultiDiscrete

from melee.enums import ControllerType, Character, Stage
from melee_env.enums import PlayerType
from melee_env.action_space import ActionSpace, ComboPad, ActionSpaceStick, ActionSpaceCStick, ControllerStateCombo, \
    SimpleActionSpace, InputQueue, ActionControllerInterface, SSBMActionSpace
from melee_env.observation_space_v2 import ObsBuilder
from melee_env.rewards import RewardFunction, SSBMRewards
import gc

from polaris.environments import PolarisEnv

class SSBM(PolarisEnv):
    env_id = "SSBM"

    before_game_stuck_combs = [
            (Character.FOX, Character.ROY, Stage.POKEMON_STADIUM),
            (Character.FALCO, Character.CPTFALCON, Stage.YOSHIS_STORY),
            (Character.FALCO, Character.GANONDORF, Stage.YOSHIS_STORY),
            (Character.FALCO, Character.JIGGLYPUFF, Stage.YOSHIS_STORY),
            (Character.MARIO, Character.FALCO, Stage.YOSHIS_STORY),
            (Character.MARIO, Character.MARTH, Stage.YOSHIS_STORY),
            (Character.MARTH, Character.LUIGI, Stage.POKEMON_STADIUM),
            (Character.MARTH, Character.DOC, Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.FOX, Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.FALCO,Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.MARIO, Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.MARTH, Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.ROY, Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.JIGGLYPUFF, Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.LINK, Stage.POKEMON_STADIUM), # ?
            (Character.MARTH, Character.YLINK, Stage.POKEMON_STADIUM), # ?
            (Character.ROY, Character.FOX, Stage.BATTLEFIELD), # ?
            (Character.ROY, Character.FALCO, Stage.BATTLEFIELD), # ?
            (Character.ROY, Character.MARIO, Stage.BATTLEFIELD), # ?
            (Character.ROY,Character.DOC, Stage.BATTLEFIELD), # ?
            (Character.ROY, Character.MARTH, Stage.BATTLEFIELD), # ?
            (Character.ROY, Character.ROY, Stage.BATTLEFIELD),
            (Character.ROY, Character.CPTFALCON, Stage.BATTLEFIELD),
            (Character.ROY, Character.GANONDORF, Stage.BATTLEFIELD),
            (Character.ROY,Character.JIGGLYPUFF, Stage.BATTLEFIELD),
            (Character.ROY, Character.LINK, Stage.BATTLEFIELD),
            (Character.ROY, Character.LUIGI, Stage.BATTLEFIELD),
            (Character.ROY, Character.YLINK, Stage.BATTLEFIELD),
            (Character.CPTFALCON, Character.LUIGI, Stage.YOSHIS_STORY),
            (Character.CPTFALCON, Character.ROY, Stage.POKEMON_STADIUM),
            (Character.CPTFALCON, Character.JIGGLYPUFF, Stage.FINAL_DESTINATION),
            (Character.CPTFALCON, Character.LINK, Stage.DREAMLAND),
            (Character.JIGGLYPUFF, Character.CPTFALCON, Stage.BATTLEFIELD),
            (Character.JIGGLYPUFF, Character.LINK, Stage.DREAMLAND),
            (Character.LINK, Character.FOX, Stage.YOSHIS_STORY),
            (Character.LINK, Character.FALCO, Stage.YOSHIS_STORY),
            (Character.LINK, Character.DOC, Stage.YOSHIS_STORY),
            (Character.LINK, Character.MARIO, Stage.YOSHIS_STORY),
            (Character.LINK, Character.MARTH, Stage.YOSHIS_STORY),
            (Character.LINK, Character.ROY, Stage.YOSHIS_STORY),
            (Character.LINK, Character.CPTFALCON, Stage.YOSHIS_STORY),
            (Character.LINK, Character.GANONDORF, Stage.YOSHIS_STORY),
            (Character.LINK, Character.JIGGLYPUFF, Stage.YOSHIS_STORY),
            (Character.LINK, Character.LINK, Stage.YOSHIS_STORY),
            (Character.LINK, Character.LUIGI, Stage.YOSHIS_STORY),
            (Character.LINK, Character.YLINK, Stage.YOSHIS_STORY),
            (Character.LINK, Character.YLINK, Stage.POKEMON_STADIUM),
            (Character.LUIGI, Character.CPTFALCON, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.FOX, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.FOX, Stage.POKEMON_STADIUM),
            (Character.YLINK, Character.FALCO, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.MARIO, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.DOC, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.MARTH, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.ROY, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.CPTFALCON, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.CPTFALCON, Stage.POKEMON_STADIUM),
            (Character.YLINK, Character.GANONDORF, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.YLINK, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.LINK, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.JIGGLYPUFF, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.CPTFALCON, Stage.BATTLEFIELD),
            (Character.YLINK, Character.CPTFALCON, Stage.DREAMLAND),
            (Character.YLINK, Character.LINK, Stage.POKEMON_STADIUM),
            (Character.YLINK, Character.LUIGI, Stage.POKEMON_STADIUM),
            (Character.YLINK, Character.LUIGI, Stage.YOSHIS_STORY),
            (Character.MARIO, Character.ROY, Stage.FINAL_DESTINATION),
            (Character.JIGGLYPUFF, Character.ROY, Stage.YOSHIS_STORY),
            (Character.CPTFALCON, Character.GANONDORF, Stage.YOSHIS_STORY),
            (Character.CPTFALCON, Character.JIGGLYPUFF, Stage.YOSHIS_STORY),
            (Character.CPTFALCON, Character.MARIO, Stage.YOSHIS_STORY),
            (Character.DOC, Character.CPTFALCON, Stage.POKEMON_STADIUM),
            (Character.LINK, Character.LINK, Stage.DREAMLAND),
            (Character.DOC, Character.GANONDORF, Stage.BATTLEFIELD),
            (Character.CPTFALCON, Character.YLINK, Stage.YOSHIS_STORY),
            (Character.CPTFALCON, Character.FOX, Stage.YOSHIS_STORY),
            (Character.DOC, Character.FOX, Stage.POKEMON_STADIUM),
            (Character.FALCO, Character.ROY, Stage.DREAMLAND),
            (Character.JIGGLYPUFF, Character.GANONDORF, Stage.POKEMON_STADIUM),
            (Character.CPTFALCON, Character.LINK, Stage.YOSHIS_STORY),
            (Character.YLINK, Character.LINK, Stage.FINAL_DESTINATION),
            (Character.CPTFALCON, Character.MARTH, Stage.YOSHIS_STORY),
            (Character.CPTFALCON, Character.ROY, Stage.YOSHIS_STORY),
            (Character.JIGGLYPUFF, Character.MARIO, Stage.YOSHIS_STORY),
            (Character.CPTFALCON, Character.FALCO, Stage.YOSHIS_STORY),
            (Character.JIGGLYPUFF, Character.GANONDORF, Stage.FINAL_DESTINATION),
            (Character.MARTH, Character.FALCO, Stage.BATTLEFIELD),
            (Character.CPTFALCON, Character.DOC, Stage.YOSHIS_STORY),
            (Character.JIGGLYPUFF, Character.LINK, Stage.POKEMON_STADIUM),
            (Character.MARTH, Character.LINK, Stage.BATTLEFIELD),

            # TODO : fix those combinations where p2 is stuck (apparently ?)
            #       the game still goes on, but p2 cannot die, I see controller states and position updating though
            (Character.CPTFALCON, Character.CPTFALCON, Stage.FINAL_DESTINATION),

        #(Character.CPTFALCON,  Character.FOX, Stage.YOSHIS_STORY),

    ]

    ingame_stuck_combs = [
        (Character.DOC, Character.JIGGLYPUFF, Stage.FINAL_DESTINATION),
        (Character.FALCO, Character.LINK, Stage.YOSHIS_STORY),
        (Character.FALCO, Character.YLINK, Stage.YOSHIS_STORY),
        (Character.JIGGLYPUFF, Character.FOX, Stage.POKEMON_STADIUM),
        (Character.DOC, Character.ROY, Stage.POKEMON_STADIUM), # ?
        (Character.DOC, Character.LINK, Stage.POKEMON_STADIUM), # ?
    ]

    PAD_INTERFACE = ActionControllerInterface


    def __init__(self, env_index=-1, **config):

        super().__init__(env_index=env_index, **config)

        print(self.config)

        self.render = (
            self.config["render"] and self.env_index == self.config["render_idx"]
        )

        self.slippi_port = 51441 + self.env_index

        home = os.path.expanduser("~")
        path = home + "/SlippiOnline/debug/%s/dolphin"

        self.path = path % "gui"

        self.iso = home + "/isos/melee.iso"

        self.console = None
        self.controllers = None
        self.previously_crashed = False
        self.ingame_stuck_frozen_counter = 0

        self.pad_at_end = self.config["obs"]["delay"]

        self.reached_end = False

        self.players = copy(self.config["players"])
        if not (PlayerType.HUMAN in self.config["players"] or PlayerType.HUMAN_DEBUG in self.config["players"]):
            print(self.env_index, self.config["n_eval"])
            if self.env_index <= self.config["n_eval"]:
                self.players[1] = PlayerType.CPU
            else:
                print(self.players)

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
        self.current_matchup = None
        self.action_queues: Dict_T[int, InputQueue] = None
        self.reward_function: RewardFunction = None
        self.state = {aid: self.observation_space.sample() for aid in self.get_agent_ids()}

        atexit.register(self.close)

    def step_nones(self, reset_if_stuck=False) -> Union[GameState, None]:
        max_w = 2000000
        c = 0
        state = None
        while state is None:
            try:
                state = self.console.step()
            except BrokenPipeError as e:
                print(e)
                break
            c += 1
            if c > max_w:
                ## This causes full crash
                print("we are stuck")
                break
        if state is None and reset_if_stuck:
            print("Closing and restarting...")
            self.close()
            sleep(2)
            self.setup()
            print("Restarted.")

            return self.step_nones(reset_if_stuck=True)

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
                             platform=None if self.render else "headless"
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
            options: Optional[dict] = None,
    ):
        gc.collect()

        if self.config["full_reset"] or self.previously_crashed:
            if self.console is not None:
                self.close()
                sleep(2)
                self.previously_crashed = False


        if self.console is None:
            #print("SETUP")
            self.setup()
        else:
            pass
            #print("no setup needed")

        #print("INIT")
        state = self.step_nones(reset_if_stuck=True)

        # Reset some attributes
        self.reward_function = RewardFunction()
        self.episode_metrics = defaultdict(float)
        self.pad_at_end = self.config["obs"]["delay"] + 1
        self.reached_end = False
        self.ingame_stuck_frozen_counter = 0
        # For now, with this corny implementation, we need to re-instantiate to reset the pads reliably
        self.discrete_controllers = {
            p: SSBMActionSpace() for p in self._agent_ids | self._debug_port
        }
        self.action_queues = {port: InputQueue() for port in self._agent_ids | self._debug_port}
        self.episode_reward = 0.
        # self.om.__init__(self.config) # resets fast obs reader , so no

        combination_crash_checking = not self.render

        # For now we do random stages
        stage = np.random.choice(self.config["stages"]) #if (options is None or "stage" not in options) else options["stage"]
        chars = np.random.choice(self.config["chars"], len(self.config["players"])) if (
                    options is None) \
            else tuple(options[aid]["character"] for aid in self.get_agent_ids())

        # TODO fair randomization, some chars will be sampled less as a result
        if options is not None:
            while combination_crash_checking and (*chars, stage) in SSBM.before_game_stuck_combs + SSBM.ingame_stuck_combs:
                # TODO : make it swap player ports.
                print("Tried playing as", (*chars, stage), "but this combination does not work")
                stage = np.random.choice(self.config["stages"])

        else:
            while combination_crash_checking and (*chars, stage) in SSBM.before_game_stuck_combs + SSBM.ingame_stuck_combs:
                stage = np.random.choice(self.config["stages"])
                chars = np.random.choice(self.config["chars"], len(self.config["players"]))

        n_start = 70
        counter = 0
        self.current_matchup = tuple(chars) + (stage,)

        while state.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH, None]:
            state = self.step_nones(reset_if_stuck=True)
            if state is None:
                return None

        while state.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            for port, controller in self.controllers.items():
                if controller._type != PlayerType.HUMAN:
                    cpu_level = 9 if self.players[port-1] == PlayerType.CPU else 0
                    if port == 2:
                        auto_start = counter > n_start
                    else:

                        auto_start = False

                    melee.MenuHelper.menu_helper_simple(
                        state,
                        controller,
                        chars[port-1],
                        stage,
                        autostart=auto_start,
                        cpu_level=cpu_level
                    )

            state = self.step_nones(reset_if_stuck=True)
            if state is None:
                # Probably selection screen ?
                # seems to depend on the chars
                print("stuck at selection screen", self.current_matchup)
                return None

            counter += 1
            if counter > 3800:
                if counter > 3900:
                    self.previously_crashed = True
                    return self.reset()
                print("STUCK", state.menu_state, state.frame, state.stage_select_cursor_x, state.stage_select_cursor_y)


        init_frames = 77 # 77
        for _ in range(init_frames):
            state = self.step_nones(reset_if_stuck=True)
            if state is None:
                return None

        self.om.update(state)
        self.game_frame = 0

        print("Went Through!")

        return self.om.build(self.state), {i: {} for i in self.om.bot_ports}

    def get_next_state_reward(self, every=3) \
            -> Tuple_T[Union[GameState, None], Dict_T[int, SSBMRewards], bool]:

        collected_rewards = {p: SSBMRewards() for p in self._agent_ids}
        active_actions = {p: None for p in self._agent_ids | self._debug_port}

        # Are we done ? check if one of the teams are out
        game_is_finished = (self.reward_function.episode_finished or
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
                            SSBM.PAD_INTERFACE.send_controller(active_actions[port], self.controllers[port],
                                                               # Additionally pass some info to filter out dumb actions
                                                               self.get_gamestate().players[port],
                                                               # If the action is dumb, we want to terminate the current
                                                               # sequence
                                                               curr_sequence)
                            if self.config["debug"]:
                                print(f"Sent input {active_actions[port]} of sequence {curr_sequence} on port {port}.")
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
                                if self.ingame_stuck_frozen_counter > 100:
                                    print(f"TODO: port {port} is frozen ingame with {self.current_matchup} for"
                                        f"{self.ingame_stuck_frozen_counter} frames", old_state.players[port].action_frame)
                            else:
                                self.ingame_stuck_frozen_counter = 0

                if self.config["debug"]:
                    input("Press [Enter] to step a frame")

                if next_state is None:
                    # Weird state
                    print(f"Stuck here?, crashed with {self.current_matchup}")
                    # force full reset here
                    self.previously_crashed = True
                    return next_state, collected_rewards, game_is_finished
                    #return {}, {}, {"__all__": True}, {"__all__": True}, {}
                else:
                    # Put in our custom frame counter
                    self.game_frame += 1
                    next_state.custom["game_frame"] = self.game_frame

                    # process rewards
                    self.reward_function.compute(next_state, active_actions, collected_rewards)

                # check if we are done
                game_is_finished = self.reward_function.episode_finished
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

        state, rewards, is_game_finished = self.get_next_state_reward()

        if state is None:
            # We crashed, return dummy stuff
            return {}, {}, {"__all__": True}, {"__all__": True}, {}

        elif state.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            # Might have happened if we reached time limit ?
            print("We are here, so this happens sometimes ?")

            # Pad if we have delay
            done = self.pad_at_end <= 0
            self.pad_at_end -= 1
            self.reached_end = True
            # Do not give the new state, because it looks weird right now
            # Just update the game frame
            self.om.update(state, game_frame=True)
            self.om.build(self.state)

        elif is_game_finished:
            done = self.pad_at_end <= 0
            self.pad_at_end -= 1

            if self.reached_end:
                self.om.update(state, game_frame=True)
            else:
                self.om.update(state)
            self.reached_end = True
            self.om.build(self.state)
        else:
            done = False
            self.om.update(state)
            self.om.build(self.state)

        # Punish for overtime
        # if self.game_frame > 8 * 60 * 60 + 63:
        #     print(self.game_frame)
        #     punish_all += 0 # 5

        dones = {
            i: done for i in self._agent_ids
        }
        dones["__all__"] = done

        if done and self.config["debug"]:
            print(f"Collected rewards: {rewards}")

        total_rewards = {}
        for aid, reward in rewards.items():
            total_rewards[aid] = reward.total()
            for rid, r in reward.to_dict().items():
                self.episode_metrics[f"{aid}/{rid}"] += r

        return self.state, total_rewards, dones, dones, {}

    def close(self):
        print(self.env_index,
              "STOPING=================================================================================================")

        if self.env_index == 1:
            # Flush updated frame data to csv
            self.om.FFD.override()

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
