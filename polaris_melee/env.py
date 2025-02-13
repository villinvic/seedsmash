import atexit
import time
from collections import defaultdict
from functools import partial
from typing import Optional, List, Union, SupportsFloat, Any
from typing import Dict as Dict_T
from typing import Tuple as Tuple_T
from copy import copy, deepcopy

import psutil
from gymnasium.core import ActType, ObsType
from gymnasium.error import ResetNeeded
from melee import GameState, Console
import melee
import numpy as np
import os
from melee.enums import ControllerType, Character, Stage, AttackState, Action
from melee.slippstream import EnetDisconnected
from polaris_melee.character_specific_observations import get_character_specific_observations

from polaris_melee.combo_tracker import ComboTracker
from polaris_melee.enums import PlayerType
from polaris_melee.action_space import ComboPad, InputQueue, ActionControllerInterface, SSBMActionSpace
from polaris_melee.observation_space import ObsBuilder
from polaris_melee.preferences import RewardFunction, StepRewards
import gc
from polaris.environments import PolarisEnv

from seedsmash.bot import Bot
from seedsmash.bots.bot_config import BotConfig


def build_console(
    slippi_port: int,
    config: dict,
    render: bool,
) -> melee.Console:

    online_delay = config["online_delay"]
    save_replays = config["save_replays"]
    polling_mode = config["polling_mode"]

    kwargs = dict(
        copy_home_directory=False,
        slippi_port=slippi_port,
        blocking_input=True,
        online_delay=online_delay,
        save_replays=save_replays,
        fullscreen=False,
    )

    if render:
        kwargs.update(
            path=config["paths"]["FM"],
            #copy_home_directory=True,
            enable_ffw=False,
            gfx_backend='',
            disable_audio=False,
            use_exi_inputs=False,
            polling_timeout=0, # 60
            #blocking_input=False,
            fullscreen=True,
            polling_mode=polling_mode
        )
    else:
        kwargs.update(
            path=config["paths"]["ExiAI"],
            enable_ffw=config["use_ffw"],
            gfx_backend='Null',
            disable_audio=True,
            use_exi_inputs=True,
            polling_timeout=60,
            polling_mode=polling_mode
        )

    return melee.Console(**kwargs)


def build_controllers(
        console: melee.Console,
        player_types: Dict_T[int, PlayerType],
) -> Dict_T[int, ComboPad]:
    return {
        port : ComboPad(console=console, port=port, type=(
        ControllerType.GCN_ADAPTER if p_type == PlayerType.HUMAN else ControllerType.STANDARD
    )) for port, p_type in player_types.items()
    }


def plug_setup(
        console: melee.Console,
        controllers: Dict_T[int, ComboPad]
):
    try:
        connected = [console.connect()] + [controller.connect() for _, controller in controllers.items()]
    except Exception as e:
        raise ResetNeeded(f"Something went wrong plugging the setup: {e}")
    return all(connected)


def run_console(
        console: melee.Console,
        render: bool,
        config: dict,
):
    platform = None
    if not render and console.dolphin_version.mainline:
        platform = 'headless'
    return console.run(config["paths"]["iso"], platform=platform)


class SSBM(PolarisEnv):
    env_id = "SSBM-2"

    def __init__(
            self,
            env_index=-1,
            **config
    ):
        super().__init__(env_index=env_index, **config)

        self.render = env_index in (0, -123) and self.config["render"]
        p = psutil.Process()
        if env_index >= 0:
            self.cpu_affinity = [2*env_index, 2*env_index+1]
            p.cpu_affinity(self.cpu_affinity)
        else:
            self.cpu_affinity = [0]

        self.polling_mode = config["polling_mode"]
        self.slippi_port = 51441 + self.env_index

        # TODO: Netplay
        self.player_types = {i+1: p_type for i, p_type in enumerate(config["player_types"])}
        self.populated_ports = set(self.player_types)

        self.observation_builder = ObsBuilder(config)
        self._agent_ids = set(self.observation_builder.bot_ports)
        self._debug_port = set([port for port, player_type in self.player_types.items()
                                if player_type == PlayerType.HUMAN_DEBUG])

        self.observation_space = self.observation_builder.gym_specs
        # You need a space for each player, as the action sequences do not copy.
        self.action_space_constructor = partial(SSBMActionSpace,
                                                delay=config["online_delay"]
                                                )
        self.action_space = self.action_space_constructor().gym_spec

        atexit.register(self.close)

        self.game_info = {}

        self.empty_info_dict = {p: {} for p in self.get_agent_ids()}

    def initialise_setup(self):
        self.console = build_console(
            self.slippi_port,
            self.config,
            self.render
        )
        self.controllers = build_controllers(
            self.console,
            self.player_types
        )
        run_console(self.console, self.render, self.config)
        #psutil.Process(self.console._process.pid).cpu_affinity(self.cpu_affinity)
        success = plug_setup(self.console, self.controllers)
        return success

    def step_console(
            self,
            num_steps=1,
    ) -> melee.GameState:
        gamestate = None
        try:
            for _ in range(num_steps):
                if self.polling_mode:
                    gamestate = None
                    tries = 0
                    while gamestate is None:
                        gamestate = self.console.step()
                        tries += 1
                        if tries > 500_000:
                            raise Exception(f"Not receiving any gamestate after {tries} polling attempts.")
                else:
                    gamestate = self.console.step()
        except Exception as e:
            raise ResetNeeded(f"Something went wrong stepping the console: {e}")

        return gamestate

    def handle_menus(
            self,
            options,
    ) -> melee.GameState:
        characters = {}
        costumes = {}
        taken_costumes = []
        stage_sampling_weights = {s: 1. for s in self.config["playable_stages"]}
        for port in self.populated_ports:
            if port in options:
                character = options[port].character
                costume = options[port].costume_id
                if (character, costume) in taken_costumes:
                    # make sure we do not go out of bound
                    costume = 0

                stage_sampling_weights[options[port].preferred_stage] += 1.
            else:
                character = np.random.choice(self.config["playable_characters"])
                costume = 0

            while (character, costume) in taken_costumes:
                costume += 1

            characters[port] = options[port].character
            costumes[port] = costume
            taken_costumes.append((character, costume))

        p = np.array(list(stage_sampling_weights.values()))
        p /= p.sum()
        stage = np.random.choice(self.config["playable_stages"], p=p)
        self.current_matchup = {"characters": characters, "stage": stage}

        gamestate = self.step_console()

        while gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH, None]:
            gamestate = self.step_console()

        if self.render:
            gamestate = self.step_console(num_steps=1)#60 * 7)
            # Seedsmash specific
            # Wait for the matchmaking animation to end
            # TODO: do this more cleanly

        lvl = 9
        ready = [True for _ in range(4)]
        press_start = False
        menu_helper = melee.MenuHelper()
        css_counter = 0
        while gamestate.menu_state != melee.Menu.IN_GAME:
            for port, controller in self.controllers.items():
                p_type = self.player_types[port]
                if p_type != PlayerType.HUMAN:
                    cpu_level = lvl if p_type == PlayerType.CPU else 0
                    menu_helper.menu_helper_simple(
                        gamestate,
                        controller,
                        characters[port],
                        stage,
                        connect_code="",
                        costume=costumes[port],
                        autostart=press_start,
                        swag=False,
                        cpu_level=cpu_level
                    )

            press_start = all(ready)
            gamestate = self.step_console()

            css_counter += 1
            if not self.render and css_counter > 3000:
                raise ResetNeeded(f"Stuck in CSS, selecting {self.current_matchup}")

        return gamestate

    def step_until_ready_go(
            self,
    ) -> melee.GameState:
        entrance = False
        c = 0
        for port, controller in self.controllers.items():
            if controller._type != PlayerType.HUMAN:
                controller.release_all()

        while not entrance:
            gamestate = self.step_console()

            next_entrance = True
            for p, player in gamestate.players.items():
                next_entrance = next_entrance and (player.action not in (Action.ENTRY_START, Action.ENTRY, Action.ENTRY_END))
            entrance = next_entrance
            c += 1
            if c > 500:
                raise ResetNeeded("Stuck at game entrance.")

        # step some more to ensure first frame is actionable
        return self.step_console(num_steps=20)

    def iterate_port_until_success(self):
        success = self.initialise_setup()
        tries = 0
        while not success:
            self.close()
            self.slippi_port += 100
            success = self.initialise_setup()
            tries += 1
            if tries > 5:
                raise ResetNeeded(f"Can't find proper port for dolphin n°{self.env_index}")

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: dict[int, Bot],
    ) -> Tuple_T[Dict_T[int, dict], dict]:

        self.game_info = {}
        self.is_done = False


        if not hasattr(self, "console"):
            success = self.iterate_port_until_success()

        # Reset episodic attributes
        self.episode_metrics = defaultdict(float)
        self.reward_function = RewardFunction(options)
        self.combo_counters = {p: ComboTracker(
            ObsBuilder.MAX_COMBO,
            self.observation_builder.FD
        ) for p in self.populated_ports}

        self.character_specific_observations = {p: get_character_specific_observations(
            options[p].character,
            self.config["online_delay"],
        ) for p in self.populated_ports}

        self.episode_length = 1
        self.discrete_controllers = {p: self.action_space_constructor() for p in self._agent_ids | self._debug_port}
        self.action_queues = {port: InputQueue() for port in self._agent_ids | self._debug_port}
        self.episode_reward = 0.

        # select characters and stages
        self.handle_menus(options)
        for port, controller in self.controllers.items():
            if self.player_types[port] != PlayerType.HUMAN:
                controller.release_all()

        gamestate = self.step_until_ready_go()

        self.game_info["stage"] = gamestate.stage
        self.game_info["bot_a"] = options[1].tag
        self.game_info["bot_b"] = options[2].tag

        self.observation_builder.reset()
        self.observation_builder.update(gamestate)
        return self.observation_builder.build(), self.empty_info_dict

    def is_episode_finished(self):
        if self.is_done:
            return True

        gamestate = self.get_gamestate()

        p1_down = gamestate.players[1].stock == 0
        p2_down = gamestate.players[2].stock == 0

        if p1_down and p2_down:
            self.is_done = True
        elif p1_down:
            self.game_info["winner"] = self.game_info["bot_b"]
            self.is_done = True
        elif p2_down:
            self.game_info["winner"] = self.game_info["bot_a"]
            self.is_done = True

        if self.is_done:
            self.game_info["length"] = self.episode_length

        return self.is_done



    def get_next_state_reward(self, every=3) \
            -> Tuple_T[Union[GameState, None], Dict_T[int, StepRewards]]:

        step_rewards = {p: defaultdict(float) for p in self._agent_ids | self._debug_port}
        active_actions = {p: None for p in self._agent_ids | self._debug_port}

        next_gamestate = self.get_gamestate()
        if not self.is_episode_finished():
            for frame in range(every):
                gamestate = self.get_gamestate()
                players = gamestate.players

                for port in self._agent_ids | self._debug_port:
                    if port in players:
                        next_input = self.action_queues[port].pull(
                            frame == 0,
                            players[port]
                        )
                        if next_input:
                            active_actions[port], curr_sequence = next_input
                            ActionControllerInterface.send_controller(active_actions[port], self.controllers[port],
                                                               # Additionally pass some info to filter out dumb actions
                                                               gamestate, players[port],
                                                               # If the action is dumb, we want to terminate the current
                                                               # sequence
                                                               curr_sequence)
                            if self.config["debug"]:
                                print(f"Sent input {active_actions[port]} of sequence {curr_sequence} on port {port}.")
                                print(f"curr action_state:{players[port].action}")
                        elif self.config["debug"]:
                            print(f"Waiting a frame before new input on port {port}.")

                next_gamestate = self.step_console()

                if self.config["debug"]:
                    input("Press [Enter] to step a frame")
                if next_gamestate is None:
                    # Weird state
                    print(f"Stuck here?, crashed with {self.current_matchup}")
                    # force full reset here
                    self.dump_bad_combination_and_raise_error(3)
                else:
                    # process differences in gamestates
                    # Put in our custom combo counter and char specific helpers

                    next_gamestate.custom["combo_counters"] = {}
                    next_gamestate.custom["character_specific"] = {}

                    for port in self.populated_ports:
                        other_port = port % 2 + 1
                        next_gamestate.custom["combo_counters"][port] = self.combo_counters[port].update(
                            players[port],
                            players[other_port]
                        )
                        next_gamestate.custom["character_specific"][port] = self.character_specific_observations[port].update(
                            player=players[port],
                            gamestate=next_gamestate
                        )

                    self.reward_function.accumulate(step_rewards, next_gamestate)

                # check if we are done, and exit if it is the case
                if self.is_episode_finished():
                    break

        return next_gamestate, step_rewards

    def handle_controller_inputs(
            self,
            action_dict: Dict_T[int, int]
    ):
        for port, action_idx in action_dict.items():
            self.action_queues[port].push(self.discrete_controllers[port][action_idx])

        for port in self._debug_port:
            action = None
            while action is None:
                try:
                    print(self.discrete_controllers[port])
                    action = input(f"Choose an action for port {port}: ")
                    if action == "":
                        action = self.discrete_controllers[port].RESET_CONTROLLER
                    else:
                        action = self.discrete_controllers[port][int(action)]
                except Exception as e:
                    print(e)
            print()
            self.action_queues[port].push(action)


    def step(
        self, action_dict: Dict_T[int, int]
    ) -> Tuple_T[dict, dict, dict, dict, dict]:

        self.handle_controller_inputs(action_dict)

        gamestate, step_rewards = self.get_next_state_reward()

        done = False
        if gamestate is None:
            # Should not be going there
            self.dump_bad_combination_and_raise_error(4)
        elif self.is_episode_finished():
            counter = 0
            done = True
            while gamestate.menu_state not in [melee.Menu.CHARACTER_SELECT, melee.Menu.SLIPPI_ONLINE_CSS]:
                gamestate = self.step_console()
                counter += 1
                if counter > 1000:
                    raise ResetNeeded("Stuck post game.")
        elif gamestate.menu_state == melee.Menu.IN_GAME:
            done = False
        else:
            raise ResetNeeded(f"We went to a strange state: {gamestate.menu_state}")

        self.observation_builder.update(gamestate)
        obs_dict = self.observation_builder.build()

        dones = {
            i: done for i in self._agent_ids
        }
        dones["__all__"] = done

        if done:
            # merge all metrics:
            reward_function_metrics = self.reward_function.get_metrics(self.episode_length)
            self.game_info["metrics"] = {}
            for p, k in zip(self.reward_function.get_metrics(self.episode_length), ["bot_a", "bot_b"]):
                self.game_info["metrics"][k] = reward_function_metrics[p] | self.combo_counters[p].get_metrics()
            self.game_info["replay"] = None # todo
            self.episode_metrics["game_info"] = self.game_info

        self.episode_length += 1

        return obs_dict, self.reward_function.zero_sum(step_rewards), dones, dones, {}


    def get_gamestate(self) -> GameState:
        if self.console is None:
            return None
        else:
            return self.console._prev_gamestate

    def get_episode_metrics(self):

        return self.episode_metrics

    def dump_bad_combination_and_raise_error(self, errnum):
        #self.bad_combinations.dump_on_error(*self.current_matchup, errnum)

        raise ResetNeeded(f"Dolphin {self.env_index} crashed with {self.current_matchup} [error:{errnum}]")

    def close(self):
        if self.env_index == 1:
            # Flush updated frame data to csv
            self.observation_builder.FFD.save()

        try:
            self.console.stop()
        except:
            pass
        try:
            for port, controller in self.controllers.items():
                    controller.disconnect()
        except:
            pass
