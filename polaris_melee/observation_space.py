import copy
from copy import deepcopy
from enum import Enum

from sortedcontainers import SortedDict
from gymnasium.spaces.dict import Dict
from typing import Tuple

import melee
from melee import Stage, PlayerState, Character, Action, stages, enums, Projectile, GameState, AttackState, \
    left_platform_position, right_platform_position, top_platform_position, ProjectileType, Moves, character_moves
from polaris_melee.actions import is_shield, AERIAL_MOVEMENT_ACTIONS, CROUCH_ACTIONS, DODGE_ACTIONS

from polaris_melee.compiled_libmelee_framedata import CompiledFrameData
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete

from polaris_melee.enums import PlayerType
from polaris_melee.make_data import FrameData as FastFrameData
from polaris_melee.normalised_char_attributes import NormalisedCharacterAttributes
from polaris_melee.playstyle_tracker import PlayStyleTracker

action_idx = {
    s: i for i, s in enumerate(Action)
}
idx_to_action = {
    i: s for i, s in enumerate(Action)
}

action_state_idx = {
    s: i for i, s in enumerate(AttackState)
}


move_idx= {}
i = 341
for char, moves in character_moves.items():
    for move in moves:
        if move not in Moves._value2member_map_:
            move_idx[(char, move.value)] = i
            i += 1

n_actions = len(action_idx)
n_moves = len(move_idx)

def randall_position(frame, stage):
    y, x1, x2 = stages.randall_position(frame)

    if stage != Stage.YOSHIS_STORY:
        y = 0.
        x1 = 0.
        x2 = 0.

    return y, x1, x2

platform_presences = {
    Stage.YOSHIS_STORY: (1, 1, 1, 1),
    Stage.FINAL_DESTINATION: (0, 0, 0, 0),
    Stage.DREAMLAND: (1, 1, 1, 0),
    Stage.POKEMON_STADIUM: (1, 1, 0, 0),
    Stage.BATTLEFIELD: (1, 1, 1, 0),
    Stage.FOUNTAIN_OF_DREAMS: (1, 1, 1, 0),
}

class ActionType(Enum):
    DODGE = 0
    ATTACK = 1
    SHIELD = 2
    GRAB = 3
    CROUCH = 4
    OTHER = 5





class StateDataInfo:
    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"

    HARD_BOUNDS = (-10., 10)

    def __init__(
            self, extractor, nature, name="UNSET", scale=1., size=1, bounds=None, player_port=None, config={}
    ):
        self.name = name
        self.base_name = None  # utility for player dependency
        self.extractor = extractor
        self.nature = nature
        self.scale = scale
        self.size = size if nature != StateDataInfo.CATEGORICAL else 1
        self.bounds = StateDataInfo.HARD_BOUNDS if bounds is None else tuple(b*scale for b in bounds)
        self.n_values = None if nature != StateDataInfo.CATEGORICAL else size
        self.delay = config["obs_config"]["delay"]
        self.debug = config["debug"]
        self.delay_idx = 0
        self.init_values()
        self.gym_space = self.get_gym_space()
        self.update = self.build_op()
        self.player = player_port

    def is_player_dependent(self):
        return not self.player is None

    def init_values(self):
        if self.nature == StateDataInfo.CONTINUOUS:
            dtype = np.float32
        elif self.nature in (StateDataInfo.BINARY, StateDataInfo.CATEGORICAL):
            dtype = np.int32
        else:
            dtype = np.float32
        #dtype = np.float32 if self.nature in (StateDataInfo.CONTINUOUS, StateDataInfo.CATEGORICAL) else np.int8

        self.value = np.zeros(
            (self.delay + 1, self.size), dtype=dtype,
        )

    def observe(self, delay=0):
        observed = (self.delay_idx % (self.delay + 1))

        assert delay <= self.delay, f"Got unexpectedly large delay requested for observation {self.name}"

        observed -= delay

        if self.debug:
            print(self.name, self.value[self.delay_idx % (self.delay + 1)], "(undelayed)", self.value[observed], "(observed)")

        if self.nature == StateDataInfo.CONTINUOUS:
            return np.clip(self.value[observed] * self.scale, *self.bounds)

        return self.value[observed]

    def reset(self):
        self.value[:] = 0
        self.delay_idx = 0

    def get_gym_space(self):
        if self.nature == StateDataInfo.CONTINUOUS:
            return Box(*self.bounds, (self.size,), dtype=np.float32)
        elif self.nature == StateDataInfo.CATEGORICAL:
            return Box(0, self.n_values - 1, (1,), dtype=np.float32)
        elif self.nature == StateDataInfo.BINARY:
            return MultiBinary(self.size)
        else:
            raise NotImplementedError

    def build_op(self):
        def op(state):
            extracted = self.extractor(state)
            try:
                self.value[self.delay_idx % (self.delay + 1), :] = extracted
            except Exception as e:
                print(self.name)
                raise e
        return op

    def advance(self):
        self.delay_idx += 1


class PostProcessFeature:

    def __init__(
            self, extractor, nature, name="UNSET", scale=1., size=1, config={}
    ):
        self.name = name
        self.base_name = None  # utility for player dependency
        self.nature = nature
        self.scale = scale
        self.size = size if nature != StateDataInfo.CATEGORICAL else 1
        self.n_values = None if nature != StateDataInfo.CATEGORICAL else size
        self.debug = config["debug"]
        self.gym_space = self.get_gym_space()
        self.extractor = extractor

    def observe(self, features):
        value = self.extractor(features)
        if self.nature == StateDataInfo.CONTINUOUS:
            value = np.clip(value * self.scale, *StateDataInfo.HARD_BOUNDS)
        return value

    def get_gym_space(self):
        if self.nature == StateDataInfo.CONTINUOUS:
            return Box(*StateDataInfo.HARD_BOUNDS, (self.size,), dtype=np.float32)
        elif self.nature == StateDataInfo.CATEGORICAL:
            return Box(0, self.n_values - 1, (1,), dtype=np.float32)
        elif self.nature == StateDataInfo.BINARY:
            return MultiBinary(self.size)
        else:
            raise NotImplementedError


class ObsBuilder:
    FRAME_SCALE = 0.02
    SPEED_SCALE = 0.5
    POS_SCALE = 0.05
    HITBOX_SCALE = 0.1
    PERCENT_SCALE = 0.01
    FD = CompiledFrameData()
    FFD = FastFrameData()

    CONTINOUS = "continuous"
    BINARY = "binary"
    DISCRETE = "discrete"
    STAGE = "stage"
    PROJECTILE = "projectile"
    ECB = "ecb"
    PLAYER = "player"
    PLAYERS = {
        i: "player" + "_" + str(i) for i in range(1, 5)
    }

    PLAYER_EMBED = "player_embedding"
    CONTROLLER_STATE = "controller_state"
    EXTRA = "extra"

    player_permuts = {
        1: {
            1: "1", 2: "2", 3: "3", 4: "4"
        },
        2: {
            1: "2", 2: "1", 3: "4", 4: "3"
        },
        3: {
            1: "3", 2: "4", 3: "1", 4: "2"
        },
        4: {
            1: "4", 2: "3", 3: "2", 4: "1"
        }
    }

    MAX_COMBO = 4

    def __init__(
            self,
            config: dict,
    ):

        self.config = config
        all_stages_to_used = {
            s: i
            for i, s in enumerate(config["playable_stages"])
        }
        all_chars_to_used = {
            s: i
            for i, s in enumerate(config["playable_characters"])
        }

        self.character_data = NormalisedCharacterAttributes(observed=("size", "weight", "Gravity", "Friction", "AirFriction"))
        num_tracked_options = PlayStyleTracker(self.FD).dim

        n_characters = len(all_chars_to_used)
        n_stages = len(all_stages_to_used)

        # TODO online port assignement
        self.bot_ports = [i + 1 for i, p_type in enumerate(config["player_types"]) if p_type == PlayerType.BOT]

        self.num_players = len(config["player_types"])

        def platforms(gamestate: GameState):
            obs = (stages.side_platform_position(right_platform=False, gamestate=gamestate)
                   + stages.side_platform_position(right_platform=True, gamestate=gamestate)
                   +stages.top_platform_position(gamestate.stage)
                   + randall_position(gamestate.frame, gamestate.stage)
                   )

            return obs

        def platform_distance(platform: Tuple[float, float, float], player: PlayerState):
            py, px1, px2 = platform
            dy = py - player.position.y

            if px1 < player.position.x < px2:
                dx = 0
            else:
                absx1 = abs(px1 - player.position.x)
                absx2 = abs(px2 - player.position.x)
                if absx1 < absx2:
                    dx = px1 - player.position.x
                else:
                    dx = px2 - player.position.x
            return dx, dy

        def platform_distances(gamestate: GameState, player: PlayerState):

            distances = ()
            for platform, plaform_present in zip(
                    (stages.side_platform_position(right_platform=False, gamestate=gamestate),
                    stages.side_platform_position(right_platform=True, gamestate=gamestate),
                    stages.top_platform_position(gamestate.stage),
                    randall_position(gamestate.frame, gamestate.stage)),
                    platform_presences[gamestate.stage]
            ):
                if plaform_present:
                    distances += platform_distance(platform, player)
                else:
                    distances += (0., 0.)
            return distances

        def action_type(player: PlayerState):
            action = player.action
            act_type = ActionType.OTHER.value
            if self.FD.is_grab(player.character, action):
                act_type = ActionType.GRAB.value
            elif self.FD.is_attack(player.character, action):
                act_type = ActionType.ATTACK.value
            elif is_shield(player):
                act_type = ActionType.SHIELD.value
            elif action in DODGE_ACTIONS:
                act_type = ActionType.DODGE.value
            elif action in CROUCH_ACTIONS:
                act_type = ActionType.CROUCH.value

            return act_type

        def projectile_dist(p: Projectile, player: PlayerState):
            return np.sqrt(np.square(p.position.x - player.position.x) + np.square(p.position.y - player.position.y))
            return np.sqrt(np.square(p.position.x - player.position.x) + np.square(p.position.y - player.position.y))

        def own_projectile_getter(state, port):
            other_port = 1 + port % 2
            # we want the projectile that is the closest to other_port
            own_projectiles = [
                p for p in state.projectiles if (
                        p.owner in (port, -1)
                        and p.type != ProjectileType.UNKNOWN_PROJECTILE
                )
            ]
            if len(own_projectiles) > 0:
                nearest_own_projectile = min(
                    own_projectiles, key=lambda p: projectile_dist(p, state.players[other_port])
                )
                return nearest_own_projectile.position.x, nearest_own_projectile.position.y, 1.
            else:

                return 0., 0., 0.

        def get_projectiles(gamestate: GameState, owner_port=-1):
            projectile_infos = []
            projs = gamestate.projectiles[:config["obs_config"]["max_projectiles_per_owner"]]
            n_proj = 0
            for projectile in projs:
                if projectile.type == ProjectileType.UNKNOWN_PROJECTILE or projectile.owner != owner_port:
                    continue
                n_proj += 1
                ownership = [0, 0, 0]
                if owner_port == -1:
                    ownership[0] = 1
                else:
                    ownership[owner_port] = 1
                projectile_infos.extend([projectile.x_speed * self.SPEED_SCALE, projectile.y_speed * self.SPEED_SCALE,
                                         projectile.x * self.POS_SCALE, projectile.y * self.POS_SCALE,
                ] + ownership)

            padding = [0] * 7 * (config["obs_config"]["max_projectiles_per_owner"] - n_proj)
            projectile_infos = projectile_infos + padding

            return projectile_infos

        def get_nearest_platform(state, port):
            """
            gets nearest platform ledges.
            """

            x, y = state.players[port].position.x,  state.players[port].position.y
            n_p_y, n_p_x1, n_p_x2 = left_platform_position(state)
            no_plat = (
                n_p_y== 0.0 and  n_p_x1 == 0.0 and  n_p_x2 == 0.0
            )
            if no_plat:
                n_dist = np.inf
            else:
                n_dist = np.minimum(
                        ( (x - n_p_x1) ** 2 + 0.3*(y - n_p_y) ** 2) ** 0.5,
                        ( (x - n_p_x2) ** 2 + 0.3*(y - n_p_y) ** 2) ** 0.5,
                    )

            for p_y, p_x1, p_x2 in (right_platform_position(state), top_platform_position(state),
                                    randall_position(state.frame, state.stage)):
                no_plat = (
                        p_y == 0.0 and p_x1 == 0.0 and p_x2 == 0.0
                )
                if no_plat:
                    dist = np.inf
                else:
                    dist = np.minimum(
                        ((x - p_x1) ** 2 + 0.3*(y - p_y) ** 2) ** 0.5,
                        ((x - p_x2) ** 2 + 0.3*(y - p_y) ** 2) ** 0.5,
                    )

                if dist < n_dist:
                    n_p_x1, n_p_y, n_p_x2 = p_x1, p_y, p_x2

            return n_p_y, n_p_x1, n_p_x2


        def get_pos(state: GameState, port: int):
            if state.players[port].action.value <= 0xa:
                return 0., stages.BLASTZONES[state.stage][-1]
            return ObsBuilder.FD.FD.roll_end_position(state.players[port], state), state.players[port].position.y


        stage_value_dict = dict(
            stage=StateDataInfo(lambda s: all_stages_to_used.get(s.stage, 0),
                                StateDataInfo.CATEGORICAL,
                                size=n_stages,
                                config=self.config,
                                ),
            stage_width=StateDataInfo(lambda s: stages.EDGE_POSITION[s.stage],
                                StateDataInfo.CONTINUOUS,
                                scale=self.POS_SCALE,
                                config=self.config,
                                ),
            platforms=StateDataInfo(platforms,
                                      StateDataInfo.CONTINUOUS,
                                      scale=self.POS_SCALE,
                                      size=3*4,
                                      config=self.config,
                                      ),
            platform_presences=StateDataInfo(lambda s: platform_presences[s.stage],
                                    StateDataInfo.BINARY,
                                    size=4,
                                    config=self.config,
                                    ),

            projectiles=StateDataInfo(lambda s: get_projectiles(s, -1),
                                    StateDataInfo.CONTINUOUS,
                                    size=config["obs_config"]["max_projectiles_per_owner"] * 7, # x, y, vx, vy, existence
                                    config=self.config,
            ),
        )

        def get_action_index(state: GameState, port: int):
            action = state.players[port].action
            return action_idx[action]

            a_val = action.value
            char = state.players[port].character
            if a_val in character_moves[char]._value2member_map_:
                # either a general move, or a character specific move
                if a_val in Moves._value2member_map_:
                    return action_idx[action]
                # this is a character specific move, fabric a custom index
                # everything will be fed to an embedding lookup table
                return move_idx[(char, a_val)]
            else:
                return action_idx[action]

        def make_player_dict(port):
            """
            Helper function for indexing in lambda functions
            """
            # FD.frame_count is slower that our FFD.remaining_frame ?
            # iasa also iterates over dicts...
            def frames_before_next_hitbox(char_state):
                next_hitbox_frame = ObsBuilder.FD.frames_before_next_hitbox(char_state.character,
                                                                            char_state.action,
                                                                            char_state.action_frame
                                                        )

                return next_hitbox_frame

            return dict(
                iasa=StateDataInfo(lambda s: ObsBuilder.FD.iasa[s.players[port].character][
                                                                    s.players[port].action]
                                                                     - s.players[port].action_frame,
                                   StateDataInfo.CONTINUOUS,
                                   scale=self.FRAME_SCALE,
                                   bounds=(0., 180.),
                                   player_port=port,
                                   config=self.config),
                frames_before_next_hitbox=StateDataInfo(lambda s: frames_before_next_hitbox(
                    s.players[port]
                ),
                                   StateDataInfo.CONTINUOUS,
                                   scale=ObsBuilder.FRAME_SCALE,
                                   bounds=(0., 180.),
                                   player_port=port,
                                   config=self.config),
                attack_state=StateDataInfo(
                    lambda s: ObsBuilder.FD.attack_state(s.players[port].character, s.players[port].action, s.players[port].action_frame).value,
                    StateDataInfo.CATEGORICAL,
                    size=len(AttackState),
                    player_port=port,
                    config=self.config
                ),
                # is_attack=StateDataInfo(lambda s: ObsBuilder.FD.is_attack(s.players[port].character,
                #                                                           s.players[port].action),
                #                         StateDataInfo.BINARY,
                #                         player_port=port,
                #                         config=self.config),
                percent=StateDataInfo(lambda s: s.players[port].percent,
                                      # putting other_port inseast could be a trick to help combos,
                                      # but prevents overreacting to hitstun, etc.
                                      # however, action_frame, action and other stuff is leaking info about ourself
                                      StateDataInfo.CONTINUOUS,
                                      scale=ObsBuilder.PERCENT_SCALE,
                                      bounds=(0., 300.),
                                      player_port=port,
                                      config=self.config),
                shield_strength=StateDataInfo(lambda s: s.players[port].shield_strength,
                                              StateDataInfo.CONTINUOUS,
                                              scale=0.017,
                                              bounds=(0., 60.),
                                              player_port=port,
                                              config=self.config),
                # stock=StateDataInfo(lambda s: s.players[port].stock,
                #                     StateDataInfo.CATEGORICAL,
                #                     size=5,
                #                     player_port=port,
                #                     config=self.config),
                stock=StateDataInfo(lambda s: s.players[port].stock,
                                    StateDataInfo.CONTINUOUS,
                                    scale=1/4,
                                    player_port=port,
                                    config=self.config),
                action_frame=StateDataInfo(lambda s: self.FFD.remaining_frame(s.players[port].character,
                                                                              s.players[port].action,
                                                                              s.players[port].action_frame),
                                           StateDataInfo.CONTINUOUS,
                                           scale=ObsBuilder.FRAME_SCALE,
                                           bounds=(0., 180.),
                                           player_port=port,
                                           config=self.config),
                facing=StateDataInfo(lambda s: np.int32(s.players[port].facing),
                                     StateDataInfo.BINARY,
                                     player_port=port,
                                     config=self.config),
                invulnerability_type=StateDataInfo(lambda s: s.players[port].invulnerability_type.value,
                                           StateDataInfo.CATEGORICAL,
                                           size=3,
                                           player_port=port,
                                           config=self.config),
                # invulnerability_left=StateDataInfo(lambda s: s.players[port].invulnerability_left,
                #                                    StateDataInfo.CONTINUOUS,
                #                                    scale=ObsBuilder.FRAME_SCALE,
                #                                    player_port=port,
                #                                    config=self.config),
                hitlag_left=StateDataInfo(lambda s: s.players[port].hitlag_left,
                                          StateDataInfo.CONTINUOUS,
                                          scale=1/10,
                                          player_port=port,
                                          bounds=(0., 180.),
                                          config=self.config),
                hitstun_left=StateDataInfo(lambda s: s.players[port].hitstun_frames_left,
                                           StateDataInfo.CONTINUOUS,
                                           scale=ObsBuilder.FRAME_SCALE,
                                           player_port=port,
                                           bounds=(0., 180.),
                                           config=self.config),
                on_ground=StateDataInfo(lambda s: s.players[port].on_ground,
                                        StateDataInfo.BINARY,
                                        size=1,
                                        player_port=port,
                                        config=self.config),
                speed_air_x_self=StateDataInfo(lambda s: s.players[port].speed_air_x_self,
                                               StateDataInfo.CONTINUOUS,
                                               scale=ObsBuilder.SPEED_SCALE,
                                               player_port=port,
                                               config=self.config),
                speed_y_self=StateDataInfo(lambda s: s.players[port].speed_y_self,
                                           StateDataInfo.CONTINUOUS,
                                           scale=ObsBuilder.SPEED_SCALE,
                                           player_port=port,
                                           config=self.config),
                speed_x_attack=StateDataInfo(lambda s: s.players[port].speed_x_attack,
                                             StateDataInfo.CONTINUOUS,
                                             scale=ObsBuilder.SPEED_SCALE,
                                             player_port=port,
                                             config=self.config),
                speed_y_attack=StateDataInfo(lambda s: s.players[port].speed_y_attack,
                                             StateDataInfo.CONTINUOUS,
                                             scale=ObsBuilder.SPEED_SCALE,
                                             player_port=port,
                                             config=self.config),
                speed_ground_x_self=StateDataInfo(lambda s: s.players[port].speed_ground_x_self,
                                                  StateDataInfo.CONTINUOUS,
                                                  scale=ObsBuilder.SPEED_SCALE,
                                                  player_port=port,
                                                  config=self.config),
                # off_stage=StateDataInfo(lambda s: s.players[port].off_stage,
                #                         StateDataInfo.BINARY,
                #                         player_port=port,
                #                         config=self.config),
                # moonwalk=StateDataInfo(lambda s: s.players[port].moonwalkwarning,
                #                        StateDataInfo.BINARY,
                #                        player_port=port,
                #                        config=self.config),
                # this appears bugged
                # powershield=StateDataInfo(lambda s: s.players[port].is_powershield,
                #                           StateDataInfo.BINARY,
                #                           player_port=port,
                #                           config=self.config),
                jumps_left=StateDataInfo(lambda s: s.players[port].jumps_left,
                                         StateDataInfo.CONTINUOUS,
                                         scale=1/2,
                                         player_port=port,
                                         config=self.config),
                # jumps_left=StateDataInfo(lambda s: s.players[port].jumps_left,
                #                          StateDataInfo.CATEGORICAL,
                #                          size=7,
                #                          player_port=port,
                #                          config=self.config),
                # TODO: not working in ff mode: we encode our own controller inputs instead of passing "past_action"
                encoded_action=StateDataInfo(lambda s: 0 if "encored_action" not in s.players[port].custom else
                                       s.players[port].custom["encoded_action"],
                                       StateDataInfo.CONTINUOUS,
                                       size=2+2+5+1,
                                       player_port=port,
                                       config=self.config),
                controller_a=StateDataInfo(lambda s:
                                       s.players[port].controller_state.button[enums.Button.BUTTON_A],
                                       StateDataInfo.BINARY,
                                       player_port=port,
                                       config=self.config),
                controller_b=StateDataInfo(lambda s:
                                       s.players[port].controller_state.button[enums.Button.BUTTON_B],
                                       StateDataInfo.BINARY,
                                       player_port=port,
                                       config=self.config),
                controller_jump=StateDataInfo(lambda s:
                                          int(
                                              s.players[port].controller_state.button[enums.Button.BUTTON_X]
                                              or
                                              s.players[port].controller_state.button[enums.Button.BUTTON_Y]
                                          ),
                                          StateDataInfo.BINARY,
                                          player_port=port,
                                          config=self.config),
                controller_shield=StateDataInfo(lambda s:
                                            int(
                                                s.players[port].controller_state.button[enums.Button.BUTTON_L]
                                                or
                                                s.players[port].controller_state.button[enums.Button.BUTTON_R]
                                            ),
                                            StateDataInfo.BINARY,
                                            player_port=port,
                                            config=self.config),
                controller_z=StateDataInfo(lambda s:
                                       s.players[port].controller_state.button[enums.Button.BUTTON_Z],
                                       StateDataInfo.BINARY,
                                       player_port=port,
                                       config=self.config),
                controller_sticks=StateDataInfo(lambda s:
                                     s.players[port].controller_state.main_stick
                                     + s.players[port].controller_state.c_stick,
                                     StateDataInfo.CONTINUOUS,
                                     size=4,
                                     player_port=port,
                                     config=self.config),
                ecb=StateDataInfo(lambda s: (
                    s.players[port].ecb.top.y, s.players[port].ecb.top.x,
                    s.players[port].ecb.right.y, s.players[port].ecb.right.x,
                    s.players[port].ecb.bottom.y, s.players[port].ecb.bottom.x,
                    s.players[port].ecb.left.y, s.players[port].ecb.left.x,
                ),
                                  StateDataInfo.CONTINUOUS,
                                  size=8,
                                  scale=ObsBuilder.POS_SCALE,
                                  player_port=port,
                                  config=self.config),
                # we actually predict the roll position in the x pos here.
                position=StateDataInfo(lambda s: get_pos(s, port),
                                       StateDataInfo.CONTINUOUS,
                                       size=2,
                                       scale=ObsBuilder.POS_SCALE,
                                       player_port=port,
                                       config=self.config),
                # platform_distances=StateDataInfo(lambda s: platform_distances(s, s.players[port]),
                #                        StateDataInfo.CONTINUOUS,
                #                        size=8,
                #                        scale=ObsBuilder.POS_SCALE,
                #                        player_port=port,
                #                        config=self.config),
                character=StateDataInfo(lambda s: all_chars_to_used.get(s.players[port].character, 0),
                                        StateDataInfo.CATEGORICAL,
                                        size=n_characters,
                                        player_port=port,
                                        config=self.config),
                character_stats=StateDataInfo(lambda s: self.character_data.get(s.players[port].character),
                                        StateDataInfo.CONTINUOUS,
                                        size=self.character_data.dim,
                                        player_port=port,
                                        config=self.config),
                # split general and char_specific actions
                action=StateDataInfo(lambda s: get_action_index(s, port),
                                     StateDataInfo.CATEGORICAL,
                                     size=n_actions,#+n_moves,
                                     player_port=port,
                                     config=self.config),
                action_type=StateDataInfo(lambda s: action_type(s.players[port]),
                                     StateDataInfo.CATEGORICAL,
                                     size=len(ActionType),
                                     player_port=port,
                                     config=self.config),
                # Projectiles
                # TODO : get projectiles of players (and unowned for bombs)
                # TODO: should split in two, continuous and binary!
                # projectile=StateDataInfo(lambda s: own_projectile_getter(s, port),
                #                          StateDataInfo.CONTINUOUS,
                #                          size=3,
                #                          scale=np.array([ObsBuilder.POS_SCALE, ObsBuilder.POS_SCALE, 1], dtype=np.float32),
                #                          player_port=port,
                #                          config=self.config
                #                          ),
                owned_projectiles=StateDataInfo(lambda s: get_projectiles(s, port),
                                        StateDataInfo.CONTINUOUS,
                                        size=config["obs_config"]["max_projectiles_per_owner"] * 7, # x, y, vx, vy, existence
                                        player_port=port,
                                        config=self.config,
                ),

                consecutive_hits=StateDataInfo(lambda s: 0. if "combo_counter" not in s.players[port].custom else s.players[port].custom["combo_counter"],
                                         StateDataInfo.CONTINUOUS,
                                         scale=1/self.MAX_COMBO,
                                         bounds=(0, self.MAX_COMBO),
                                         player_port=port,
                                         config=self.config
                                         ),
                # playstyle=StateDataInfo(lambda s: 0. if "playstyle" not in s.players[port].custom else s.players[port].custom["playstyle"],
                #                          StateDataInfo.CONTINUOUS,
                #                          size=num_tracked_options,
                #                          bounds=(0, 1),
                #                          player_port=port,
                #                          config=self.config
                #                          ),
                # Luigi cyclone, etc.
                character_specific=StateDataInfo(lambda s: 0. if "character_specific" not in s.players[port].custom else s.players[port].custom["character_specific"],
                                         StateDataInfo.CONTINUOUS,
                                         scale=1,
                                         bounds=(0, 1),
                                         player_port=port,
                                         config=self.config
                ),
                elo_delta=StateDataInfo(lambda s: 0. if "elo_delta" not in s.players[port].custom else (s.players[port].custom["elo_delta"] - s.players[(port % 2) + 1].custom["elo_delta"]),
                                         StateDataInfo.CONTINUOUS,
                                         scale=1/400,
                                         bounds=(-1, 1),
                                         player_port=port,
                                         config=self.config
                ),
            )

        player_value_dict = [make_player_dict(idx + 1) for idx in range(self.num_players)]

        extra_value_dict = dict(
            frame=StateDataInfo(lambda s: s.frame,
                                StateDataInfo.CONTINUOUS,
                                scale=1 / (8 * 60 * 60),
                                bounds=(0, 8 * 60 * 60),
                                config=self.config),
        )

        if not self.config["obs_config"]["stage"]:
            stage_value_dict.pop("stage")
            stage_value_dict.pop("stage_width")


        to_pop = []
        if not self.config["obs_config"]["ecb"]:
            to_pop.append("ecb")
        if not self.config["obs_config"]["character"]:
            to_pop.append("character")
        if not self.config["obs_config"]["controller_state"]:
            to_pop.extend([
                "controller_a",
                "controller_b",
                "controller_jump",
                "controller_shield",
                "controller_z",
                "controller_sticks"])
        if self.config["obs_config"]["max_projectiles_per_owner"] == 0:
            to_pop.append("owned_projectiles")
            stage_value_dict.pop("projectiles")



        for d in player_value_dict:
            for item in to_pop:
                d.pop(item)

        features = []
        all_dicts = [stage_value_dict, extra_value_dict] + player_value_dict
        for d in all_dicts:
            for k, v in d.items():
                v.name = k
                if v.is_player_dependent():
                    v.base_name = k
                    v.name += str(v.player)

                features.append(v)

        self.features = list(sorted(features, key=lambda feature: feature.nature+feature.name))
        self.initialise()

    def update(self, state: GameState, **specific):
        if len(specific) > 0:
            for feature in self.features:
                feature.advance()
                for k in specific:
                    if k == feature.name:
                        feature.update(state)
        else:
            for feature in self.features:
                feature.advance()
                feature.update(state)

        if self.config['debug']:
            print()

    def advance(self):
        for feature in self.features:
            feature.advance()

    def reset(self):
        for feature in self.features:
            feature.reset()

    def build(self, delays: dict):
        obs_dict = {}
        # if hasattr(self, "obs_dict"):
        #     return self.obs_dict

        for port in self.bot_ports:
            obs = {
                StateDataInfo.BINARY: SortedDict(),
                StateDataInfo.CATEGORICAL: SortedDict(),
                StateDataInfo.CONTINUOUS: SortedDict(),
            }
            obs["ground_truth"] = deepcopy(obs)
            self.build_for(port, obs, delays[port])
            obs_dict[port] = obs
        #self.obs_dict = obs_dict
        return obs_dict

    def build_for(self, player_idx, obs, delay):
        for feature in self.features:
            if feature.is_player_dependent():
                p = feature.player

                obs_slot = feature.base_name + ObsBuilder.player_permuts[player_idx][p]

                obs[feature.nature][obs_slot] = feature.observe(delay=delay)

                if feature.name in ["hitlag_left", "hitstun_left"]:
                    obs["ground_truth"][feature.nature][obs_slot] = feature.observe(delay=delay)
                else:
                    obs["ground_truth"][feature.nature][obs_slot] = feature.observe()
            else:
                obs[feature.nature][feature.name] = feature.observe(delay=delay)

                if feature.name in ["hitlag_left", "hitstun_left"]:
                    obs["ground_truth"][feature.nature][feature.name] = feature.observe(delay=delay)
                else:
                    obs["ground_truth"][feature.nature][feature.name] = feature.observe()


    def initialise(self):
        spec_dict = {
                nature: Dict({feature.name: feature.gym_space for feature in self.features
                              if feature.nature == nature}) for nature in (StateDataInfo.CONTINUOUS,
                                                                           StateDataInfo.BINARY,
                                                                           StateDataInfo.CATEGORICAL)
        }
        spec_dict["ground_truth"] = Dict(copy.deepcopy(spec_dict))
        self.gym_specs = Dict(spec_dict)

        game_state = GameState()
        game_state.players = {i + 1: PlayerState() for i in range(len(self.config["player_types"]))}

        for v in game_state.players.values():
            v.character = Character.FOX
        self.update(game_state)

