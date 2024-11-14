import os
from pprint import pprint
from time import time
from typing import List
from gymnasium.spaces.dict import Dict

import pandas as pd

import melee
from melee import Stage, PlayerState, Character, Action, stages, enums, Projectile, GameState, AttackState
from melee_env.enums import UsedCharacter, UsedStage
from melee.framedata import FrameData
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiBinary, Tuple, MultiDiscrete

from melee_env.enums import PlayerType
from melee_env.make_data import FrameData as FastFrameData
from melee_env.ssbm_config import SSBMConfig, SSBM_OBS_Config


all_stages_to_used = {
    Stage.YOSHIS_STORY: 0,
    Stage.BATTLEFIELD: 1,
    Stage.FINAL_DESTINATION: 2,
    Stage.POKEMON_STADIUM: 3,
    Stage.DREAMLAND: 4,
    Stage.FOUNTAIN_OF_DREAMS: 5
}
all_chars_to_used = {
    Character.MARIO: 0,
    Character.DOC: 1,
    Character.LINK: 2,
    Character.YLINK: 3,
    Character.MARTH: 4,
    Character.ROY: 5,
    Character.FALCO: 6,
    Character.FOX: 7,
    Character.CPTFALCON: 8,
    Character.GANONDORF: 9,
    Character.JIGGLYPUFF: 10,
}

#print(used_character_idx)

n_characters = len(all_chars_to_used)
n_stages = len(all_stages_to_used)

action_idx = {
    s: i for i, s in enumerate(Action)
}

action_state_idx = {
    s: i for i, s in enumerate(AttackState)
}

n_actions = len(action_idx)


def randall_position(frame, stage):
    y, x1, x2 = stages.randall_position(frame)

    if stage != Stage.YOSHIS_STORY:
        y = 0.
        x1 = 0.
        x2 = 0.

    return y, x1, x2


class StateBlock:

    def __init__(
            self,
            name,
            value_dict: dict,
            delay=0,
            debug=False,
    ):
        # value_dict = {v: (nature, size, scale) for v in values}
        self.name = name
        self.size = {
            "continuous": 0,
            "binary": 0,
            #"discrete": [],
        }
        self.registered = {}
        self.idxs = {}
        self.items = {}
        self.relative_idxs = {}

        for v, info in value_dict.items():
            self.register(v, info)

        self.delay_idx = 0
        self.delay = delay
        self.debug = debug
        self.values = {
            "continuous": np.zeros((delay + 1, self.size["continuous"]), dtype=np.float32),
            "binary": np.zeros((delay + 1, self.size["binary"]), dtype=np.int8),
            #"discrete": np.zeros((delay + 1, len(self.size["discrete"])), dtype=np.int16),
        }


    def register(self, v, info):
        self.idxs[v] = self.size[info["nature"]]
        self.items[v] = info

        def op(self, state):
            value = info["extractor"](state)
            # if np.max(np.abs(value)) * info["scale"] > 15:
            #     print("BIG VALUE", self.name, v, info, value)
            d = self.delay_idx % (self.delay+1)
            if info["nature"] == "continuous":
                self.values[info["nature"]][d, self.idxs[v]: self.idxs[v] + info["size"]]\
                    = np.float32(value) * info["scale"]
            else: # binary
                if info["size"]>1:
                    self.values[info["nature"]][d, self.idxs[v]: self.idxs[v] + info["size"]] = 0
                    if value >= 0:
                        try:
                            self.values[info["nature"]][d, self.idxs[v] + np.int32(value)] = 1
                        except:
                            print(value, self.name, v)
                else:
                    self.values[info["nature"]][d, self.idxs[v]] = np.int8(value)

            if self.debug:
                observed = (self.delay_idx % (self.delay + 1))
                observed -= self.delay
                print(self.name , v, self.relative_idxs.get(v), ":", value, "(Undelayed) -- ",
                      self.values[info["nature"]][observed, self.idxs[v]: self.idxs[v] + info["size"]], "(Observed)",
                      info["nature"])


        self.registered[v] = op

        self.size[info["nature"]] += info["size"]

    def update(self, state):

        for v, op in self.registered.items():
            op(self, state)

    def __call__(self, obs_c, idx_c, obs_b, idx_b, undelay=False):

        idx_c_next = idx_c + self.size["continuous"]
        idx_b_next = idx_b + self.size["binary"]
        
        observed = (self.delay_idx % (self.delay+1))
        if not undelay:
            observed -= self.delay
        if not self.relative_idxs:
            self.relative_idxs = {}
            for k, idx in self.idxs.items():
                if self.items[k]["nature"] == ObsBuilder.CONTINOUS:
                    self.relative_idxs[k] = (self.items[k]["nature"], np.arange(idx+idx_c, idx+idx_c+self.items[k]["size"]))
                else:
                    self.relative_idxs[k] = (self.items[k]["nature"], np.arange(idx+idx_b, idx+idx_b+self.items[k]["size"]))

        obs_c[idx_c: idx_c_next] = self.values["continuous"][observed]

        obs_b[idx_b: idx_b_next] = self.values["binary"][observed]

        return idx_c_next, idx_b_next

    def advance(self):
        self.delay_idx += 1


class StateDataInfo(dict):

    def __init__(
            self,
            extractor, nature, scale=1., size=1
    ):
        super().__init__()
        self["extractor"] = extractor
        self["nature"] = nature
        self["scale"] = scale
        self["size"] = size


class ObsBuilder:
    FRAME_SCALE = 0.01
    SPEED_SCALE = 0.5
    POS_SCALE = 0.01
    HITBOX_SCALE = 0.1
    PERCENT_SCALE = 0.009
    CHARDATA_SCALE = np.array([
        0.1, 16., 0.23, 0.05, 1.6, 4.2, 3.5, 1.35, 0.05, 0.7, 1.2
    ], dtype=np.float32)
    FD = FrameData()
    FFD = FastFrameData()
    CONTINOUS = "continuous"
    BINARY = "binary"
    DISCRETE = "discrete"

    STAGE = "stage"
    PROJECTILE = "projectile"
    ECB = "ecb"
    PLAYER = "player"
    PLAYERS = {
        i: "player" + "_" + str(i) for i in range(1,5)
    }

    PLAYER_EMBED = "player_embedding"
    CONTROLLER_STATE = "controller_state"
    EXTRA = "extra"


    def __init__(self, config, player_types):

        self.config = config
        self.name2idx = None


        # TODO online port assignement


        self.bot_ports = [i + 1 for i, p in enumerate(player_types) if p == PlayerType.BOT]
        # self.player_state_idxs = None if len(self.bot_ports) < 2 else {
        #     p: None for p in range(1, 5)
        # }

        dummy_console = melee.Console(path="tmp")

        self.char_data = {
            Character(c): np.array(list(v.values())[2:], dtype=np.float32) / ObsBuilder.CHARDATA_SCALE
            for c, v in dummy_console.characterdata.items()
        }

        dummy_console.stop()


        self.num_players = len(self.config["players"])

        stage_value_dict = dict(
                stage=                  StateDataInfo(lambda s: all_stages_to_used[s.stage],
                                        ObsBuilder.BINARY,
                                        size=n_stages,
                                        ),
                blastzone=              StateDataInfo(lambda s: stages.BLASTZONES[s.stage],
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.POS_SCALE,
                                        size=4,
                                        ),
                edge_position=          StateDataInfo(lambda s: stages.EDGE_POSITION[s.stage],
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.POS_SCALE,
                                        ),
                edge_ground_position =  StateDataInfo(lambda s: stages.EDGE_GROUND_POSITION[s.stage],
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.POS_SCALE,
                                        ),
                # tweaked side_platfrom and top_platform functions to return 0s instead of Nones
                platform_position =     StateDataInfo(lambda s: (stages.top_platform_position(s.stage)
                                        + stages.side_platform_position(right_platform=True, stage=s.stage)
                                        + stages.side_platform_position(right_platform=False, stage=s.stage)),
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.POS_SCALE,
                                        size=9,
                                        ),
                randall_position =      StateDataInfo(lambda s: randall_position(s.frame, s.stage),
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.POS_SCALE,
                                        size=3,
                                        )

            )

        def make_projectile_dict(i):
            return dict(
                speed=                  StateDataInfo(lambda s: 0. if len(s.projectiles) <= i else (
                                            s.projectiles[i].speed.x,
                                            s.projectiles[i].speed.y,
                                        ),
                                        ObsBuilder.CONTINOUS,
                                        size=2,
                                        scale=ObsBuilder.SPEED_SCALE
                                        ),
                position=               StateDataInfo(lambda s: 0. if len(s.projectiles) <= i else (
                                            s.projectiles[i].position.x,
                                            s.projectiles[i].position.y,
                                        ),
                                        ObsBuilder.CONTINOUS,
                                        size=2,
                                        scale=ObsBuilder.POS_SCALE
                                        ),
                frame_remaining=         StateDataInfo(lambda s: 0. if len(s.projectiles) <= i else
                                        np.clip(s.projectiles[i].frame, 0, 100.),
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.FRAME_SCALE
                                        ),
                owner=                  StateDataInfo(lambda s: -1 if len(s.projectiles) <= i else
                                        int(s.projectiles[i].owner) - 1,
                                        ObsBuilder.BINARY,
                                        size=4
                                        ),

            )

        projectile_value_dict = [make_projectile_dict(i) for i in range(self.config["obs"]["max_projectiles"])]

        
        def make_player_dict(port):
            """
            Helper function for indexing in lambda functions
            """
            # FD.frame_count is slower that our FFD.remaining_frame ?
            # iasa also iterates over dicts...
            frame_data_info = dict(
                attack_state=           StateDataInfo(lambda s: action_state_idx[ObsBuilder.FD.attack_state(
                                                      s.players[port].character,
                                                      s.players[port].action,
                                                      s.players[port].action_frame
                                                      )],
                                        ObsBuilder.BINARY,
                                        size=4
                                        ),
                # dj_height
                # first_hitbox_frame
                # frame_count
                frames_until_dj_apex=   StateDataInfo(lambda s: ObsBuilder.FD.frames_until_dj_apex(s.players[port]),
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.FRAME_SCALE,
                                        ),
                # hitbox_count
                iasa=                   StateDataInfo(lambda s: max(ObsBuilder.FD.iasa(s.players[port].character,
                                                                                   s.players[port].action
                                                                                   ) - s.players[port].action_frame, 0.),
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.FRAME_SCALE,
                                        ),
                is_attack=              StateDataInfo(lambda s: ObsBuilder.FD.is_attack(s.players[port].character,
                                                                                        s.players[port].action),
                                        ObsBuilder.BINARY,
                                        ),
                is_bmove=               StateDataInfo(lambda s: ObsBuilder.FD.is_bmove(s.players[port].character,
                                                                          s.players[port].action),
                                        ObsBuilder.BINARY,
                                        ),
                is_grab=                StateDataInfo(lambda s: ObsBuilder.FD.is_grab(s.players[port].character,
                                                                      s.players[port].action),
                                        ObsBuilder.BINARY,
                                        ),
                is_roll=                StateDataInfo(lambda s: ObsBuilder.FD.is_roll(s.players[port].character,
                                                                        s.players[port].action),
                                        ObsBuilder.BINARY,
                                        ),
                is_shield=              StateDataInfo(lambda s: ObsBuilder.FD.is_shield(s.players[port].action),
                                        ObsBuilder.BINARY,
                                        ),
                project_hit_location=   StateDataInfo(lambda s: ObsBuilder.FD.project_hit_location(s.players[port],
                                                                                                   s.stage,
                                                                                                   ),
                                        ObsBuilder.CONTINOUS,
                                        size=3,
                                        scale=ObsBuilder.POS_SCALE
                                        ),
                # range_backward
                # range_forward
                roll_end_position=      StateDataInfo(lambda s: ObsBuilder.FD.roll_end_position(s.players[port],
                                                                                                s.stage,
                                                                                                ),
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.POS_SCALE
                                        ),

                # slide_distance=         StateDataInfo(lambda s: ObsBuilder.FD.slide_distance(s.players[port],
                #                                                                              s.players[port].speed_ground_x_self +
                #                                                                              s.players[port].speed_x_attack,
                #                                                                                 10),
                #                         ObsBuilder.CONTINOUS,
                #                         scale=ObsBuilder.POS_SCALE
                #                         ),

            )
            in_range_attack = {
                f"in_range_{other_port}": StateDataInfo(lambda s: ObsBuilder.FD.in_range(s.players[port],
                                                                                       s.players[other_port],
                                                                                       s.stage
                                                                                       ),
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.FRAME_SCALE
                                        )
                for other_port in range(1, self.num_players+1) if other_port != port
            }

            
            return dict(
                character_info=         StateDataInfo(lambda s: self.char_data[s.players[port].character],
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.CHARDATA_SCALE,
                                        size=11,
                                        ),
                percent=                StateDataInfo(lambda s: s.players[port].percent,
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.PERCENT_SCALE,
                                        ),
                shield_strength=        StateDataInfo(lambda s: s.players[port].shield_strength,
                                        ObsBuilder.CONTINOUS,
                                        scale=0.017,
                                        ),
                stock=                  StateDataInfo(lambda s: s.players[port].stock,
                                        ObsBuilder.CONTINOUS,
                                        scale=0.25,
                                        ),
                action_frame=           StateDataInfo(lambda s: self.FFD.remaining_frame(s.players[port].character,
                                                                             s.players[port].action,
                                                                             s.players[port].action_frame),
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.FRAME_SCALE
                                        ),
                facing=                 StateDataInfo(lambda s: np.int32(s.players[port].facing),
                                        ObsBuilder.BINARY,
                                        size=2,
                                        ),
                invulnerable=           StateDataInfo(lambda s: s.players[port].invulnerable,
                                        ObsBuilder.BINARY,
                                        ),
                invulnerability_left=   StateDataInfo(lambda s: s.players[port].invulnerability_left,
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.FRAME_SCALE
                                        ),
                hitlag_left=            StateDataInfo(lambda s: s.players[port].hitlag_left,
                                        ObsBuilder.CONTINOUS,
                                        scale=0.1
                                        ),
                hitstun_left=           StateDataInfo(lambda s: s.players[port].hitstun_frames_left,
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.FRAME_SCALE
                                        ),
                on_ground=              StateDataInfo(lambda s: s.players[port].on_ground,
                                        ObsBuilder.BINARY,
                                        size=2,
                                        ),
                speed_air_x_self=       StateDataInfo(lambda s: s.players[port].speed_air_x_self,
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.SPEED_SCALE,
                                        ),
                speed_y_self=           StateDataInfo(lambda s: s.players[port].speed_y_self,
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.SPEED_SCALE,
                                        ),
                speed_x_attack=         StateDataInfo(lambda s: s.players[port].speed_x_attack,
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.SPEED_SCALE,
                                        ),
                speed_y_attack=         StateDataInfo(lambda s: s.players[port].speed_y_attack,
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.SPEED_SCALE,
                                        ),
                speed_ground_x_self=    StateDataInfo(lambda s: s.players[port].speed_ground_x_self,
                                        ObsBuilder.CONTINOUS,
                                        scale=ObsBuilder.SPEED_SCALE,
                                        ),
                off_stage=              StateDataInfo(lambda s: s.players[port].off_stage,
                                        ObsBuilder.BINARY,
                                        ),
                moonwalk=               StateDataInfo(lambda s: s.players[port].moonwalkwarning,
                                        ObsBuilder.BINARY,
                                        ),
                powershield=            StateDataInfo(lambda s: s.players[port].is_powershield,
                                        ObsBuilder.BINARY,
                                        ),
                jumps_left=             StateDataInfo(lambda s: s.players[port].jumps_left,
                                        ObsBuilder.BINARY,
                                        size=7
                                        ),
                button_a=               StateDataInfo(lambda s:
                                        s.players[port].controller_state.button[enums.Button.BUTTON_A],
                                        ObsBuilder.BINARY
                                        ),
                button_b=               StateDataInfo(lambda s:
                                        s.players[port].controller_state.button[enums.Button.BUTTON_B],
                                        ObsBuilder.BINARY
                                        ),
                button_jump=            StateDataInfo(lambda s:
                                        int(
                                            s.players[port].controller_state.button[enums.Button.BUTTON_X]
                                            or
                                            s.players[port].controller_state.button[enums.Button.BUTTON_Y]
                                        ),
                                        ObsBuilder.BINARY
                                        ),
                button_shield=          StateDataInfo(lambda s:
                                        int(
                                            s.players[port].controller_state.button[enums.Button.BUTTON_L]
                                            or
                                            s.players[port].controller_state.button[enums.Button.BUTTON_R]
                                        ),
                                        ObsBuilder.BINARY
                                        ),
                button_z=               StateDataInfo(lambda s:
                                        s.players[port].controller_state.button[enums.Button.BUTTON_Z],
                                        ObsBuilder.BINARY
                                        ),
                sticks=                 StateDataInfo(lambda s:
                                        s.players[port].controller_state.main_stick
                                        +s.players[port].controller_state.c_stick,
                                        ObsBuilder.CONTINOUS,
                                        size=4
                                        ),
                ecb=                    StateDataInfo(lambda s: (
                                        s.players[port].ecb.top.y, s.players[port].ecb.top.x,
                                        s.players[port].ecb.right.y, s.players[port].ecb.right.x,
                                        s.players[port].ecb.bottom.y, s.players[port].ecb.bottom.x,
                                        s.players[port].ecb.left.y, s.players[port].ecb.left.x,
                                        ),
                                        ObsBuilder.CONTINOUS,
                                        size=8,
                                        scale=ObsBuilder.POS_SCALE
                                        ),
                position=               StateDataInfo(lambda s: (s.players[port].position.x, s.players[port].position.y),
                                        ObsBuilder.CONTINOUS,
                                        size=2,
                                        scale=ObsBuilder.POS_SCALE
                                        ),
                **frame_data_info,
                **in_range_attack
            )
        
        player_value_dict = [make_player_dict(idx+1) for idx in range(self.num_players)]

        def make_player_char_embed_dict(port):
            return dict(
                character=              StateDataInfo(lambda s: all_chars_to_used[s.players[port].character],
                                        ObsBuilder.CONTINOUS,
                                        size=1,
                                        )
            )
        player_char_embedding_dict = [make_player_char_embed_dict(idx+1) for idx in range(self.num_players)]
        
        def make_player_as_embed_dict(port):
            return dict(
                action=                 StateDataInfo(lambda s: action_idx[s.players[port].action],
                                        ObsBuilder.CONTINOUS,
                                        size=1
                                        )
            )
        player_as_embedding_dict = [make_player_as_embed_dict(idx+1) for idx in range(self.num_players)]

        extra_value_dict = dict(
            frame=              StateDataInfo(lambda s: 0. if "game_frame" not in s.custom else s.custom["game_frame"],
                                ObsBuilder.CONTINOUS,
                                scale=1 / (8 * 60 * 60),
                                ),
        )

        if not self.config["obs"]["stage"]:
            stage_value_dict = {}
        if not self.config["obs"]["ecb"]:
            for d in player_value_dict:
                d.pop("ecb")
        if not self.config["obs"]["character"]:
            for d in player_value_dict:
                d.pop("character", None)
                d.pop("character_info")
        if not self.config["obs"]["controller_state"]:
            for d in player_value_dict:
                d.pop("button_a")
                d.pop("button_b")
                d.pop("button_jump")
                d.pop("button_shield")
                d.pop("button_z")
                d.pop("sticks")

        self._stage = [StateBlock(
            name=ObsBuilder.STAGE,
            value_dict=stage_value_dict,
            debug=self.config["debug"]

        )]

        self._players = [StateBlock(
            name=ObsBuilder.PLAYER + f"_{i+1}",
            value_dict=player_value_dict[i],
            delay=self.config["obs"]["delay"],
            debug=self.config["debug"]
        ) for i in range(self.num_players)]

        self._players_char_embeddings = [StateBlock(
            name=ObsBuilder.PLAYER_EMBED + f"_char_{i+1}",
            value_dict=player_char_embedding_dict[i],
            delay=0,
            debug=self.config["debug"]
        ) for i in range(self.num_players)]

        self._players_as_embeddings = [StateBlock(
            name=ObsBuilder.PLAYER_EMBED + f"_as_{i + 1}",
            value_dict=player_as_embedding_dict[i],
            delay=self.config["obs"]["delay"],
            debug=self.config["debug"]
        ) for i in range(self.num_players)]

        self._projectiles = [StateBlock(
            name=ObsBuilder.PROJECTILE + f"_{i}",
            value_dict=projectile_value_dict[i],
            debug=self.config["debug"]
        ) for i in range(self.config["obs"]["max_projectiles"])]

        self._extra = [StateBlock(
            name=ObsBuilder.EXTRA,
            value_dict=extra_value_dict,
            debug=self.config["debug"]
        )]

        self._blocks = self._stage + self._projectiles + self._players + self._players_char_embeddings \
                       + self._players_as_embeddings + self._extra
        self.non_embedding_blocks = self._stage + self._projectiles + self._players + self._extra
        self._base_blocks = self._stage + self._projectiles + self._extra
        self._port_specific_blocks = self._players + self._players_char_embeddings + self._players_as_embeddings

        self._player_blocks = {
            block.name: block for block in self._players
        }
        self._player_char_embedding_blocks = {
            block.name: block for block in self._players_char_embeddings
        }
        self._player_as_embedding_blocks = {
            block.name: block for block in self._players_as_embeddings
        }

        self.initialize()

    def update(self, state: GameState, **specific):
        if len(specific)>0:
            for b in self._blocks:
                b.advance()
                for k in specific:
                    if k in b.registered:
                        b.registered[k](b, state)

        else:
            for b in self._base_blocks:
                b.advance()
                b.update(state)
            if len(state.players) == self.num_players:
                for b in self._port_specific_blocks:
                    b.advance()
                    b.update(state)
            else:
                print("Weird state there.")

        if self.config['debug']:
            print()

    def advance(self):
        for b in self._blocks:
            b.advance()

    def build(self, obs_dict):
        for idx, obs in obs_dict.items():
            self.build_for(idx, obs)
        return obs_dict

    def build_for(self, player_idx, obs):
        if player_idx == 1:
            p1 = 1
            p2 = 2
            p3 = 3
            p4 = 4
        elif player_idx == 2:
            p1 = 2
            p2 = 1
            p3 = 3
            p4 = 4
        elif player_idx == 3:
            p1 = 3
            p2 = 4
            p3 = 1
            p4 = 2
        else:
            p1 = 4
            p2 = 3
            p3 = 2
            p4 = 1

        (obs_c,
         obs_b,
         obs_char,
         obs_action_state
         ) = obs
        (idx_c,
         idx_b,
         idx_char,
         idx_as
         ) = (0,
              0,
              0,
              0)

        for base_block in self._base_blocks:
            idx_c, idx_b = base_block(obs_c, idx_c, obs_b, idx_b)

        for i, block_name in enumerate([
            ObsBuilder.PLAYERS[p1],
            ObsBuilder.PLAYERS[p2],
            ObsBuilder.PLAYERS[p3],
            ObsBuilder.PLAYERS[p4]
        ]):
            if block_name in self._player_blocks:
                idx_c, idx_b = self._player_blocks[block_name](obs_c, idx_c, obs_b, idx_b,
                                                               # We assume that we can accurately predict ourselves
                                                               undelay=i==0)
        for i, block_name in enumerate([
            ObsBuilder.PLAYER_EMBED + f"_char_{p1}",
            ObsBuilder.PLAYER_EMBED + f"_char_{p2}",
            ObsBuilder.PLAYER_EMBED + f"_char_{p3}",
            ObsBuilder.PLAYER_EMBED + f"_char_{p4}",
        ]):
            if block_name in self._player_char_embedding_blocks:
                idx_char, _ = self._player_char_embedding_blocks[block_name]\
                    (obs_char, idx_char, obs_b, idx_b,
                    # We assume that we can accurately predict ourselves
                    undelay=i==0)
                
        for i, block_name in enumerate([
            ObsBuilder.PLAYER_EMBED + f"_as_{p1}",
            ObsBuilder.PLAYER_EMBED + f"_as_{p2}",
            ObsBuilder.PLAYER_EMBED + f"_as_{p3}",
            ObsBuilder.PLAYER_EMBED + f"_as_{p4}",
        ]):
            if block_name in self._player_as_embedding_blocks:
                idx_as, _ = self._player_as_embedding_blocks[block_name]\
                    (obs_action_state, idx_as, obs_b, idx_b,
                    # We assume that we can accurately predict ourselves
                    undelay=i==0)

        np.clip(obs_c, -20., 20., out=obs_c)



    def initialize(self):
        size_c, size_b = 0, 0
        for block in self.non_embedding_blocks:
            c, b = block.size.values()
            size_c += c
            size_b += b

        character_embedding = Box(low=0, high=n_characters-1, shape=(self.num_players,), dtype=np.float32)
        action_embedding = Box(low=0, high=n_actions-1, shape=(self.num_players,), dtype=np.float32) # MultiDiscrete([n_actions, n_actions], dtype=np.int16)

        self.gym_specs = Dict({
            i: Tuple([
                Box(low=-30., high=30., shape=(size_c,)),
                MultiBinary(size_b),
                character_embedding,
                action_embedding
            ])
            for i in self.bot_ports
        })

        game_state = GameState()
        game_state.projectiles = [Projectile() for _ in range(self.config["obs"]["max_projectiles"])]
        game_state.players = {i + 1: PlayerState() for i in range(len(self.config["players"]))}
        # game_state.players[1].stock = 16
        # game_state.players[1].action = Action.DOWN_B_GROUND
        # game_state.players[1].action_frame = 3

        for v in game_state.players.values():
            v.character = Character.MARIO

        self.update(game_state)
        self.build(self.gym_specs.sample())

        self.name2idx = {}
        for block in self._stage + self._extra:
            self.name2idx.update(
                {k: v if nature == ObsBuilder.CONTINOUS else v + size_c for k, (nature, v) in block.relative_idxs.items()}
            )
        for block in self._projectiles + self._players:
            self.name2idx.update(
                {block.name + "_" + k: v if nature == ObsBuilder.CONTINOUS else v + size_c
                 for k, (nature, v) in block.relative_idxs.items()}
            )

    def __getitem__(self, item):
        """
        #   TODO: Unsupported for projectiles
        :param item: feature to look for
        :return: Returns the index in the tuple obs and the
        list of indices where you find that item in the post-process observation
        """
        return self.name2idx.get(item)

    def get_player_obs_idx(self, item, player_port):

        return self.name2idx.get(ObsBuilder.PLAYERS[player_port] + "_" + item)



if __name__ == '__main__':
    obs_config = (
        SSBM_OBS_Config()
        # .character()
        .ecb()
        .stage()
        .max_projectiles(3)
        .controller_state()
        .delay(2)
    )

    env_config = (
        SSBMConfig()
        .chars([
            Character.FOX,
            Character.FALCO,
            Character.MARIO,
            Character.DOC,
            Character.MARTH,
            Character.ROY,
            Character.CPTFALCON,
            Character.GANONDORF,
            Character.JIGGLYPUFF,
        ])
        .stages([
            Stage.FINAL_DESTINATION,
            Stage.YOSHIS_STORY,
            Stage.POKEMON_STADIUM,
            Stage.BATTLEFIELD,
            Stage.DREAMLAND
        ])
        .players([PlayerType.BOT, PlayerType.BOT])
        .n_eval(0)
        .set_obs_conf(obs_config)

        .render()
        .debug()
    )

    ob = ObsBuilder(env_config, env_config["players"])
    print(ob.gym_specs)
    print(ob.name2idx)
    print(ob.get_player_obs_idx("stock", 1))


    # game_state = GameState()
    # game_state.projectiles = [Projectile() for _ in range(env_config["obs"]["max_projectiles"])]
    # game_state.players = {i + 1: PlayerState() for i in range(len(env_config["players"]))}
    # for v in game_state.players.values():
    #     v.character = Character.MARIO
    #
    # dummies = ob.gym_specs.sample()
    #
    # ob.update(game_state)
    # ob.build(dummies)
    # print(dummies)
    # print("ok")
    # ob.update(game_state)
    # ob.build(dummies)
    #
    # print(dummies)

    # pprint(len(out))
