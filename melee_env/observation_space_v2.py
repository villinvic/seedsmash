import copy
from collections import defaultdict

from optree.functools import partial
from scipy.special.cython_special import radian
from sortedcontainers import SortedDict
from gymnasium.spaces.dict import Dict
from melee import Stage, PlayerState, Character, Action, stages, enums, Projectile, GameState, AttackState

from melee_env.compiled_libmelee_framedata import CompiledFrameData
from melee_env.enums import UsedCharacter, UsedStage
from melee.framedata import FrameData
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiBinary, Tuple, MultiDiscrete

from melee_env.enums import PlayerType
from melee_env.make_data import FrameData as FastFrameData
from melee_env.ssbm_config import SSBMConfig, SSBM_OBS_Config

action_idx = {
    s: i for i, s in enumerate(Action)
}
idx_to_action = {
    i: s for i, s in enumerate(Action)
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


class StateDataInfo:
    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"

    HARD_BOUNDS = (-10., 10)

    def __init__(
            self, extractor, nature, name="UNSET", scale=1., size=1, player_port=None, config={}
    ):
        self.name = name
        self.base_name = None  # utility for player dependency
        self.extractor = extractor
        self.nature = nature
        self.scale = scale
        self.size = size if nature != StateDataInfo.CATEGORICAL else 1
        self.n_values = None if nature != StateDataInfo.CATEGORICAL else size
        self.delay = config["obs"]["delay"]
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

    def observe(self, undelay=False):
        observed = (self.delay_idx % (self.delay + 1))
        if not undelay:
            observed -= self.delay

        if self.debug:
            print(self.name, self.value[self.delay_idx % (self.delay + 1)], "(undelayed)", self.value[observed], "(observed)")

        if self.nature == StateDataInfo.CONTINUOUS:
            return np.clip(self.value[observed] * self.scale, *StateDataInfo.HARD_BOUNDS)

        return self.value[observed]

    def reset(self):
        self.value[:] = 0
        self.delay_idx = 0

    def get_gym_space(self):
        if self.nature == StateDataInfo.CONTINUOUS:
            return Box(*StateDataInfo.HARD_BOUNDS, (self.size,), dtype=np.float32)
        elif self.nature == StateDataInfo.CATEGORICAL:
            return Box(0, self.n_values - 1, (1,), dtype=np.float32)
        elif self.nature == StateDataInfo.BINARY:
            return MultiBinary(self.size)
        else:
            raise NotImplementedError

    def build_op(self):
        def op(state):
            extracted = self.extractor(state)
            self.value[self.delay_idx % (self.delay + 1), :] = extracted
            # except Exception as e:
            #     print("failed to extract values for: ", self.name, e)

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

    def __init__(self, config, player_types):

        self.config = config
        self.name2idx = None

        all_stages_to_used = {
            s: i
            for i, s in enumerate(config["stages"])
        }
        all_chars_to_used = {
            s: i
            for i, s in enumerate(config["chars"])
        }
        used_to_chars = {
            i: s
            for i, s in enumerate(config["chars"])
        }

        n_characters = len(all_chars_to_used)
        n_stages = len(all_stages_to_used)

        # TODO online port assignement

        self.bot_ports = [i + 1 for i, p in enumerate(player_types) if p == PlayerType.BOT]


        self.num_players = len(self.config["players"])

        def projectile_dist(p: Projectile, player: PlayerState):
            return np.sqrt(np.square(p.position.x - player.position.x) + np.square(p.position.y - player.position.y))

        def own_projectile_getter(state, port):
            other_port = 1 + port % 2
            # we want the projectile that is the closest to other_port
            own_projectiles = [
                p for p in state.projectiles if p.owner in (port, -1)
            ]
            if len(own_projectiles) > 0:
                nearest_own_projectile = sorted(
                    own_projectiles, key=lambda p: projectile_dist(p, state.players[other_port])
                )[0]
                return nearest_own_projectile.position.x, nearest_own_projectile.position.y, 1.
            else:
                return 0., 0., 0.


        stage_value_dict = dict(
            stage=StateDataInfo(lambda s: all_stages_to_used.get(s.stage, 0),
                                StateDataInfo.CATEGORICAL,
                                size=n_stages,
                                config=self.config,
                                ),
            # blastzone=StateDataInfo(lambda s: stages.BLASTZONES[s.stage],
            #                         StateDataInfo.CONTINUOUS,
            #                         scale=ObsBuilder.POS_SCALE,
            #                         size=4,
            #                         config=self.config,
            #                         ),
            # edge_position=StateDataInfo(lambda s: stages.EDGE_POSITION[s.stage],
            #                             StateDataInfo.CONTINUOUS,
            #                             scale=ObsBuilder.POS_SCALE,
            #                             config=self.config,
            #                             ),
            # edge_ground_position=StateDataInfo(lambda s: stages.EDGE_GROUND_POSITION[s.stage],
            #                                    StateDataInfo.CONTINUOUS,
            #                                    scale=ObsBuilder.POS_SCALE,
            #                                    config=self.config,
            #                                    ),
            # tweaked side_platfrom and top_platform functions to return 0s instead of Nones
            # platform_position=StateDataInfo(lambda s: (stages.top_platform_position(s)
            #                                            + stages.side_platform_position(right_platform=True,
            #                                                                            gamestate=s)
            #                                            + stages.side_platform_position(right_platform=False,
            #                                                                            gamestate=s)),
            #                                 StateDataInfo.CONTINUOUS,
            #                                 scale=ObsBuilder.POS_SCALE,
            #                                 size=9,
            #                                 config=self.config,
            #                                 ),
            # randall_position=StateDataInfo(lambda s: randall_position(s.frame, s.stage),
            #                                StateDataInfo.CONTINUOUS,
            #                                scale=ObsBuilder.POS_SCALE,
            #                                size=3,
            #                                config=self.config,
            #                                ),
        )

        def make_player_dict(port):
            """
            Helper function for indexing in lambda functions
            """
            # FD.frame_count is slower that our FFD.remaining_frame ?
            # iasa also iterates over dicts...

            other_port = 1 + (port % 2)
            def check_if_not_reaching(state, p1, p2):
                reaching_frame = ObsBuilder.FD.in_range(state.players[p1],
                                                        state.players[p2],
                                                        state.stage
                                                        )
                return -1 / 0.01 if reaching_frame == -1 else reaching_frame -1

            def frames_before_next_hitbox(char_state):
                next_hitbox_frame = ObsBuilder.FD.frames_before_next_hitbox(char_state.character,
                                                                            char_state.action,
                                                                            char_state.action_frame
                                                        )

                return -1 * 50 if next_hitbox_frame == -1 else next_hitbox_frame-1


            def iasa_or_actionable(s):
                frames_before_iasa = np.clip(ObsBuilder.FD.iasa[s.players[port].character][s.players[port].action]
                                       - s.players[port].action_frame, 1, 51) - 1
                if frames_before_iasa == 0:
                    frames_before_iasa = -50

                return frames_before_iasa


            frame_data_info = dict(
                # in_range=StateDataInfo(
                #     lambda s: check_if_not_reaching(s, port, other_port),
                #     StateDataInfo.CONTINUOUS,
                #     scale=0.04,
                #     player_port=port,
                #     config=self.config
                # ),
                # attack_state=StateDataInfo(lambda s: action_state_idx[ObsBuilder.FD.attack_state(
                #     s.players[port].character,
                #     s.players[port].action,
                #     s.players[port].action_frame
                # )],
                #                            StateDataInfo.CATEGORICAL,
                #                            size=4,
                #                            player_port=port,
                #                            config=self.config),
                # dj_height
                # first_hitbox_frame
                # frame_count
                # frames_until_dj_apex=StateDataInfo(lambda s: ObsBuilder.FD.frames_until_dj_apex(s.players[port]),
                #                                    StateDataInfo.CONTINUOUS,
                #                                    scale=ObsBuilder.FRAME_SCALE,
                #                                    player_port=port,
                #                                    config=self.config),

                # hitbox_count
                iasa=StateDataInfo(lambda s: np.maximum(0., ObsBuilder.FD.iasa[s.players[port].character][
                                                                    s.players[port].action]
                                                                     - s.players[port].action_frame),
                                   StateDataInfo.CONTINUOUS,
                                   scale=self.FRAME_SCALE,
                                   player_port=port,
                                   config=self.config),
                frames_before_next_hitbox=StateDataInfo(lambda s: np.minimum(frames_before_next_hitbox(
                    s.players[port]
                ), 50.),
                                   StateDataInfo.CONTINUOUS,
                                   scale=ObsBuilder.FRAME_SCALE,
                                   player_port=port,
                                   config=self.config),

                # is_attack=StateDataInfo(lambda s: ObsBuilder.FD.is_attack(s.players[port].character,
                #                                                           s.players[port].action),
                #                         StateDataInfo.BINARY,
                #                         player_port=port,
                #                         config=self.config),
                # is_bmove=StateDataInfo(lambda s: ObsBuilder.FD.is_bmove(s.players[port].character,
                #                                                         s.players[port].action),
                #                        StateDataInfo.BINARY,
                #                        player_port=port,
                #                        config=self.config),
                # is_grab=StateDataInfo(lambda s: ObsBuilder.FD.is_grab(s.players[port].character,
                #                                                       s.players[port].action),
                #                       StateDataInfo.BINARY,
                #                       player_port=port,
                #                       config=self.config),
                # is_roll=StateDataInfo(lambda s: ObsBuilder.FD.is_roll(s.players[port].character,
                #                                                       s.players[port].action),
                #                       StateDataInfo.BINARY,
                #                       player_port=port,
                #                       config=self.config),
                # is_shield=StateDataInfo(lambda s: ObsBuilder.FD.is_shield(s.players[port].action),
                #                         StateDataInfo.BINARY,
                #                         player_port=port,
                #                         config=self.config),
                # project_hit_location=StateDataInfo(lambda s: ObsBuilder.FD.project_hit_location(s.players[port],
                #                                                                                 s,
                #                                                                                 ),
                #                                    StateDataInfo.CONTINUOUS,
                #                                    size=3,
                #                                    scale=ObsBuilder.POS_SCALE,
                #                                    player_port=port,
                #                                    config=self.config),
                # range_backward
                # range_forward
                # roll_end_position=StateDataInfo(lambda s: ObsBuilder.FD.FD.roll_end_position(s.players[port],
                #                                                                           s,
                #                                                                           ),
                #                                 StateDataInfo.CONTINUOUS,
                #                                 scale=ObsBuilder.POS_SCALE,
                #                                 player_port=port,
                #                                 config=self.config),
            )


            def get_ledge_distance(s):
                dr = np.clip(s.players[port].position.x - stages.EDGE_POSITION[s.stage], -60., 60.)
                dl = np.clip(s.players[port].position.x + stages.EDGE_POSITION[s.stage], -60., 60.)
                return dr, dl

            def get_side_blastzone_distance(s):
                dr = np.clip(s.players[port].position.x - stages.BLASTZONES_X[s.stage], -60., 60.)
                dl = np.clip(s.players[port].position.x - (-stages.BLASTZONES_X[s.stage]), -60., 60.)
                return dr, dl

            def get_vertical_blastzone_distance(s):
                upper_y, lower_y = stages.BLASTZONES_Y[s.stage]
                dr = np.clip(s.players[port].position.y - upper_y, -60., 60.)
                dl = np.clip(s.players[port].position.y - lower_y, -60., 60.)
                return dr, dl

            def get_platform_distance(s: GameState):
                # returns y dist as well as (x dist to left and right edges of platforms)
                x = s.players[port].position.x
                y = s.players[port].position.y

                topy, topx1, topx2 = stages.top_platform_position(s.stage)
                dtopy = np.clip(y - topy, -50., 50.)
                dtopx1 = np.clip(x - topx1, -50., 50.)
                dtopx2 = np.clip(x - topx2, -50., 50.)

                lefty, leftx1, leftx2 = stages.side_platform_position(right_platform=False,
                                                 gamestate=s)
                dlefty = np.clip(y - lefty, -50., 50.)
                dleftx1 = np.clip(x - leftx1, -50., 50.)
                dleftx2 = np.clip(x - leftx2, -50., 50.)

                righty, rightx1, rightx2 = stages.side_platform_position(right_platform=True,
                                                 gamestate=s)
                drighty = np.clip(y - righty, -50., 50.)
                drightx1 = np.clip(x - rightx1, -50., 50.)
                drightx2 = np.clip(x - rightx2, -50., 50.)

                return (dtopy, dtopx1, dtopx2, dlefty, dleftx1, dleftx2, drighty, drightx1, drightx2)

            def get_randall_distance(s):
                if s.stage != Stage.YOSHIS_STORY:
                    return 0., 0., 0.
                y, x1, x2 = randall_position(s.frame, s.stage)
                dy = np.clip(s.players[port].position.y-y, -50., 50.)
                dx1 = np.clip(s.players[port].position.x-x1, -50., 50.)
                dx2 = np.clip(s.players[port].position.x-x2, -50., 50.)
                return dy, dx1, dx2


            def hitstun_or_actionable(s):
                hitstun_left = np.clip(s.players[port].hitstun_frames_left, 1, 51) - 1
                if hitstun_left == 0:
                    hitstun_left = -50

                return hitstun_left

            def hitlag_or_actionable(s):
                hitlag_left = np.clip(s.players[port].hitlag_left, 1, 51) - 1
                if hitlag_left == 0:
                    hitlag_left = -50

                return hitlag_left


            return dict(
                percent=StateDataInfo(lambda s: s.players[port].percent,
                                      # putting other_port inseast could be a trick to help combos,
                                      # but prevents overreacting to hitstun, etc.
                                      # however, action_frame, action and other stuff is leaking info about ourself
                                      StateDataInfo.CONTINUOUS,
                                      scale=ObsBuilder.PERCENT_SCALE,
                                      player_port=port,
                                      config=self.config),
                shield_strength=StateDataInfo(lambda s: s.players[port].shield_strength,
                                              StateDataInfo.CONTINUOUS,
                                              scale=0.017,
                                              player_port=port,
                                              config=self.config),
                stock=StateDataInfo(lambda s: s.players[port].stock,
                                    StateDataInfo.CATEGORICAL,
                                    size=4,
                                    player_port=port,
                                    config=self.config),
                action_frame=StateDataInfo(lambda s: np.clip(self.FFD.remaining_frame(s.players[port].character,
                                                                              s.players[port].action,
                                                                              s.players[port].action_frame),1, 51) - 1,
                                           StateDataInfo.CONTINUOUS,
                                           scale=ObsBuilder.FRAME_SCALE,
                                           player_port=port,
                                           config=self.config),
                facing=StateDataInfo(lambda s: np.int32(s.players[port].facing),
                                     StateDataInfo.BINARY,
                                     player_port=port,
                                     config=self.config),
                invulnerable=StateDataInfo(lambda s: s.players[port].invulnerable,
                                           StateDataInfo.BINARY,
                                           player_port=port,
                                           config=self.config),
                # invulnerability_left=StateDataInfo(lambda s: s.players[port].invulnerability_left,
                #                                    StateDataInfo.CONTINUOUS,
                #                                    scale=ObsBuilder.FRAME_SCALE,
                #                                    player_port=port,
                #                                    config=self.config),
                hitlag_left=StateDataInfo(hitlag_or_actionable,
                                          StateDataInfo.CONTINUOUS,
                                          scale=ObsBuilder.FRAME_SCALE,
                                          player_port=port,
                                          config=self.config),
                hitstun_left=StateDataInfo(hitstun_or_actionable,
                                           StateDataInfo.CONTINUOUS,
                                           scale=ObsBuilder.FRAME_SCALE,
                                           player_port=port,
                                           config=self.config),
                on_ground=StateDataInfo(lambda s: s.players[port].on_ground,
                                        StateDataInfo.BINARY,
                                        size=1,
                                        player_port=port,
                                        config=self.config),
                # speed_air_x_self=StateDataInfo(lambda s: s.players[port].speed_air_x_self,
                #                                StateDataInfo.CONTINUOUS,
                #                                scale=ObsBuilder.SPEED_SCALE,
                #                                player_port=port,
                #                                config=self.config),
                # speed_y_self=StateDataInfo(lambda s: s.players[port].speed_y_self,
                #                            StateDataInfo.CONTINUOUS,
                #                            scale=ObsBuilder.SPEED_SCALE,
                #                            player_port=port,
                #                            config=self.config),
                # speed_x_attack=StateDataInfo(lambda s: s.players[port].speed_x_attack,
                #                              StateDataInfo.CONTINUOUS,
                #                              scale=ObsBuilder.SPEED_SCALE,
                #                              player_port=port,
                #                              config=self.config),
                # speed_y_attack=StateDataInfo(lambda s: s.players[port].speed_y_attack,
                #                              StateDataInfo.CONTINUOUS,
                #                              scale=ObsBuilder.SPEED_SCALE,
                #                              player_port=port,
                #                              config=self.config),
                # speed_ground_x_self=StateDataInfo(lambda s: s.players[port].speed_ground_x_self,
                #                                   StateDataInfo.CONTINUOUS,
                #                                   scale=ObsBuilder.SPEED_SCALE,
                #                                   player_port=port,
                #                                   config=self.config),
                # off_stage=StateDataInfo(lambda s: s.players[port].off_stage,
                #                         StateDataInfo.BINARY,
                #                         player_port=port,
                #                         config=self.config),
                # moonwalk=StateDataInfo(lambda s: s.players[port].moonwalkwarning,
                #                        StateDataInfo.BINARY,
                #                        player_port=port,
                #                        config=self.config),
                powershield=StateDataInfo(lambda s: s.players[port].is_powershield,
                                          StateDataInfo.BINARY,
                                          player_port=port,
                                          config=self.config),
                jumps_left=StateDataInfo(lambda s: s.players[port].jumps_left,
                                         StateDataInfo.CATEGORICAL,
                                         size=6,
                                         player_port=port,
                                         config=self.config),
                button_a=StateDataInfo(lambda s:
                                       s.players[port].controller_state.button[enums.Button.BUTTON_A],
                                       StateDataInfo.BINARY,
                                       player_port=port,
                                       config=self.config),
                button_b=StateDataInfo(lambda s:
                                       s.players[port].controller_state.button[enums.Button.BUTTON_B],
                                       StateDataInfo.BINARY,
                                       player_port=port,
                                       config=self.config),
                button_jump=StateDataInfo(lambda s:
                                          int(
                                              s.players[port].controller_state.button[enums.Button.BUTTON_X]
                                              or
                                              s.players[port].controller_state.button[enums.Button.BUTTON_Y]
                                          ),
                                          StateDataInfo.BINARY,
                                          player_port=port,
                                          config=self.config),
                button_shield=StateDataInfo(lambda s:
                                            int(
                                                s.players[port].controller_state.button[enums.Button.BUTTON_L]
                                                or
                                                s.players[port].controller_state.button[enums.Button.BUTTON_R]
                                            ),
                                            StateDataInfo.BINARY,
                                            player_port=port,
                                            config=self.config),
                button_z=StateDataInfo(lambda s:
                                       s.players[port].controller_state.button[enums.Button.BUTTON_Z],
                                       StateDataInfo.BINARY,
                                       player_port=port,
                                       config=self.config),
                sticks=StateDataInfo(lambda s:
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
                position=StateDataInfo(lambda s: (ObsBuilder.FD.FD.roll_end_position(s.players[port], s), s.players[port].position.y),
                                       StateDataInfo.CONTINUOUS,
                                       size=2,
                                       scale=ObsBuilder.POS_SCALE,
                                       player_port=port,
                                       config=self.config),
                # ledge_distance=StateDataInfo(get_ledge_distance,
                #                        StateDataInfo.CONTINUOUS,
                #                        size=2,
                #                        scale=1/50,
                #                        player_port=port,
                #                        config=self.config,
                # ),
                # side_blastzone_distance=StateDataInfo(get_side_blastzone_distance,
                #                        StateDataInfo.CONTINUOUS,
                #                        size=2,
                #                        scale=1/60,
                #                        player_port=port,
                #                        config=self.config,
                # ),
                # vertical_blastzone_distance=StateDataInfo(get_vertical_blastzone_distance,
                #                                       StateDataInfo.CONTINUOUS,
                #                                       size=2,
                #                                       scale=1/60,
                #                                       player_port=port,
                #                                       config=self.config,
                # ),
                # platform_distances=StateDataInfo(get_platform_distance,
                #                                           StateDataInfo.CONTINUOUS,
                #                                           size=9,
                #                                           scale=1/50,
                #                                           player_port=port,
                #                                           config=self.config,
                #                                           ),
                # randall_distances=StateDataInfo(get_randall_distance,
                #                                  StateDataInfo.CONTINUOUS,
                #                                  size=3,
                #                                  scale=1/50,
                #                                  player_port=port,
                #                                  config=self.config,
                #                                  ),
                character=StateDataInfo(lambda s: all_chars_to_used.get(s.players[port].character, 0),
                                        StateDataInfo.CATEGORICAL,
                                        size=n_characters,
                                        player_port=port,
                                        config=self.config),
                action=StateDataInfo(lambda s: action_idx[s.players[port].action],
                                     StateDataInfo.CATEGORICAL,
                                     size=n_actions,
                                     player_port=port,
                                     config=self.config),
                # Projectiles
                # TODO : get projectiles of players, and unowned projectiles
                # TODO: what are unowned projectiles again ?
                projectile=StateDataInfo(lambda s: own_projectile_getter(s, port),
                                         StateDataInfo.CONTINUOUS,
                                         size=3,
                                         scale=np.array([ObsBuilder.POS_SCALE, ObsBuilder.POS_SCALE, 1], dtype=np.float32),
                                         player_port=port,
                                         config=self.config
                                         ),

                **frame_data_info,
                #**in_range_attack,
            )

        player_value_dict = [make_player_dict(idx + 1) for idx in range(self.num_players)]

        extra_value_dict = dict(
            frame=StateDataInfo(lambda s: 0. if "game_frame" not in s.custom else s.custom["game_frame"],
                                StateDataInfo.CONTINUOUS,
                                scale=1 / (8 * 60 * 60),
                                config=self.config),
        )

        if not self.config["obs"]["stage"]:
            stage_value_dict = {}
        if not self.config["obs"]["ecb"]:
            for d in player_value_dict:
                d.pop("ecb")
        if not self.config["obs"]["character"]:
            for d in player_value_dict:
                d.pop("character", None)
        if not self.config["obs"]["controller_state"]:
            for d in player_value_dict:
                d.pop("button_a")
                d.pop("button_b")
                d.pop("button_jump")
                d.pop("button_shield")
                d.pop("button_z")
                d.pop("sticks")
        if self.config["obs"]["max_projectiles"] == 0:
            for d in player_value_dict:
                to_pop = []
                for k in d:
                    if "projectile" in k:
                        to_pop.append(k)
                for k in to_pop:
                    d.pop(k)

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

        def get_relative_position(obs):
            return np.clip((obs[StateDataInfo.CONTINUOUS]["position1"] - obs[StateDataInfo.CONTINUOUS]["position2"])/self.POS_SCALE, -10., 10.)

        def get_hitbox_dist(obs, p=1):
            other_p = 1 + (p % 2)
            x, y = obs[StateDataInfo.CONTINUOUS][f"position{other_p}"] / self.POS_SCALE
            opp_char_size = float(self.FD.FD.characterdata[used_to_chars.get(obs[StateDataInfo.CATEGORICAL][f"character{other_p}"][0], Character.FOX)]["size"])
            y += opp_char_size

            xs, ys = obs[StateDataInfo.CONTINUOUS][f"position{p}"] / self.POS_SCALE

            # TODO: care about the value here
            facing = obs[StateDataInfo.CONTINUOUS][f"facing{p}"][0]
            char = used_to_chars.get(obs[StateDataInfo.CATEGORICAL][f"character{p}"][0], Character.FOX)
            action = idx_to_action.get(obs[StateDataInfo.CATEGORICAL][f"action{p}"][0], Action.UNKNOWN_ANIMATION)

            infos = self.FD.get_move_next_hitboxes(
                char,
                action,
                self.FFD[char][action] - int(obs[StateDataInfo.CONTINUOUS][f"action_frame{p}"] / (self.FRAME_SCALE * 2))
            )
            min_abs_dx = np.inf
            min_dy = 0. #3. / self.POS_SCALE
            min_abs_dy = np.inf
            min_dx = 0. #3. / self.POS_SCALE
            has_hitboxes = 0

            pseudo_half_radius = opp_char_size * 0.05


            for xa, ya, r in infos:
                xa = facing * xa + xs
                ya = ya + ys
                #print(f"yattack of p{p}", ya, f"y of p{other_p}", y, f"size of char{other_p}", opp_char_size)

                has_hitboxes = 1
                vect = np.array([x-xa, y-ya])
                unit_v = vect / np.linalg.norm(vect)
                radian_angle = np.arccos(
                    np.clip(
                        np.dot(unit_v, np.array([1, 0])),
                        -1., 1.
                    )
                )

                dx = (x + pseudo_half_radius * np.cos(radian_angle + np.pi)) - (xa + r * np.cos(radian_angle))
                if vect[0] * dx <= 0:
                    dx = 0.
                dy = (y + pseudo_half_radius * np.sin(radian_angle + np.pi)) - (ya + r * np.sin(radian_angle))
                if vect[1] * dy <= 0:
                    dy = 0.

                adx = abs(dx)
                ady = abs(dy)
                if adx < min_abs_dx:
                    min_abs_dx = adx
                    min_dx = dx
                if ady < min_abs_dy:
                    min_abs_dy = ady
                    min_dy = dy

            r = np.clip(np.array([min_dx, min_dy]), -50., 50.)
            return r

        post_process_features = [
            # PostProcessFeature(get_relative_position,
            #                                 StateDataInfo.CONTINUOUS,
            #                                 scale=0.1,
            #                                 size=2,
            #                                 name="relative_position",
            #                                 config=self.config,
            #                                 ),
            # PostProcessFeature(partial(get_hitbox_dist, p=1),
            #                    StateDataInfo.CONTINUOUS,
            #                    scale=1/50,
            #                    size=2,
            #                    name="hitbox_dist1",
            #                    config=self.config,
            # ),
            # PostProcessFeature(partial(get_hitbox_dist, p=2),
            #                    StateDataInfo.CONTINUOUS,
            #                    scale=1/50,
            #                    size=2,
            #                    name="hitbox_dist2",
            #                    config=self.config,
            #                    )
        ]

        self.post_process_features = list(sorted(
            post_process_features, key=lambda feature: feature.nature + feature.name
        ))

        self.initialize()

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

    def build(self, curr_ports):
        obs_dict = {}
        for port in self.bot_ports:
            curr_port = curr_ports[port]
            obs_dict[curr_port] = {
                StateDataInfo.BINARY: SortedDict(),
                StateDataInfo.CATEGORICAL: SortedDict(),
                StateDataInfo.CONTINUOUS: SortedDict(),
                # "ground_truth": {
                #     StateDataInfo.BINARY: SortedDict(),
                #     StateDataInfo.CATEGORICAL: SortedDict(),
                #     StateDataInfo.CONTINUOUS: SortedDict(),
                # }
            }
            self.build_for(port, obs_dict[curr_port])
        return obs_dict

    def build_for(self, player_idx, obs):
        for feature in self.features:
            if feature.is_player_dependent():
                p = feature.player

                obs_slot = feature.base_name + ObsBuilder.player_permuts[player_idx][p]

                obs[feature.nature][obs_slot] = feature.observe(undelay=p == player_idx)
                #obs["ground_truth"][feature.nature][obs_slot] = feature.observe(undelay=True)
            else:
                obs[feature.nature][feature.name] = feature.observe(undelay=True)
                #obs["ground_truth"][feature.nature][feature.name] = feature.observe(undelay=True)

        for pp_feature in self.post_process_features:
            value = pp_feature.observe(obs)
            obs[pp_feature.nature][pp_feature.name] = value
            #true_value = pp_feature.observe(obs["ground_truth"])
            #obs["ground_truth"][pp_feature.nature][pp_feature.name] = true_value

    def initialize(self):

        # TODO : this is getting sorted, so our indexes are wrong

        spec_dict = {
                nature: Dict({feature.name: feature.gym_space for feature in self.features + self.post_process_features
                              if feature.nature == nature}) for nature in (StateDataInfo.CONTINUOUS,
                                                                           StateDataInfo.BINARY,
                                                                           StateDataInfo.CATEGORICAL)
        }
        #spec_dict["ground_truth"] = Dict(copy.deepcopy(spec_dict))
        self.gym_specs = Dict(spec_dict)

        game_state = GameState()
        game_state.players = {i + 1: PlayerState() for i in range(len(self.config["players"]))}
        # game_state.players[1].action = Action.DOWN_B_GROUND
        # game_state.players[1].action_frame = 3

        for v in game_state.players.values():
            v.character = Character.FOX

        self.update(game_state)

        # self.name2idx = {}
        #
        # checker = DictFlatteningPreprocessor(obs_space=self.gym_specs)
        # idx = 0
        # for feature in self.features:
        #     self.name2idx[feature.name] = idx
        #     idx += feature.size

        #self.flattened_obs = checker.transform(dummy)

    def __getitem__(self, item):
        """
        #   TODO: Unsupported for projectiles
        :param item: feature to look for
        :return: Returns the index in the tuple obs and the
        list of indices where you find that item in the post-process observation
        """
        return self.name2idx.get(item)

    def get_player_obs_idx(self, item, player_port):

        return self.name2idx.get(item + str(player_port))


if __name__ == '__main__':
    obs_config = (
        SSBM_OBS_Config()
        .character()
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
        #.debug()
    )

    ob = ObsBuilder(env_config, env_config["players"])

    #pprint(ob.gym_specs)
    # pprint(ob.name2idx)
    # pprint(ob.get_player_obs_idx("stock", 1))

    game_state = GameState()
    game_state.projectiles = [Projectile() for _ in range(env_config["obs"]["max_projectiles"])]
    game_state.players = {i + 1: PlayerState() for i in range(len(env_config["players"]))}
    for v in game_state.players.values():
        v.character = Character.MARIO
    game_state.stage = Stage.FINAL_DESTINATION
    game_state.custom["game_frame"] = 33
    game_state.players[1].action = Action.GRABBED

    dummies = ob.gym_specs.sample()
    ob.update(game_state)
    ob.build(dummies)

    #flattened_obs = DictFlatteningPreprocessor(ob.gym_specs).transform(dummies)
    #print(flattened_obs, np.where(flattened_obs==Action.GRABBED.value))
    print(ob.get_player_obs_idx("action", 1))
    # print("one",dummies)
    # ob.update(game_state)
    # ob.build(dummies)
    # print("two",dummies)
    # ob.update(game_state)
    # ob.build(dummies)
    # print("three",dummies)

    # ob.update(game_state)
    # ob.build(dummies)
    #
    # print("here2")
    # print(dummies)

    # print("ok")
    # ob.update(game_state)
    # ob.build(dummies)
    #
    # print(dummies)

    # pprint(len(out))
