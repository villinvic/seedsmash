
from gymnasium.spaces.dict import Dict
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

# print(used_character_idx)

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

class StateDataInfo:
    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"

    HARD_BOUNDS = (-5., 5)

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
        self.value = self.init_values()
        self.gym_space = self.get_gym_space()
        self.update = self.build_op()

        self.player = player_port

    def is_player_dependent(self):
        return not self.player is None

    def init_values(self):
        if self.nature == StateDataInfo.CONTINUOUS:
            dtype = np.float32
        elif self.nature in (StateDataInfo.BINARY, StateDataInfo.CATEGORICAL):
            dtype = np.float32
        else:
            dtype = np.float32
        #dtype = np.float32 if self.nature in (StateDataInfo.CONTINUOUS, StateDataInfo.CATEGORICAL) else np.int8

        return np.zeros(
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


class ObsBuilder:
    FRAME_SCALE = 0.01
    SPEED_SCALE = 0.2
    POS_SCALE = 0.01
    HITBOX_SCALE = 0.1
    PERCENT_SCALE = 0.009
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

        # TODO online port assignement

        self.bot_ports = [i + 1 for i, p in enumerate(player_types) if p == PlayerType.BOT]
        # self.player_state_idxs = None if len(self.bot_ports) < 2 else {
        #     p: None for p in range(1, 5)
        # }

        self.num_players = len(self.config["players"])

        def projectile_getter(state, port, op=lambda proj: proj.position.x):
            projectiles_properties = [op(p) for p in state.projectiles if p.owner == port]
            if len(projectiles_properties) < self.config["obs"]["max_projectiles"]:
                projectiles_properties.extend(
                    [0.] * (self.config["obs"]["max_projectiles"] - len(projectiles_properties)))

            return projectiles_properties[:self.config["obs"]["max_projectiles"]]

        def make_projectile_dict(port):
            if port == -1:
                port = None

            return dict(
                projectile_x=StateDataInfo(
                    lambda s: projectile_getter(s, port, op=lambda proj: proj.position.x),
                    StateDataInfo.CONTINUOUS,
                    scale=ObsBuilder.POS_SCALE,
                    size=self.config["obs"]["max_projectiles"],
                    config=self.config,
                    player_port=port
                ),
                projectile_y=StateDataInfo(
                    lambda s: projectile_getter(s, port, op=lambda proj: proj.position.y),
                    StateDataInfo.CONTINUOUS,
                    scale=ObsBuilder.POS_SCALE,
                    size=self.config["obs"]["max_projectiles"],
                    config=self.config,
                    player_port=port
                ),
                projectile_x_speed=StateDataInfo(
                    lambda s: projectile_getter(s, port, op=lambda proj: proj.x_speed),
                    StateDataInfo.CONTINUOUS,
                    scale=ObsBuilder.SPEED_SCALE,
                    size=self.config["obs"]["max_projectiles"],
                    config=self.config,
                    player_port=port
                ),
                projectile_y_speed=StateDataInfo(
                    lambda s: projectile_getter(s, port, op=lambda proj: proj.y_speed),
                    StateDataInfo.CONTINUOUS,
                    scale=ObsBuilder.SPEED_SCALE,
                    size=self.config["obs"]["max_projectiles"],
                    config=self.config,
                    player_port=port
                ),

                projectile_remaining_frames=StateDataInfo(
                    lambda s: projectile_getter(s, port, op=lambda proj: np.clip(proj.frame, 0, 300.)),
                    StateDataInfo.CONTINUOUS,
                    scale=ObsBuilder.FRAME_SCALE,
                    size=self.config["obs"]["max_projectiles"],
                    config=self.config,
                    player_port=port
                )
            )

        stage_value_dict = dict(
            stage=StateDataInfo(lambda s: all_stages_to_used[s.stage],
                                StateDataInfo.CATEGORICAL,
                                size=n_stages,
                                config=self.config,
                                ),
            blastzone=StateDataInfo(lambda s: stages.BLASTZONES[s.stage],
                                    StateDataInfo.CONTINUOUS,
                                    scale=ObsBuilder.POS_SCALE,
                                    size=4,
                                    config=self.config,
                                    ),
            edge_position=StateDataInfo(lambda s: stages.EDGE_POSITION[s.stage],
                                        StateDataInfo.CONTINUOUS,
                                        scale=ObsBuilder.POS_SCALE,
                                        config=self.config,
                                        ),
            edge_ground_position=StateDataInfo(lambda s: stages.EDGE_GROUND_POSITION[s.stage],
                                               StateDataInfo.CONTINUOUS,
                                               scale=ObsBuilder.POS_SCALE,
                                               config=self.config,
                                               ),
            # tweaked side_platfrom and top_platform functions to return 0s instead of Nones
            platform_position=StateDataInfo(lambda s: (stages.top_platform_position(s)
                                                       + stages.side_platform_position(right_platform=True,
                                                                                       gamestate=s)
                                                       + stages.side_platform_position(right_platform=False,
                                                                                       gamestate=s)),
                                            StateDataInfo.CONTINUOUS,
                                            scale=ObsBuilder.POS_SCALE,
                                            size=9,
                                            config=self.config,
                                            ),
            randall_position=StateDataInfo(lambda s: randall_position(s.frame, s.stage),
                                           StateDataInfo.CONTINUOUS,
                                           scale=ObsBuilder.POS_SCALE,
                                           size=3,
                                           config=self.config,
                                           ),
            **make_projectile_dict(-1)

        )

        def make_player_dict(port):
            """
            Helper function for indexing in lambda functions
            """
            # FD.frame_count is slower that our FFD.remaining_frame ?
            # iasa also iterates over dicts...
            frame_data_info = dict(
                attack_state=StateDataInfo(lambda s: action_state_idx[ObsBuilder.FD.attack_state(
                    s.players[port].character,
                    s.players[port].action,
                    s.players[port].action_frame
                )],
                                           StateDataInfo.CATEGORICAL,
                                           size=4,
                                           player_port=port,
                                           config=self.config),
                # dj_height
                # first_hitbox_frame
                # frame_count
                # frames_until_dj_apex=StateDataInfo(lambda s: ObsBuilder.FD.frames_until_dj_apex(s.players[port]),
                #                                    StateDataInfo.CONTINUOUS,
                #                                    scale=ObsBuilder.FRAME_SCALE,
                #                                    player_port=port,
                #                                    config=self.config),

                # hitbox_count
                iasa=StateDataInfo(lambda s: np.int8(ObsBuilder.FD.iasa(s.players[port].character,
                                                                    s.players[port].action
                                                                    ) - s.players[port].action_frame < 3),
                                   StateDataInfo.BINARY,
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
                is_grab=StateDataInfo(lambda s: ObsBuilder.FD.is_grab(s.players[port].character,
                                                                      s.players[port].action),
                                      StateDataInfo.BINARY,
                                      player_port=port,
                                      config=self.config),
                is_roll=StateDataInfo(lambda s: ObsBuilder.FD.is_roll(s.players[port].character,
                                                                      s.players[port].action),
                                      StateDataInfo.BINARY,
                                      player_port=port,
                                      config=self.config),
                is_shield=StateDataInfo(lambda s: ObsBuilder.FD.is_shield(s.players[port].action),
                                        StateDataInfo.BINARY,
                                        player_port=port,
                                        config=self.config),
                project_hit_location=StateDataInfo(lambda s: ObsBuilder.FD.project_hit_location(s.players[port],
                                                                                                s,
                                                                                                ),
                                                   StateDataInfo.CONTINUOUS,
                                                   size=3,
                                                   scale=ObsBuilder.POS_SCALE,
                                                   player_port=port,
                                                   config=self.config),
                # range_backward
                # range_forward
                # roll_end_position=StateDataInfo(lambda s: ObsBuilder.FD.roll_end_position(s.players[port],
                #                                                                           s.stage,
                #                                                                           ),
                #                                 StateDataInfo.CONTINUOUS,
                #                                 scale=ObsBuilder.POS_SCALE,
                #                                 player_port=port,
                #                                 config=self.config),
            )

            def check_if_not_reaching(state, p1, p2):
                reaching_frame = ObsBuilder.FD.in_range(state.players[p1],
                                                        state.players[p2],
                                                        state.stage
                                                        )
                return -1 / ObsBuilder.FRAME_SCALE if reaching_frame == 0 else reaching_frame

            # TODO : Adapt to 4 players
            in_range_attack = {
                f"in_range": StateDataInfo(lambda s: check_if_not_reaching(s, port, other_port),
                                           StateDataInfo.CONTINUOUS,
                                           scale=ObsBuilder.FRAME_SCALE,
                                           player_port=port,
                                           config=self.config)
                for i, other_port in enumerate(range(1, self.num_players + 1)) if other_port != port
            }

            return dict(
                percent=StateDataInfo(lambda s: s.players[port].percent,
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
                                    StateDataInfo.CONTINUOUS,
                                    scale=0.25,
                                    player_port=port,
                                    config=self.config),
                action_frame=StateDataInfo(lambda s: self.FFD.remaining_frame(s.players[port].character,
                                                                              s.players[port].action,
                                                                              s.players[port].action_frame),
                                           StateDataInfo.CONTINUOUS,
                                           scale=ObsBuilder.FRAME_SCALE,
                                           player_port=port,
                                           config=self.config),
                facing=StateDataInfo(lambda s: np.int32(s.players[port].facing),
                                     StateDataInfo.CATEGORICAL,
                                     size=2,
                                     player_port=port,
                                     config=self.config),
                invulnerable=StateDataInfo(lambda s: s.players[port].invulnerable,
                                           StateDataInfo.BINARY,
                                           player_port=port,
                                           config=self.config),
                invulnerability_left=StateDataInfo(lambda s: s.players[port].invulnerability_left,
                                                   StateDataInfo.CONTINUOUS,
                                                   scale=ObsBuilder.FRAME_SCALE,
                                                   player_port=port,
                                                   config=self.config),
                hitlag_left=StateDataInfo(lambda s: s.players[port].hitlag_left,
                                          StateDataInfo.CONTINUOUS,
                                          scale=0.1,
                                          player_port=port,
                                          config=self.config),
                hitstun_left=StateDataInfo(lambda s: s.players[port].hitstun_frames_left,
                                           StateDataInfo.CONTINUOUS,
                                           scale=ObsBuilder.FRAME_SCALE,
                                           player_port=port,
                                           config=self.config),
                on_ground=StateDataInfo(lambda s: s.players[port].on_ground,
                                        StateDataInfo.CATEGORICAL,
                                        size=2,
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
                off_stage=StateDataInfo(lambda s: s.players[port].off_stage,
                                        StateDataInfo.BINARY,
                                        player_port=port,
                                        config=self.config),
                # moonwalk=StateDataInfo(lambda s: s.players[port].moonwalkwarning,
                #                        StateDataInfo.BINARY,
                #                        player_port=port,
                #                        config=self.config),
                powershield=StateDataInfo(lambda s: s.players[port].is_powershield,
                                          StateDataInfo.BINARY,
                                          player_port=port,
                                          config=self.config),
                jumps_left=StateDataInfo(lambda s: s.players[port].jumps_left,
                                         StateDataInfo.CONTINUOUS,
                                         scale=0.14,
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
                position=StateDataInfo(lambda s: (s.players[port].position.x, s.players[port].position.y),
                                       StateDataInfo.CONTINUOUS,
                                       size=2,
                                       scale=ObsBuilder.POS_SCALE,
                                       player_port=port,
                                       config=self.config),
                character=StateDataInfo(lambda s: all_chars_to_used[s.players[port].character],
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

                **frame_data_info,
                **in_range_attack,
                **make_projectile_dict(port)
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

        features = []
        all_dicts = [stage_value_dict, extra_value_dict] + player_value_dict
        for d in all_dicts:
            for k, v in d.items():
                v.name = k
                if v.is_player_dependent():
                    v.base_name = k
                    v.name += str(v.player)

                features.append(v)

        self.features = sorted(features, key=lambda feature: feature.nature+feature.name)

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

    def build(self, obs_dict):
        for idx, obs in obs_dict.items():
            self.build_for(idx, obs)
        return obs_dict

    def build_for(self, player_idx, obs):

        for feature in self.features:
            if feature.is_player_dependent():
                p = feature.player

                obs_slot = feature.base_name + ObsBuilder.player_permuts[player_idx][p]


                obs[feature.nature][obs_slot] = feature.observe(undelay=p == player_idx)
            else:
                obs[feature.nature][feature.name] = feature.observe(undelay=True)

    def initialize(self):

        # TODO : this is getting sorted, so our indexes are wrong

        self.gym_specs = Dict({
                nature: Dict({feature.name: feature.gym_space for feature in self.features
                              if feature.nature == nature}) for nature in (StateDataInfo.CONTINUOUS,
                                                                           StateDataInfo.BINARY,
                                                                           StateDataInfo.CATEGORICAL)
        })

        game_state = GameState()
        game_state.projectiles = [Projectile() for _ in range(self.config["obs"]["max_projectiles"])]
        game_state.players = {i + 1: PlayerState() for i in range(len(self.config["players"]))}
        # game_state.players[1].action = Action.DOWN_B_GROUND
        # game_state.players[1].action_frame = 3

        for v in game_state.players.values():
            v.character = Character.FOX

        self.update(game_state)
        dummy = {aid: self.gym_specs.sample() for aid in self.bot_ports}
        self.build(dummy)

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
