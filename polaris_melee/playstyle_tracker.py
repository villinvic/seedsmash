from enum import Enum
from typing import Tuple, List

from melee import Action, PlayerState, GameState, Character, YoshiMoves
from polaris_melee.actions import MOVEMENT_ACTIONS, JUMP_ACTIONS, AERIAL_MOVEMENT_ACTIONS, DODGE_ACTIONS, \
    CROUCH_ACTIONS, NO_GROUND_TECH_ACTIONS, LEDGE_ATTACK_ACTIONS, LEDGE_NEUTRAL_ACTIONS, LEDGE_JUMP_ACTIONS, \
    LEDGE_ROLL_ACTIONS, FALLING_ACTIONS, NEUTRAL_GETUP_ACTIONS, BACKWARD_GETUP_ACTIONS, FORWARD_GETUP_ACTIONS, is_shield
from polaris_melee.compiled_libmelee_framedata import CompiledFrameData


class NeutralOptions(Enum):
    DODGE = 0
    GROUND_MOVEMENT = 1
    AERIAL_MOVEMENT = 2
    JUMP = 3
    AERIAL_ATTACK = 4
    GROUND_ATTACK = 5
    SHIELD = 6
    GRAB = 7
    CROUCH = 8

class GroundTechOptions(Enum):
    NO_TECH = 0
    NEUTRAL = 1
    LEFT = 2
    RIGHT = 3
    # We can ignore wall techs I think

class LedgeOptions(Enum):
    DROP = 0
    NEUTRAL = 1
    ATTACK = 2
    ROLL = 3
    JUMP = 4

class GetupOptions(Enum):
    NEUTRAL = 0
    ATTACK = 1
    LEFT = 2
    RIGHT = 3
    
class DodgeOptions(Enum):
    NEUTRAL = 0
    LEFT = 2
    RIGHT = 3

# DI options are annoying to compute


class PlayStyleTrackerModule:
    def __init__(
            self,
            framedata: CompiledFrameData,
            botstats: "BotStats",
            options: type[Enum],
            max_window_size: int = 120,
            min_window_size: int = 3,
            max_distance: int = 50,
    ):
        """
        Simple module to keep track of recent moves picked by the player
        """
        # with more adaptability, the tracker adapts faster to the player playstyle.
        adaptability = 0 if botstats is None else botstats.adaptability
        rigidity = round((100 - adaptability) / 100)

        window_size = min_window_size + (max_window_size - min_window_size) * rigidity
        self.ema_factor = 1 - 1 / window_size

        self.max_distance = max_distance

        self.option_weights = {
            opt: 1 / len(options)
            for opt in options
        }
        self.framedata = framedata
        self.last_option = None

        self.dim = len(options)


    def update(self, gamestate: GameState, player: PlayerState):
        # count options only when players are near.
        if gamestate.distance > 50:
            return

        new_option = self._update(gamestate, player)
        if new_option is None or new_option == self.last_option:
            return

        self.last_option = new_option
        self.update_weight(new_option)

    def _update(self, gamestate: GameState, player: PlayerState):
        pass

    def update_weight(self, curr_option):

        self.option_weights = {
            opt: w * self.ema_factor + float(curr_option == opt) * (1 - self.ema_factor)
            for opt, w in self.option_weights.items()
        }

    def get_weights(self):
        return [w for w in self.option_weights.values()]

class NeutralOptionsTracker(PlayStyleTrackerModule):
    def __init__(
            self,
            **kwargs
    ):

        super().__init__(
            max_window_size=150,
            min_window_size=10,
            options=NeutralOptions,
            **kwargs,
        )

    def _update(self, gamestate: GameState, player: PlayerState):
        option = None
        if self.framedata.is_grab(player.character, player.action):
            option = NeutralOptions.GRAB
        elif self.framedata.is_attack(player.character, player.action):
            option = NeutralOptions.GROUND_ATTACK if player.on_ground else NeutralOptions.AERIAL_ATTACK
        elif is_shield(player):
            option = NeutralOptions.SHIELD
        elif player.action in MOVEMENT_ACTIONS:
            option = NeutralOptions.GROUND_MOVEMENT
        elif player.action in AERIAL_MOVEMENT_ACTIONS:
            option = NeutralOptions.AERIAL_MOVEMENT
        elif player.action in JUMP_ACTIONS:
            option = NeutralOptions.JUMP
        elif player.action in DODGE_ACTIONS:
            option = NeutralOptions.DODGE
        elif player.action in CROUCH_ACTIONS:
            option = NeutralOptions.CROUCH

        return option


class GroundTechOptionsTracker(PlayStyleTrackerModule):
    def __init__(
            self,
            **kwargs
    ):

        super().__init__(
            **kwargs,
            max_window_size=50,
            min_window_size=6,
            options=GroundTechOptions,
        )

    def _update(self, gamestate: GameState, player: PlayerState):
        option = None

        if player.action == Action.NEUTRAL_TECH:
            option = GroundTechOptions.NEUTRAL
        elif player.action == Action.FORWARD_TECH:
            option = GroundTechOptions.RIGHT if player.facing else GroundTechOptions.LEFT
        elif player.action == Action.BACKWARD_TECH:
            option = GroundTechOptions.LEFT if player.facing else GroundTechOptions.RIGHT
        elif player.action in NO_GROUND_TECH_ACTIONS:
            option = GroundTechOptions.NO_TECH

        return option


class LedgeOptionsTracker(PlayStyleTrackerModule):
    def __init__(
            self,
            **kwargs
    ):

        super().__init__(
            **kwargs,
            max_window_size=50,
            min_window_size=6,
            options=LedgeOptions,
        )
        self.prev_player_state = None

    def update(self, gamestate: GameState, player: PlayerState):
        super().update(gamestate, player)
        self.prev_player_state = player

    def _update(self, gamestate: GameState, player: PlayerState):
        option = None

        if player.action in LEDGE_ATTACK_ACTIONS:
            option = LedgeOptions.ATTACK
        elif player.action in LEDGE_NEUTRAL_ACTIONS:
            option = LedgeOptions.NEUTRAL
        elif player.action in LEDGE_JUMP_ACTIONS:
            option = LedgeOptions.JUMP
        elif player.action in LEDGE_ROLL_ACTIONS:
            option = LedgeOptions.ROLL
        elif self.prev_player_state == Action.EDGE_HANGING and player.action in FALLING_ACTIONS:
            option = LedgeOptions.DROP

        return option


class GetupOptionsTracker(PlayStyleTrackerModule):
    def __init__(
            self,
            **kwargs
    ):

        super().__init__(
            **kwargs,
            max_window_size=50,
            min_window_size=6,
            options=GetupOptions,
        )

    def _update(self, gamestate: GameState, player: PlayerState):
        option = None

        if player.action == Action.GETUP_ATTACK:
            option = GetupOptions.ATTACK
        elif player.action in FORWARD_GETUP_ACTIONS:
            option = GetupOptions.RIGHT if player.facing else GetupOptions.LEFT
        elif player.action in BACKWARD_GETUP_ACTIONS:
            option = GetupOptions.LEFT if player.facing else GetupOptions.RIGHT
        elif player.action in NEUTRAL_GETUP_ACTIONS:
            option = GetupOptions.NEUTRAL

        return option


class DodgeOptionsTracker(PlayStyleTrackerModule):
    def __init__(
            self,
            **kwargs
    ):

        super().__init__(
            **kwargs,
            max_window_size=50,
            min_window_size=6,
            options=DodgeOptions,
        )

    def _update(self, gamestate: GameState, player: PlayerState):
        option = None

        if player.action == Action.SPOTDODGE:
            option = DodgeOptions.NEUTRAL
        elif player.action == Action.ROLL_FORWARD:
            option = DodgeOptions.RIGHT if player.facing else DodgeOptions.LEFT
        elif player.action in Action.ROLL_FORWARD:
            option = DodgeOptions.LEFT if player.facing else DodgeOptions.RIGHT

        return option


class PlayStyleTracker:
    def __init__(
            self,
            framedata: CompiledFrameData,
            botstats: "BotStats" = None
    ):

        self.trackers = [
            tracker_cls(framedata=framedata, botstats=botstats)
            for tracker_cls in (NeutralOptionsTracker, GroundTechOptionsTracker, LedgeOptionsTracker, GetupOptionsTracker)
        ]

        self.dim = sum(tracker.dim for tracker in self.trackers)

    def update(self, gamestate: GameState, player: PlayerState) -> List[float]:
        for tracker in self.trackers:
            tracker.update(gamestate, player)

        return self.get_weights()

    def get_weights(self) -> List[float]:
        weights = []
        for tracker in self.trackers:
            weights += tracker.get_weights()
        return weights

    def print(self):
        for tracker in self.trackers:
            print(tracker.__class__.__name__, ":")
            for opt, w in tracker.option_weights.items():
                print(opt.name, ":", f"{w:.2f}")
