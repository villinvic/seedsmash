from enum import Enum

from melee import Action, PlayerState
from melee.enums import character_moves
from seedsmash.bot import BotStats


class Options(Enum):
    DODGE = 0
    RUN = 1
    JUMP = 2


class AttackOption(Enum):
    NEUTRAL = 0
    SMASH = 1
    TILT = 2
    AIR = 3


class ActionStyle(Enum):
    NEUTRAL = 0
    SHIELD = 1
    ROLL_LEFT = 2
    ROLL_RIGHT = 3
    TECH_LEFT = 4
    TECH_RIGHT = 5
    TECH_SPOT = 6
    GETUP_ATTACK = 7
    LEDGE_ATTACK = 8
    JUMP = 3

    ATTACK_FAR = 2
    ATTACK_CLOSE_GROUND = 3
    #ATTACK_


    OTHER = 8


class PlayStyleTracker:
    def __init__(self, botstats: BotStats):
        """
        Simple module to keep track of recent moves picked by the player
        """

        n = round(botstats.adaptability) * 10
        self.window = n


    def update(self, player: PlayerState):
        pass
