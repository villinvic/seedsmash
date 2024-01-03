from enum import Enum

class PlayerType(Enum):
    """A VS-mode stage """
    HUMAN = 0
    BOT = 1
    CPU = 2
    HUMAN_DEBUG = 3

class UsedCharacter(Enum):
    """A Melee character ID.

    Note:
        Numeric values are 'internal' IDs."""
    MARIO = 0
    FOX = 1
    CPTFALCON = 2
    DK = 3
    KIRBY = 4
    BOWSER = 5
    LINK = 6
    SHEIK = 7
    NESS = 8
    PEACH = 9
    POPO = 10
    PIKACHU = 11
    SAMUS = 12
    YOSHI = 13
    JIGGLYPUFF = 14
    MEWTWO = 15
    LUIGI = 16
    MARTH = 17
    ZELDA = 18
    YLINK = 19
    DOC = 20
    FALCO = 21
    PICHU = 22
    GAMEANDWATCH = 23
    GANONDORF = 24
    ROY = 25


class UsedStage(Enum):
    """A VS-mode stage """
    FINAL_DESTINATION = 0
    BATTLEFIELD = 1
    POKEMON_STADIUM = 2
    DREAMLAND = 3
    FOUNTAIN_OF_DREAMS = 4  # not used for now
    YOSHIS_STORY = 5