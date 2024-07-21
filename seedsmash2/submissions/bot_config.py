from typing import NamedTuple, Union
from melee.enums import Character
from dataclasses import dataclass
import inspect
from dataclasses import fields

color2index = {
    Character.BOWSER : {
        "green": 0,
        "red": 1,
        "blue": 2,
        "black": 3
    },
    Character.CPTFALCON : {
        "green": 4,
        "red": 2,
        "purple": 2,
        "blue": 5,
        "black": 1,
        "white": 3,
        "pink": 3
    },
    Character.DK: {
        "green": 4,
        "red": 2,
        "purple": 3,
        "blue": 3,
        "black": 1,
    },
    Character.DOC: {
        "green": 3,
        "red": 1,
        "blue": 2,
        "black": 4,
    },
    Character.FALCO: {
        "white": 0,
        "green": 3,
        "red": 1,
        "blue": 2,
    },
    Character.FOX: {
        "white": 0,
        "green": 3,
        "red": 1,
        "orange": 1,
        "purple": 2,
        "blue": 2,
    },
    Character.GANONDORF: {
        "green": 3,
        "red": 1,
        "purple": 4,
        "blue": 2,
    },
    Character.POPO: {
        "green": 1,
        "red": 3,
        "orange": 2,
    },
    Character.JIGGLYPUFF: {
        "green": 3,
        "red": 1,
        "blue": 2,
        "black": 4,
    },
    Character.KIRBY: {
        "green": 4,
        "red": 3,
        "white": 5,
        "blue": 2,
        "yellow": 1,
    },
    Character.LINK: {
        "green": 0,
        "red": 1,
        "white": 4,
        "blue": 2,
        "black": 3,
    },
    Character.LUIGI: {
        "green": 0,
        "red": 3,
        "white": 1,
        "blue": 2,
    },
    Character.MARIO: {
        "green": 4,
        "red": 0,
        "black": 2,
        "blue": 3,
        "yellow": 1,
    },
    Character.MARTH: {
        "green": 2,
        "red": 1,
        "blue": 0,
        "black": 3,
        "white": 4
    },
    Character.MEWTWO: {
        "green": 3,
        "red": 1,
        "orange": 1,
        "blue": 2,
    },
    Character.GAMEANDWATCH: {
        "green": 3,
        "red": 1,
        "blue": 2,
        "black": 0,
    },
    Character.NESS: {
        "green": 3,
        "red": 0,
        "blue": 2,
        "yellow": 1
    },
    Character.PEACH: {
        "green": 4,
        "red": 1,
        "yellow": 1,
        "orange": 1,
        "blue": 3,
        "white": 2
    },
    Character.PICHU: {
        "yellow": 0,
        "red": 1,
        "blue": 2,
        "green": 3
    },
    Character.PIKACHU: {
        "green": 3,
        "red": 1,
        "blue": 2,
        "yellow": 0
    },
    Character.ROY: {
        "green": 3,
        "red": 1,
        "blue": 2,
        "white": 2,
        "yellow": 4,
    },
    Character.SAMUS: {
        "green": 3,
        "red": 1,
        "orange": 0,
        "pink": 1,
        "blue": 4,
        "purple": 4,
        "black": 2,
    },
    Character.YOSHI: {
        "green": 0,
        "red": 1,
        "blue": 2,
        "yellow": 3,
        "pink": 4,
        "lightblue": 5
    },
    Character.YLINK: {
        "green": 0,
        "red": 1,
        "blue": 2,
        "black": 4,
        "white": 3
    },
    Character.ZELDA: {
        "pink": 0,
        "green": 3,
        "red": 1,
        "blue": 2,
        "white": 4,
        "purple": 4
    },
    Character.SHEIK: {
        "pink": 0,
        "green": 3,
        "red": 1,
        "blue": 2,
        "white": 4,
        "purple": 4
    },
}


@dataclass
class BotConfig:
    """SEEDSMASH Bot configuration file:
Please fill properly/completely.
    """

    __field_docs__ = {}

    tag: str = "seedsmash"
    __field_docs__["tag"] = """Bot tag
    
    Tags length should be between 3 and 16 characters.
Special characters may end up be removed.
    """

    character: str = "MARIO"
    __field_docs__["character"] = """Character to be mained.

Should be among the following:
MARIO, FOX, CPTFALCON, LINK, JIGGLYPUFF, MARTH, YLINK, DOC, FALCO, GANONDORF, ROY

Yet unsupported characters:
DK, KIRBY, BOWSER, NESS, PEACH, POPO, PIKACHU, SAMUS, YOSHI, MEWTWO, LUIGI, ZELDA, PICHU, GAMEANDWATCH
    """

    costume: Union[int, str] = "default"
    __field_docs__["costume"] = """Costume (color) to be selected.

Should be either:
- The ingame index of the costume, e.g., 0 for default (see https://www.ssbwiki.com/Alternate_costume_(SSBM)),
- The name of the color, e.g., blue, or "default".
       """

    """
Follows define the preferences of your bot.
Each component affects how your bot learns, shaping its optimal strategy.

Note: An extreme configuration will certainly lead to poor results. Experiment at your own risk.
    """

    # TODO: reward shaping !

    lookahead_half_life: float = 6
    __field_docs__["lookahead_half_life"] = """Lookahead half-life.

Your bot cares decreasingly less about rewards as we go further into the future.
This value is for the number of seconds before your bot cares about half of the rewards it is getting.
Essentially defines how much your bot cares about future rewards.

Higher the value, harder and longer it is to learn, obviously.

For example:
- 1 will lead to a bot only caring about instant rewards, such as smashing
- 5-10 is a generally a good range.
    """

