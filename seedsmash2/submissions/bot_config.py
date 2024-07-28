from typing import NamedTuple, Union

import numpy as np
from melee.enums import Character, Stage
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

for char, costume_dict in color2index.items():
    costume_dict["default"] = 0

stage_to_enum = {
    stage.name: stage for stage in Stage
}

char_to_enum = {
    char.name: char for char in Character
}


@dataclass
class BotConfig:
    """SEEDSMASH Bot configuration file:
Please fill properly/completely.

"Play-style" parameters shape the strategy you will get. Note that their effect is not 100% clear,
as the combined effect of our chosen parameters is not obvious.
To illustrate with an example, a bot with a default configuration is rewarded with:
> 1 point for killing, -1 point for dying,
> 0.005 point per percent dealt, -0.005 point per percent suffered,
> -0.0002 point every 3 frames
NOTE: those rewards and other play-style related rewards affect the rewards collected by your bot for training only.
This is different from the actual performance metrics of your bot: rank, winrate, rating... which only depend on
whether your bot wins or loses.

* Will later be added:
- details about the algorithm used for training,
- the model used for the policy.
    """

    def __post_init__(self):
        self._id = np.random.randint(2**32)
        default_config = BotConfig

        # validate inputs
        # raise an error if one input is wrong, or switch to default.
        def validation_message(value, field, default):
            print(f"({self.tag} BotConfig) The value {value} for parameter {field} is invalid, using default {default} instead.")

        try:
            self.character = char_to_enum[self.character]
        except:
            validation_message(self.character, "character", default_config.character)
            self.character = default_config.character

        try:
            costume_id = int(self.costume)
            self.costume = costume_id
        except Exception:
            try:
                costume_color = self.costume.lower()
                self.costume = color2index[self.character][costume_color]
            except Exception:
                validation_message(self.costume, "costume", default_config.costume)
                self.costume = default_config.costume

        try:
            self.preferred_stage = stage_to_enum[self.preferred_stage]
        except Exception:
            validation_message(self.character, "preferred_stage", default_config.preferred_stage)
            self.preferred_stage = default_config.preferred_stage

        try:
            self.random_action_chance = np.float32(self.random_action_chance)
        except Exception:
            validation_message(self.random_action_chance, "random_action_chance", default_config.random_action_chance)
            self.random_action_chance = default_config.random_action_chance

        for field in self.__characterisation_fields__:
            try:
                v =  np.float32(getattr(self, field))
                if not (0<=v<=100):
                    raise ValueError
                else:
                    setattr(self, field, v)
            except Exception:
                default_value = getattr(default_config, field)
                validation_message(v, field,
                                   default_value)
                setattr(self, field, default_value)


    __field_docs__ = {}
    __characterisation_fields__ = sorted([
        "reflexion",
        "agressivity",
        "winning_desire",
        "patience",
        "creativity",
        "off_stage_plays",
        "combo_game",
        "combo_breaker"
    ])

    tag: str = "your_bot_name"
    __field_docs__["tag"] = """Bot tag
    
    Tags length should be between 3 and 16 characters.
Special characters may end up be removed.
For now, if your tag is taken, a number will be added at the end of the tag. 
    """

    character: Union[str, Character] = "MARIO"
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

    preferred_stage: Union[str, Stage] = "BATTLEFIELD"
    __field_docs__["character"] = """Stage preferred by your bot.

Your bot picks this stage with a higher chance.

Should be among the following:
FINAL_DESTINATION, BATTLEFIELD, POKEMON_STADIUM, DREAMLAND
Yet unsupported stages (buggy):
FOUNTAIN_OF_DREAMS, YOSHIS_STORY
        """

    # We should control entropy on our side, in order to ensure no bot collapses mid training

    random_action_chance: float = 1
    __field_docs__["random_action_chance"] = """Random action chance.
Valid range: [0, 10]


Percentage of actions to be picked completely randomly.
As your bot progress throughout learning, its certainty about its actions will grow, sometimes stopping exploration 
entirely in specific situations. 
A value higher than 0 should ensure that your bot keeps considering every action and constant exploration.
Nevertheless, this gives your bot a chance to make some very bad decisions.
This offers a constant tradeoff between performance and strategy exploration.

Ex: since bots take 20 actions per second, with 1%, your bot will pick 1 random action every 5 seconds. 
By default, this is set to 1.
        """

    reflexion: float = 50
    __field_docs__["reflexion"] = """(Play-style) Reflexion.
Valid range: [0, 100]

Your bot cares decreasingly less about rewards as we go further into the future.
Essentially defines how much your bot cares about future rewards.
With a value of 0, your bot cares about half the rewards it will get two seconds into the future. This will lead
to a bot only caring about instant rewards, such as smashing.
With a value of 100, it will be 20 seconds instead.

Note: Higher the value, harder and longer it is to learn.

For example:
- 40-60 is a generally a good range.
By default, this is set to 50.
        """

    agressivity: float = 50
    __field_docs__["agressivity"] = """(Play-style) Agressivity.
Valid range: [0, 100]

How much your bot prioritises damaging/killing over getting damaged/dying.

Ex:
A value of 50 leads to a bot equally caring about defense and attack.
A value of 100 leads to a bot that does not care about dying.
A value of 0 leads to a bot that does not care about killing/damaging its opponent.
40-60 is a good range for agressivity
By default, this is set to 50.
            """

    winning_desire: float = 0
    __field_docs__["winning_desire"] = """(Play-style) Winning desire.

Strength of the reward signal received by your bot when winning, on top of the reward received when your opponent 
loses its last stock.

A reasonable range for the signal is 0-10
By default, this is set to 0.
                """

    patience: float = 50
    __field_docs__["patience"] = """(Play-style) Patience.
Valid range: [0, 100]

By default, this is set to 50.
Every 3 frames, your bot receives a small penalty, that diminishes with patience.

With a low patience, your bot will try to finish the game more quickly.
NOTE: a timeout is always considered a draw, and not a win for the play which has more stocks and less percent.

By default, this is set to 50.
                    """

    creativity: float = 0
    __field_docs__["creativity"] = """(Play-style) Creativity.
Valid range: [0, 100]

Gives a small bonus to your bot when in action animations it rarely encounters, e.g., wall-jumping, the different Marth dancing blades.
This essentially gives some exploratory boost.

By default this is set to 0.
                        """

    off_stage_plays: float = 0
    __field_docs__["off_stage_plays"] = """(Play-style) Off-stage plays.
Valid range: [0, 100]

How much your bot likes being offstage
Boosts the positive rewards received when offstage,

By default this is set to 0.
                        """

    combo_game: float = 0
    __field_docs__["combo_game"] = """(Play-style) Combo game.
Valid range: [0, 100]

How much your bot plays around combos.
Boosts exponentially the reward received for each consecutive hit of a combo.
This incentives your bot to go for combos, rather than hit and run. 

Note: depending on the mained character, this may work as intended, or not...

By default this is set to 0.
                            """

    combo_breaker: float = 0
    __field_docs__["combo_breaker"] = """(Play-style) Combo breaker.
Valid range: [0, 100]

How much your bot avoids getting combo-ed.
Boosts exponentially the penalties received for each consecutive hit of the opponent's combo.
This incentives your bot to avoid combos/break out of combos. 
By default this is set to 0.
                            """


