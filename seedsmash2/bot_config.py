from typing import NamedTuple

class BotConfig(NamedTuple):
    tag: str
    character: str # could make this a distribution instead eventually
    color_index: int

    # preferences