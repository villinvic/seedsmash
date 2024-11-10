from typing import List

from ml_collections import ConfigDict
from melee import Stage, Character, ControllerType
from copy import deepcopy
from melee_env.enums import PlayerType


class FunctionalConfig(dict):
    """
    RLLib style config
    """

    def __init__(self, **default_config):
        super().__init__()

        self.default_config = default_config
        self.update(default_config)

    def update_config(self, item, value) -> "FunctionalConfig":
        self[item] = value
        return self

    def enable(self, item) -> "FunctionalConfig":
        self[item] = True
        return self

    def disable(self, item) -> "FunctionalConfig":
        self[item] = False
        return self


class SSBMObsConfig(FunctionalConfig):
    """
    Defines what should the observation contain, and how.
    """

    def __init__(self):
        super().__init__(
            delay=0,
            stage=False,
            character=False,
            controller_state=False,
            projectiles=False,
            ecb=False,
        )

    def stage(self) -> "SSBMObsConfig":
        return self.enable("stage")

    def ecb(self) -> "SSBMObsConfig":
        return self.enable("ecb")

    def character(self) -> "SSBMObsConfig":
        return self.enable("character")

    def controller_state(self) -> "SSBMObsConfig":
        return self.enable("controller_state")

    def projectiles(self) -> "SSBMObsConfig":
        return self.enable("projectiles")

    def delay(self, delay) -> "SSBMObsConfig":
        if delay < 0:
            raise ValueError("Cannot have negative delay.")
        return self.update_config("delay", delay)


class SSBMConfig(FunctionalConfig):
    """
    Defines the config for the Polaris env and libmelee.
    """

    def __init__(
            self,
            faster_melee_path: str,
            exiai_path: str,
            iso_path: str,
    ):

        paths = dict(
            FM=faster_melee_path,
            ExiAI=exiai_path,
            iso=iso_path
        )
        super().__init__(
            paths=paths,
            obs_config=SSBMObsConfig(),
            playable_stages=[Stage.FINAL_DESTINATION],
            playable_characters=[Character.FOX],
            player_types=[PlayerType.BOT, PlayerType.HUMAN],
            save_replays=False,
            render=False,
            use_ffw=False,
            debug=False,
            online_delay=0,
        )

        self.update(self.default_config)

    def playable_stages(self, stages: List[Stage]) -> "SSBMConfig":
        if not (isinstance(stages, list) and isinstance(stages[0], Stage)):
            raise ValueError(f"The chosen stages are not a list of valid stages: {stages}.")
        return self.update_config("playable_stages", stages)

    def playable_characters(self, characters: List[Character]) -> "SSBMConfig":
        if not (isinstance(characters, list) and isinstance(characters[0], Character)):
            raise ValueError(f"The chosen characters are not a list of valid characters: {characters}.")
        return self.update_config("playable_characters", characters)

    def player_types(self, player_types: List[PlayerType]) -> "SSBMConfig":
        if not 4 > len(player_types) > 1:
            raise ValueError(f"Wrong number of players: {player_types}.")
        return self.update_config("player_types", player_types)

    def save_replays(self) -> "SSBMConfig":
        return self.enable("save_replays")

    def render(self) -> "SSBMConfig":
        return self.enable("render")

    def use_ffw(self) -> "SSBMConfig":
        return self.enable("use_ffw")

    def debug(self) -> "SSBMConfig":
        return self.enable("debug")

    def obs_config(self, obs_conf: SSBMObsConfig) -> "SSBMConfig":
        if not isinstance(obs_conf, SSBMObsConfig):
            raise  ValueError(f"Need proper SSBM observation config object, got {obs_conf}.")
        return self.update_config("obs_config", obs_conf)

    def online_delay(self, delay = 0) -> "SSBMConfig":
        if delay < 0:
            raise  ValueError(f"Cannot have negative delay.")
        return self.update_config("online_delay", delay)


if __name__ == '__main__':

    obs_config = (
        SSBMObsConfig()
        .stage()
        .character()
        .delay(0)
    )

    env_config = (
        SSBMConfig(
            "fm",
            "exiai",
            "melee.iso"
        )
        .playable_stages([Stage.FINAL_DESTINATION])
        .playable_characters([Character.FOX])
        .player_types([PlayerType.BOT, PlayerType.BOT])
        .render()
        .use_ffw()
        .online_delay(0)
        .obs_config(obs_config)
    )

    print(env_config["online_delay"])