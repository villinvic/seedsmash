from io import BytesIO
from typing import List, NamedTuple, Dict, Any, TypedDict

import requests

from melee import Stage
from seedsmash.bot import Bot

methods = {
    "get": requests.get,
    "post": requests.post
}



class Game(TypedDict):
    bot_a: str
    bot_b: str
    winner: int | None
    stage: str
    duration: float
    replay: BytesIO | None = None


def jsonify_game(
        bot_a: Bot,
        bot_b: Bot,
        winner: str | None,
        stage: Stage,
        length: int,
        replay: str | None = None
) -> Game:
    tag_a = bot_a.tag
    tag_b = bot_b.tag
    stage = stage.name

    if replay is not None:
        # TODO: read replay from path
        # remove the replay as well
        pass

    # 3 frames per step
    duration = length / 20

    return {
        "bot_a": tag_a,
        "bot_b": tag_b,
        "winner": winner,
        "stage": stage,
        "duration": duration,
        "replay": replay
    }


class SeedSmashDataBag(NamedTuple):
    games: List[Game]
    bot_states: List[Dict[str, Any]] | None = None


class ApiInterface:
    def __init__(
            self,
            address: str
    ):
        self.address = "http://" + address
        self.api_key = input("Enter SeedSmash private API key:")


    def request(
            self,
            endpoint: str,
            data = None,
            headers = None,
            method = "get"

    ):

        method = methods[method]
        try:
            if headers is None:
                headers = {}
            headers["api-key"] = self.api_key
            response = method(self.address + endpoint, json=data, headers=headers)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

    def read_db_bots(self):
        return [Bot.from_json(js) for js in self.request("/api/bots")]

    def push_data(
            self,
            data: SeedSmashDataBag
    ):
        """
        Updates must contain:
        - new games
        - new metrics (passed every 30 mins or something)
        - out bots ? (we can handle this db side ?)
        ...


        - slippi replays
        """

        return self.request(
            "/api/update",
            data=data._asdict()
        )

    def communicate(self, data):
        self.push_data(data)
        return self.read_db_bots()





if __name__ == '__main__':

    interface = ApiInterface(
        address="192.168.1.100:5000"
    )

    print(interface.request("/api/bots"))