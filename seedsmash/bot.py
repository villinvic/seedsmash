import json
from itertools import islice
from typing import NamedTuple, Dict, Any, Tuple

import numpy as np
import tree

from melee import Character, Stage, Action



class BotConfig(NamedTuple):
    tag: str = "DEFAULT"
    character: Character = Character.MARIO
    costume_id: int = 0
    elo: float = 1000


class Bot:

    def __init__(
            self,
            tag: str,
            character: Character,
            costume_id: int,
            elo: float,
            **kwargs
    ):
        self.tag = tag
        self.character = character
        self.costume_id = costume_id
        self.elo = elo
        self.num_samples_generated = 0
        self.is_out = False

        self.metrics = {
            "rl": {},
            "progression": {}
        }

        # will be instantiated later
        self.mean_samples_at_creation = None



    @classmethod
    def from_json(cls, data):
        data["character"] = Character[data["character"]]
        return cls(
            **data
        )

    def __repr__(self):
        d = {
            "tag": self.tag,
            "character": self.character,
            "costume_id": self.costume_id,

        }
        return f"Bot({d})"


    @property
    def offset_samples_generated(self):
        return self.num_samples_generated + self.mean_samples_at_creation


    def push_metrics(
            self,
            metrics,
            registry: str
    ):
        for n, m in metrics.items():
            if n not in self.metrics[registry]:
                self.metrics[registry][n] = m
                continue

            # perform an EMA update over metrics
            # averaging over the last 40-ish games
            smoothing = 0.05
            try:
                self.metrics[registry][n] = tree.map_structure(
                    lambda x, y: x * (1-smoothing) + y * smoothing,
                    self.metrics[registry][n], m
                )
            except Exception:
                self.metrics[registry][n] = m

    def get_sorted_dicts(
            self,
            name: str,
            metric: dict
    )-> Tuple[str, dict]:
        k = 8
        if name == "__move_accuracies__":
            return "Least Accurate Moves (By Accuracy%)", dict(sorted(metric.items(), key=lambda item: item[1])[:k])
        elif name == "__move_uses__":
            return "Most Used Moves (By Usage Count)", dict(sorted(metric.items(), key=lambda item: -item[1])[:k])
        elif name == "__move_hits__":
            return "Most Hit Moves (By Hit Count)", dict(sorted(metric.items(), key=lambda item: -item[1])[:k])
        else:
            return name, metric

    def get_state(self, rank: int) -> Dict[str, Any]:
        # filter large metric dicts, suppose they are already sorted
        d = dict(
            tag=self.tag,
            is_out=self.is_out,
        )

        k = 8
        for registry, metrics in self.metrics.items():
            d[registry] = {}
            for name, metric in metrics.items():
                if isinstance(metric, dict):
                    name, metric = self.get_sorted_dicts(name, metric)
                    d[registry][name] = dict(islice(metric.items(), k))
                    continue
                d[registry][name] = metric

        d["progression"].update(
            rank=rank,
            elo= self.elo,
        )

        return tree.map_structure(
            lambda v: float(v) if isinstance(v, np.floating) else v,
            d
        )


