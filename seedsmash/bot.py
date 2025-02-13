import json
from itertools import islice
from typing import NamedTuple, Dict, Any

import tree

from melee import Character, Stage, Action

from seedsmash.utils import ActionStateCounts, ActionStateHitCounts


class BotStats(NamedTuple):
    aggressivity: float = 50
    techskill: float = 50
    offstage: float = 50
    survival: float = 50
    neutral: float = 50
    # increases the length of the move history
    adaptability: float = 50
    stagecontrol: float = 50


class BotConfig(NamedTuple):
    tag: str = "DEFAULT"
    character: Character = Character.MARIO
    costume_id: int = 0
    preferred_stage: Stage = Stage.FINAL_DESTINATION
    preferred_move: Action = Action.NAIR
    stats: BotStats = BotStats()
    elo: float = 1000
    coach_tag: str = None
    coaching_progression: int = None
    num_coaching_steps: int = 160


class Bot:

    def __init__(
            self,
            tag: str,
            character: Character,
            costume_id: int,
            preferred_stage: Stage,
            preferred_move: Action,
            stats: BotStats,
            elo: float,
            coach_tag: str = None,
            coaching_progression: int = None,
            num_coaching_steps: int = 160,

            **kwargs
    ):
        self.tag = tag
        self.character = character
        self.costume_id = costume_id
        self.preferred_stage = preferred_stage
        self.stats = stats
        self.coach_tag = coach_tag
        self.coaching_progression = coaching_progression
        self.num_coaching_steps = num_coaching_steps
        self.elo = elo
        self.num_samples_generated = 0
        self.preferred_move = preferred_move

        self.is_out = False

        self.action_state_counts = ActionStateCounts(preferred_move)
        self.action_state_hit_counts = ActionStateHitCounts(preferred_move, self.character)

        self.metrics = {
            "rl": {},
            "progression": {}
        }

        # will be instantiated later
        self.mean_samples_at_creation = None



    @classmethod
    def from_json(cls, data):
        data["stats"] = BotStats(**{s.lower(): v for s, v in data["stats"].items()})
        data["character"] = Character[data["character"]]
        data["preferred_stage"] = Stage[data["preferred_stage"]]
        data["preferred_move"] = None if data["preferred_move_id"] is None else Action(data["preferred_move_id"])
        return cls(
            **data
        )

    def __repr__(self):
        d = {
            "tag": self.tag,
            "character": self.character,
            "costume_id": self.costume_id,
            "preferred_stage": self.preferred_stage,
            "preferred_move": self.preferred_move,
            "stats": self.stats

        }
        return f"Bot({d})"


    @property
    def offset_samples_generated(self):
        return self.num_samples_generated - self.mean_samples_at_creation


    def is_coached(self):
        return not (self.coach_tag is None)


    def update_coaching_progression(self):
        """
        Returns True if we are done coaching.
        """

        if self.is_coached():
            return False

        self.coaching_progression += 1

        if self.coaching_progression == self.num_coaching_steps:
            self.coach_tag = None
            return True

        return False

    def push_metrics(
            self,
            metrics,
            registry: str
    ):
        for n, m in metrics.items():
            if n == "action_state_counts":
                self.action_state_counts.push_samples(m)
                continue
            if n == "action_state_hit_counts":
                self.action_state_hit_counts.push_samples(m)
                continue
            if n not in self.metrics[registry]:
                self.metrics[registry][n] = m
                continue

            # perform an EMA update over metrics
            # averaging over the last 20-ish games
            smoothing = 0.1
            self.metrics[registry][n] = tree.map_structure(
                lambda x, y: x * (1-smoothing) + y * smoothing,
                self.metrics[registry][n], m
            )


    def get_state(self, rank: int) -> Dict[str, Any]:
        # filter large metric dicts, suppose they are already sorted
        d = dict(
            tag=self.tag,
            is_out=self.is_out,
            coach_tag=self.coach_tag,
            coaching_progression=100 * self.coaching_progression / self.num_coaching_steps,
        )
        k = 8
        for registry, metrics in self.metrics.items():
            d[registry] = {}
            for name, metric in metrics.items():
                if isinstance(metric, dict):
                    d[registry][name] = dict(islice(metric.items(), k))
                    continue
                d[registry][name] = metric

        d["progression"].update(
            rank=rank,
            elo= self.elo,
        )

        return d


