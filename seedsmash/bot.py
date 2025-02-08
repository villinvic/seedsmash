import json
from typing import NamedTuple, Dict, Any

import tree

from melee import Character, Stage, Action

from seedsmash.utils import ActionStateCounts, ActionStateHitCounts


class BotStats(NamedTuple):
    aggressivity: float
    techskill: float
    offstage: float
    survival: float
    neutral: float
    # increases the length of the move history
    adaptability: float
    stagecontrol: float


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
        self.costume = costume_id
        self.preferred_stage = preferred_stage
        self.stats = stats
        self.coach_tag = coach_tag
        self.coaching_progression = coaching_progression
        self.num_coaching_steps = num_coaching_steps
        self.elo = elo
        self.num_samples_generated = 0

        self.is_out = False

        self.action_state_counts = ActionStateCounts(preferred_move)
        self.action_state_hit_counts = ActionStateHitCounts(preferred_move, self.character)

        self.metrics = {}

        # will be instantiated later
        self.mean_samples_at_creation = None



    @classmethod
    def from_json(cls, js):
        data = json.loads(js)
        data["stats"] = BotStats(**data["stats"])
        data["character"] = Character(data["character"])
        data["preferred_stage"] = Stage(data["preferred_stage"])
        data["preferred_move"] = Action(data["preferred_move"])
        return cls(
            **data
        )




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

    def push_metrics(self, metrics):
        for n, m in metrics.items():
            if n == "action_state_counts":
                self.action_state_counts.push_samples(m)
                continue
            if n == "action_state_hit_counts":
                self.action_state_hit_counts.push_samples(m)
                continue
            if n not in self.metrics:
                self.metrics[n] = m
                continue

            # perform an EMA update over metrics
            # averaging over the last 20-ish games
            smoothing = 0.1
            self.metrics[n] = tree.map_structure(
                lambda x, y: x * (1-smoothing) + y * smoothing,
                self.metrics[n], m
            )


    def get_state(self) -> Dict[str, Any]:
        # TODO
        return {
            "coach_tag": self.coach_tag,
            "coaching_progression": self.coaching_progression/self.num_coaching_steps,
            "elo": self.elo,
            "is_out": self.is_out,
            "action_state_probs": self.action_state_counts.get_top_k_probs(5),
            "move_preferences": self.action_state_hit_counts.get_top_k_probs(5),

            **self.metrics

        }


