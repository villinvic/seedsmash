from typing import Dict

import numpy as np


class ParameterSchedule:
    def __init__(
            self,
            **schedules,
    ):
        self.schedules = schedules
        self.schedule_steps = {
           k: np.array([int(step) for step in v.keys()])
            for k , v in schedules.items()
        }

    def get(self, version: int, parameter: str):
        steps = self.schedule_steps[parameter]
        delta = version - steps
        delta[delta < 0] = 1e8
        idx = np.argmin(delta)

        return self.schedules[parameter][steps[idx]]