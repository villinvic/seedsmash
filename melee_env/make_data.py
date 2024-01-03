from pprint import pprint

import pandas as pd
import numpy as np
from melee.enums import Character, Action
import csv
import os

# df = pd.DataFrame(
#     columns=["char", "animation", "frame"]
# )
# df["char"] = df["char"].astype(int)
# df["animation"] = df["animation"].astype(int)
# df["frame"] = df["char"].astype(float)
#
# i = 0
# for c in Character:
#     if c not in unwanted_chars:
#         for a in Action:
#             df.loc[i] = [c.value, a.value, 0.]
#             i += 1
#
# df.to_csv("frame_data.csv", index=False)


class FrameData(dict):
    path = os.path.dirname(os.path.abspath(__file__)) + "/frame_data.csv"
    unwanted_chars = [Character.UNKNOWN_CHARACTER,
                      Character.SANDBAG,
                      Character.WIREFRAME_FEMALE,
                      Character.WIREFRAME_MALE,
                      Character.NANA,
                      Character.GIGA_BOWSER,
                      ]

    def __init__(self):
        super().__init__()

        for c in Character :
            if c not in FrameData.unwanted_chars:
                self[c] = {
                    a: 0 for a in Action
                }

        # load existing data
        with open(FrameData.path, mode="r") as f:
            reader = csv.reader(f)
            next(reader)
            for d in reader:
                # care about last empty line
                try:
                    c, a, frame = d
                    self[Character(int(float(c)))][Action(int(float(a)))] = np.int32(float(frame))
                except:
                    pass

    def remaining_frame(self, char, action, frame):
        if self[char][action] < frame:
            self[char][action] = np.int32(frame)

        return self[char][action] - np.int32(frame)

    def override(self):
        with open(FrameData.path, "w") as f:
            writer = csv.writer(f)

            writer.writerows([
                (c.value, a.value, frame) for c in self.keys() for (a, frame) in self[c].items()
            ])


if __name__ == '__main__':
    fd = FrameData()
    fd.remaining_frame(Character.MARIO, Action.DOWN_B_GROUND, 1)
    fd.override()
