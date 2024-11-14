from pprint import pprint

import pandas as pd
import numpy as np
from melee.enums import Character, Action, Stage
import csv
import os
from filelock import FileLock

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
    path = os.path.dirname(os.path.abspath(__file__))
    file_name = "frame_data.csv"
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
        with open(FrameData.path + "/" + FrameData.file_name, mode="r") as f:
            reader = csv.reader(f)
            for d in reader:
                # care about last empty line
                try:
                    c, a, frame = d
                    self[Character(int(float(c)))][Action(int(float(a)))] = np.int32(float(frame))
                except:
                    pass

    def remaining_frame(self, char, action, frame):
        frame = np.clip(frame, 0, 2**16)
        if self[char][action] < frame:
            self[char][action] = np.int32(frame)

        return self[char][action] - np.int32(frame)

    def save(self, name="new_frame_data.csv"):
        with open(FrameData.path + "/" + name, "w") as f:
            writer = csv.writer(f)

            writer.writerows([
                (c.value, a.value, frame) for c in self.keys() for (a, frame) in self[c].items()
            ])

class BadCombinations:
    path = os.path.dirname(os.path.abspath(__file__))
    file_name = "bad_combinations.csv"

    def __init__(self):
        self.lock = FileLock(f"{BadCombinations.file_name}.lock")
        self.bad_combinations = []
        self.bad_combinations_with_errnum = []

    def __contains__(self, item):
        return item in self.bad_combinations

    def load_combinations(self):
        self.bad_combinations = []
        self.bad_combinations_with_errnum = []

        with self.lock:

            with open(BadCombinations.path + "/" + BadCombinations.file_name, mode="r") as f:
                reader = csv.reader(f)
                for d in reader:
                    # care about last empty line
                    try:
                        char1, char2, stage, errnum = d
                        self.bad_combinations.append(
                            (Character[char1], Character[char2], Stage[stage])
                        )
                        self.bad_combinations_with_errnum.append(
                            (Character[char1], Character[char2], Stage[stage], int(errnum))
                        )
                    except:
                        pass

    def dump_on_error(self, char1: Character, char2: Character, stage: Stage, errnum: int):

        self.load_combinations()
        if (char1, char2, stage, errnum) not in self.bad_combinations_with_errnum:
            with open(BadCombinations.path + "/" + BadCombinations.file_name, mode='a') as f:
                writer = csv.writer(f)
                new_comb = [char1.name, char2.name, stage.name, errnum]
                writer.writerow(new_comb)
                print(f"Added bad combination: {new_comb}")


if __name__ == '__main__':
    fd = FrameData()
    fd.remaining_frame(Character.MARIO, Action.DOWN_B_GROUND, 1)
    fd.save()
