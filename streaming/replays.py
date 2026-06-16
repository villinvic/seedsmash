import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List


@dataclass
class ReplayCommunication:
    commandId: str = "" # random string, doesn't really matter
    queue: List[str] = field(default_factory=list)
    replay: str = ""  # path to the replay if in normal or mirror mode
    startFrame: int = -123  # when to start watching the replay
    endFrame: int = 2147483647  # when to stop watching the replay
    outputOverlayFiles: bool = False  # outputs gameStartAt and gameStation to text files (only works in queue mode)
    isRealTimeMode: bool = False  # default false; keeps dolphin fairly close to real time (about 2-3 frames); only relevant in mirror mode
    shouldResync: bool = True  # default true; disables the resync functionality
    rollbackDisplayMethod: str = "off"  # "off" | "normal" | "visible"; // default off; normal shows like a player experienced it, visible shows ALL frames (normal and rollback)
    gameStation: str = "SeedSmash"
    mode: str = "normal"  # normal / queue / mirror

    def write(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f)