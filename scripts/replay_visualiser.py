import argparse
import os
import select
import subprocess
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path
from subprocess import Popen
from typing import NamedTuple, List
import json
import melee
from polaris_melee.replays import get_latest_file


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


class PlayBackConsole:

    def __init__(
            self,
            exe_path: str,
    ):
        self.exe_path = exe_path

        self.proc: None | Popen = None

    def run(
            self,
            replay_comm_file: Path,
            iso: str,
    ):
        command = [self.exe_path, "--cout", "-i", replay_comm_file, "-e", iso, "--slippi-spectator-port", "7777"]
        env = os.environ.copy()

        self._process = Popen(command,
                             stdout=subprocess.PIPE,
                             #stderr=subprocess.DEVNULL,
                             env=env,
                             text = True  # Returns output as a string instead of bytes
        )

    def stop(self):
        """ Stop the console.

        For Dolphin instances, this will kill the dolphin process.
        For Wiis and SLP files, it just shuts down our connection
         """
        # If dolphin, kill the process
        if self._process is not None:
            # Sadly dolphin doesn't respect terminate
            self._process.kill()
            self._process.wait()

    def wait(self):
        end_frame = 1e8

        while True:
            ready, _, _ = select.select([self._process.stdout], [], [], 10)
            if not ready:
                return
            line = self._process.stdout.readline()
            cout = line.split()
            if len(cout) != 2:
                continue
            msg_type, val = line.split()
            if msg_type == "[GAME_END_FRAME]":
                end_frame = int(val)
                continue
            if msg_type == "[CURRENT_FRAME]":
                frame = int(val)
                if end_frame == frame:
                    return

def load_next_replay(
        comm_path: Path,
        replay_dir: Path,
        replay_comm: ReplayCommunication,
        watched_replays: deque
):
    # load requested matchup if any
    # TODO
    next_replay = None
    while next_replay is None:
        try:
            next_replay = get_latest_file(replay_dir, watched_replays, extension=".sslp")
        except FileNotFoundError:
            next_replay = None

    replay_comm.replay= str(replay_dir / next_replay)
    watched_replays.append(next_replay)
    replay_comm.write(comm_path)


def auto_watch_replays(
        playback_path: str,
        iso: str,
        replay_dir: str,
        comm_path: str,
):
    watched_replays = deque(maxlen=100)
    comm_path = Path(comm_path)
    replay_dir = Path(replay_dir)
    console = PlayBackConsole(playback_path)
    replay_comm = ReplayCommunication()
    load_next_replay(comm_path, replay_dir, replay_comm, watched_replays)
    console.run(comm_path, iso)
    try:
        while True:
            console.wait()
            load_next_replay(comm_path, replay_dir, replay_comm, watched_replays)

    except KeyboardInterrupt:
        console.stop()




parser = argparse.ArgumentParser()
parser.add_argument('--comm-path', type=str, default="seedsmash_comm.json")
parser.add_argument('--playback-path', type=str, required=True)
parser.add_argument('--replay-path', type=str, required=True)
parser.add_argument('--iso', type=str, required=True)

if __name__ == '__main__':
    ARGS = parser.parse_args()
    auto_watch_replays(ARGS.playback_path, ARGS.iso, ARGS.replay_path, comm_path=ARGS.comm_path)