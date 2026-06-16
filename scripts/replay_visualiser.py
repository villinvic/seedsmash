import argparse
import asyncio
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
from seedsmash.twitch_bot import SSTwitchBot


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

class ReplayAutoWatcher:

    def __init__(
            self,
            playback_path: str,
            iso: str,
            replay_dir: str,
            comm_path: str,
            db_address: str,
            enable_bot: bool,
    ):

        self.replay_dir = Path(replay_dir)
        self.comm_path = Path(comm_path)
        self.console = PlayBackConsole(playback_path)
        self.iso = iso
        self.replay_comm = ReplayCommunication()
        self.twitchbot = SSTwitchBot(db_address)
        self.enable_bot = enable_bot

    def loop(self):
        if self.enable_bot:
            self.twitchbot.start()
            time.sleep(7)
        self.console.run(self.comm_path, self.iso)
        try:
            while True:
                replay, msg, user_replay = self.twitchbot.mu_queue.pull_replay()
                if replay is None or not self.load_next_replay(replay):
                    time.sleep(3)
                    continue
                if self.enable_bot and user_replay:
                    asyncio.run(self.twitchbot.alert(msg))
                self.console.wait()

        except KeyboardInterrupt:
            self.console.stop()
            self.twitchbot.stop()


    def load_next_replay(self, replay):
        path = self.replay_dir / replay
        if not os.path.exists(path):
            return False
        self.replay_comm.replay = str(path)
        self.replay_comm.write(self.comm_path)
        return True


parser = argparse.ArgumentParser()
parser.add_argument('--comm-path', type=str, default="seedsmash_comm.json")
parser.add_argument('--enable-bot', type=bool, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--db-address', type=str, required=True)
parser.add_argument('--playback-path', type=str, required=True)
parser.add_argument('--replay-path', type=str, required=True)
parser.add_argument('--iso', type=str, required=True)

if __name__ == '__main__':
    ARGS = parser.parse_args()

    auto = ReplayAutoWatcher(
        ARGS.playback_path,
        ARGS.iso,
        ARGS.replay_path,
        comm_path=ARGS.comm_path,
        db_address=ARGS.db_address,
        enable_bot=ARGS.enable_bot
    )

    auto.loop()