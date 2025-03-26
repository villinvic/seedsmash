import glob
import os
import time
from collections import deque
from pathlib import Path

import peppi_py as peppi
import re
from codename import codename



class SlpReplayManager:
    # TODO: https://github.com/project-slippi/slippi-js/blob/master/README.md
    def __init__(
            self,
            replay_path: str = "/home/goji/Slippi",
    ):
        self.replay_path = Path(replay_path)

        self.replay_name: None | str = None

    def tag_replay(self):
        # clean up replays
        delete_old_files(self.replay_path, age_in_seconds=60*16)
        self.replay_name = get_latest_file(self.replay_path, extension=".slp")
        print(self.replay_name)

    def inject_bot_info(
            self,
            tag1: str,
            tag2: str,
    ):
        if self.replay_name is None:
            return

        with open(self.replay_path / self.replay_name, "rb") as file:
            data = file.read()

        data = update_metadata(data, tag1, tag2)
        data = update_start_frame(data, tag1, tag2)

        actual_replay_name = f"{tag1}_{tag2}_{codename(separator='', capitalize=True)}.sslp"
        try:
            os.remove(self.replay_path / self.replay_name)
        except Exception as e:
            print(f"Error deleting {file}: {e}")
        with open(self.replay_path / actual_replay_name, "wb") as file:
            file.write(data)

        self.replay_name = None


def update_metadata(
        data: bytes,
        tag1: str,
        tag2: str
):
    """
    Very hacky function for modifying the replay metadata.
    """
    pattern = rb"(U\x05names{})"

    to_fill = "U\x05names{{U\x07netplaySU{tag_length}{tag}U\x04codeSU\x05SS#01}}"
    # TODO: underscores are not read properly
    for tag in [tag1, tag2]:
        filled = to_fill.format(tag_length=bytes([len(tag)]).decode(), tag=tag)
        data = re.sub(pattern, filled.encode(), data, count=1)

    return data

def update_start_frame(
        data: bytes,
        tag1: str,
        tag2: str
):
    """
    Very hacky function that injects the bot tags to the replay
    """
    start = 464
    b = bytearray(data)

    def format_tag(tag: str) -> bytes:
        # Ensure the tag is 112 bytes long (pad with spaces or truncate if necessary)
        tag_bytes = tag.encode('utf-8')
        if len(tag_bytes) > 30:
            return tag_bytes[:30]  # truncate if the tag is longer than 112 bytes
        return tag_bytes.ljust(30, b'\x00')  # pad with 0x00 if the tag is shorter than 112 bytes

    b[start+0] = 8
    b[start+1: start+31] = format_tag(tag1)
    b[start+32: start+62] = format_tag(tag2)
    return bytes(b)


def get_latest_file(folder_path: Path, exclude: deque = None, extension: str = "") -> str | None:

    files = glob.glob(str((folder_path / "*").with_suffix(extension)))

    if not files:
        return None

    if exclude is not None:
        files = [f for f in files if f not in exclude]
    if len(files) == 0:
        return None

    latest_file = max(files, key=os.path.getctime)

    return latest_file


def delete_old_files(folder_path: Path,  age_in_seconds: int,  extension: str = ""):

    current_time = time.time()

    files = glob.glob(str((folder_path / "*").with_suffix(extension)))
    for file in files:
        file_age = current_time - os.path.getmtime(file)

        if file_age > age_in_seconds:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting {file}: {e}")



if __name__ == '__main__':

    slp_manager = SlpReplayManager()
    slp_manager.replay_name = "Game_20250222T123710.slp"

    slp_manager.inject_bot_info("cascou", "itworks")

    game = peppi.read_slippi("/home/goji/Slippi/test.slp")

    print(game.start)