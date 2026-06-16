import os
import select
import subprocess
from pathlib import Path
from subprocess import Popen


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
                             stderr=subprocess.PIPE,
                             env=env,
                             #text=True  # Returns output as a string instead of bytes
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
        repeat_c = 0
        prev_line = "*"
        while True:
            ready, _, _ = select.select([self._process.stdout], [], [], 4)
            if not ready or repeat_c > 10:
                print("replay finished.")
                return
            newline = self._process.stdout.readline().decode("utf-8").strip()
            if prev_line == newline:
                repeat_c += 1
            else:
                repeat_c = 0
            prev_line = newline