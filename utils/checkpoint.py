"""Runtime/timing helper shared by the EstIP notebooks and scripts."""

import time


class Checkpoint:
    def __init__(self, label="Start"):
        self.start = time.time()
        print(f"{label} time: {time.ctime(self.start)}")

    def mark(self, label="Checkpoint"):
        """
        Print the current time and return it, so it can be passed to a
        later report() as the `since` reference point.
        """
        now = time.time()
        print(f"{label} time: {time.ctime(now)}")
        return now

    def report(self, label, since=None):
        """
        Print elapsed minutes/hours since `since` (default: start).
        """
        since = self.start if since is None else since
        minutes = round((time.time() - since) / 60, 2)
        hours = round(minutes / 60, 2)
        print(f"{label}: {minutes}min ({hours}hrs)")
        return minutes

    def finish(self, label="Total script runtime"):
        end = time.time()
        print(f"End time: {time.ctime(end)}")
        self.report(label)