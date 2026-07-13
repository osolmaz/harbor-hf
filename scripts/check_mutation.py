#!/usr/bin/env python3
"""Run mutmut and enforce a minimum mutation kill rate."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

STATS = re.compile(
    r"(?P<done>\d+)/(?P<total>\d+)\s+.*? (?P<killed>\d+)\s+.*? "
    r"(?P<uncovered>\d+)\s+.*? (?P<timeout>\d+)\s+.*? "
    r"(?P<suspicious>\d+)\s+.*? (?P<survived>\d+)\s+.*? "
    r"(?P<skipped>\d+)"
)


def last_stats(output: str) -> dict[str, int] | None:
    matches = list(STATS.finditer(output.replace("\r", "\n")))
    if not matches:
        return None
    return {key: int(value) for key, value in matches[-1].groupdict().items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-kill-rate", type=float, required=True)
    arguments = parser.parse_args()

    completed = subprocess.run(
        ["mutmut", "run"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        sys.stderr.write(output)
        return completed.returncode

    stats = last_stats(output)
    if stats is None or stats["done"] != stats["total"]:
        sys.stderr.write(output)
        sys.stderr.write("mutation gate failed: incomplete or unreadable statistics\n")
        return 2

    caught = stats["killed"] + stats["timeout"]
    catchable = caught + stats["survived"] + stats["uncovered"]
    if catchable == 0:
        sys.stderr.write("mutation gate failed: no mutants were generated\n")
        return 2

    rate = 100 * caught / catchable
    print(f"mutation kill rate: {rate:.1f}% ({caught}/{catchable})")
    if rate < arguments.min_kill_rate:
        sys.stderr.write(
            f"mutation gate failed: {rate:.1f}% is below "
            f"{arguments.min_kill_rate:.1f}%\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
