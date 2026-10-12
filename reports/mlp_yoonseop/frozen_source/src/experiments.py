"""Run each agreed experiment in a fresh process; do not select using test data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

IDS = ["E02", "E03", "E07", "E08", "E09", "E10", "E11"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/formal"))
    parser.add_argument("--experiments", nargs="+", choices=IDS, default=IDS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for experiment in args.experiments:
        command = [sys.executable, "-m", "src.train", "--config", f"configs/{experiment}.json",
                   "--data-root", str(args.data_root), "--runs-dir", str(args.runs_dir)]
        if args.smoke:
            command.append("--smoke")
        print(f"Starting {experiment}", flush=True)
        subprocess.run(command, check=True)
    print("All requested experiments completed; official test data was not evaluated.")


if __name__ == "__main__":
    main()
