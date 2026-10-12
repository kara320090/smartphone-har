"""Sequential fresh-process experiments; skips only verified completed runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--processed-data", type=Path, required=True)
    p.add_argument("--preprocess-stats", type=Path, default=Path("data/preprocess.npz"))
    p.add_argument("--runs-dir", type=Path, default=Path("runs/lstm"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for experiment in ("L01", "L02", "L03"):
            name = f"{experiment}_seed{seed}" + ("_smoke" if args.smoke else "")
            run = args.runs_dir / name
            status_file = run / "status.json"
            complete = status_file.exists() and json.loads(status_file.read_text())["status"] == "complete"
            if run.exists() and not complete:
                raise RuntimeError(f"Incomplete run preserved: {run}. Use a new suite directory for retry.")
            if not complete:
                command = [sys.executable, "-m", "src.train", "--config", f"configs/{experiment}.json",
                           "--processed-data", str(args.processed_data), "--preprocess-stats", str(args.preprocess_stats),
                           "--runs-dir", str(args.runs_dir), "--seed", str(seed)]
                if args.smoke:
                    command.append("--smoke")
                if experiment == "L03":
                    command += ["--initial-weights-from", str(args.runs_dir / (f"L01_seed{seed}" + ("_smoke" if args.smoke else "")))]
                print(time.strftime("%Y-%m-%d %H:%M:%S"), "START", name, flush=True)
                with (args.runs_dir / f"{name}.log").open("w", encoding="utf8") as log:
                    subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            command = [sys.executable, "-m", "src.verify_bundle", "--processed-data", str(args.processed_data),
                       "--preprocess-stats", str(args.preprocess_stats), "--run-directory", str(run),
                       "--report", str(run / "fresh_process_verification.json")]
            if args.smoke:
                command.append("--smoke")
            subprocess.run(command, cwd=root, check=True)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "VERIFIED", name, flush=True)


if __name__ == "__main__":
    main()
