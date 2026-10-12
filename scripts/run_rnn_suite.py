"""RNN baseline/tuning only on current main; archived comparisons use frozen_source."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--processed-data", type=Path, default=Path("data/processed_data.npz"))
    p.add_argument("--preprocess-stats", type=Path, default=Path("data/preprocess.npz"))
    p.add_argument("--runs-dir", type=Path, default=Path("runs/rnn"))
    p.add_argument("--experiments", nargs="+", default=["R01", "R02", "R03"],
                   choices=["R01", "R02", "R03"])
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for exp in args.experiments:
            name = f"{exp}_seed{seed}" + ("_smoke" if args.smoke else "")
            run = args.runs_dir / name
            if run.exists():
                status = run / "status.json"
                if not status.exists() or json.loads(status.read_text())["status"] != "complete":
                    raise RuntimeError(f"Incomplete run preserved: {run}; choose a new suite directory")
            else:
                command = [sys.executable, "-m", "src.rnn_train", "--config", f"configs/{exp}.json",
                    "--processed-data", str(args.processed_data), "--preprocess-stats", str(args.preprocess_stats),
                    "--runs-dir", str(args.runs_dir), "--seed", str(seed)]
                if args.smoke:
                    command += ["--smoke"]
                if exp == "R03":
                    command += ["--initial-weights-from", str(args.runs_dir / (f"R02_seed{seed}" + ("_smoke" if args.smoke else "")))]
                print("START", name, flush=True)
                with (args.runs_dir / f"{name}.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            command = [sys.executable, "-m", "src.verify_bundle", "--processed-data", str(args.processed_data),
                       "--preprocess-stats", str(args.preprocess_stats), "--run-directory", str(run),
                       "--report", str(run / "fresh_process_verification.json")]
            if args.smoke:
                command += ["--smoke"]
            subprocess.run(command, cwd=root, check=True)
            print("VERIFIED", name, flush=True)


if __name__ == "__main__":
    main()
