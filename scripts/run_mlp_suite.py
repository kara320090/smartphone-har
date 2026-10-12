"""Run Han Yoonseop's three fixed MLP seeds in separate processes, then verify.

An existing complete run is reused only when its config and source hashes match.
Incomplete runs are preserved. No official test evaluation is performed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def check_reusable(run, expected, seed, smoke):
    if not run.exists():
        return False
    status_path = run / "status.json"
    if not status_path.exists() or json.loads(status_path.read_text())["status"] != "complete":
        raise RuntimeError(f"Incomplete run preserved: {run}. Use a new --runs-dir for retry.")
    saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
    effective = dict(expected, seed=seed, smoke=smoke)
    if smoke:
        effective["epochs"] = 2
    for key, value in effective.items():
        if saved.get(key) != value:
            raise ValueError(f"Existing run config differs: {key}. Use a new --runs-dir.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-data", type=Path, default=Path("data/processed_data.npz"))
    parser.add_argument("--preprocess-stats", type=Path, default=Path("data/preprocess.npz"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/mlp_yoonseop"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Each seed must appear once")
    root = Path(__file__).resolve().parents[1]
    temporary = root / "work/mlp_tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(temporary)
    os.environ["TMP"] = str(temporary)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "work/mplconfig"))
    os.environ.setdefault("KERAS_HOME", str(root / "work/keras"))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    config_path = root / "configs/M01.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Resolve before launching processes so invocation from another folder is safe.
    data_path, stats_path, runs_dir = (p.resolve() for p in
                                      (args.processed_data, args.preprocess_stats, args.runs_dir))
    runs_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        name = f"M01_seed{seed}" + ("_smoke" if args.smoke else "")
        run = runs_dir / name
        if not check_reusable(run, config, seed, args.smoke):
            command = [sys.executable, "-m", "src.train", "--config", str(config_path),
                       "--processed-data", str(data_path), "--preprocess-stats", str(stats_path),
                       "--runs-dir", str(runs_dir), "--seed", str(seed)]
            if args.smoke:
                command.append("--smoke")
            print("START", name, flush=True)
            with (runs_dir / f"{name}.log").open("w", encoding="utf-8") as stream:
                subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
        command = [sys.executable, "-m", "src.verify_bundle", "--processed-data", str(data_path),
                   "--preprocess-stats", str(stats_path), "--run-directory", str(run),
                   "--report", str(run / "fresh_process_verification.json")]
        if args.smoke:
            command.append("--smoke")
        subprocess.run(command, cwd=root, check=True)
        print("VERIFIED", name, flush=True)


if __name__ == "__main__":
    main()
