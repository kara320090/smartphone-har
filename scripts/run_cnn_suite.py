"""Run C01 at three fixed seeds in fresh TF processes, then verify saved models.

Example:
 python scripts/run_cnn_suite.py --processed-data data/processed_data.npz
 python scripts/run_cnn_suite.py --processed-data data/processed_data.npz --smoke --runs-dir runs/cnn-smoke

Does not read official test labels or use test arrays. Completed runs are not overwritten.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-data", type=Path, required=True)
    parser.add_argument("--preprocess-stats", type=Path, default=Path("data/preprocess.npz"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/cnn"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    root = Path(__file__).resolve().parents[1]
    data = args.processed_data.resolve()
    stats = args.preprocess_stats.resolve()
    out = args.runs_dir.resolve()
    if not data.is_file() or not stats.is_file():
        parser.error(f"Missing data or stats: {data} / {stats}")
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        name = f"C01_seed{seed}" + ("_smoke" if args.smoke else "")
        run = out / name
        status = run / "status.json"
        if run.exists() and not status.is_file():
            raise RuntimeError(f"Unverified run directory exists: {run}")
        if status.is_file():
            if json.loads(status.read_text(encoding="utf-8")).get("status") != "complete":
                raise RuntimeError(f"Incomplete run preserved; use a new --runs-dir: {run}")
            print("SKIP completed", name, flush=True)
        else:
            command = [sys.executable, "-m", "src.train", "--config", "configs/C01.json",
                       "--processed-data", str(data), "--preprocess-stats", str(stats),
                       "--runs-dir", str(out), "--seed", str(seed)]
            if args.smoke:
                command.append("--smoke")
            print("TRAIN", name, flush=True)
            with (out / f"{name}.log").open("w", encoding="utf-8") as logfile:
                subprocess.run(command, cwd=root, stdout=logfile, stderr=subprocess.STDOUT, check=True)
        verify = [sys.executable, "-m", "src.verify_bundle", "--processed-data", str(data),
                  "--preprocess-stats", str(stats), "--run-directory", str(run),
                  "--report", str(run / "fresh_process_verification.json")]
        if args.smoke:
            verify.append("--smoke")
        subprocess.run(verify, cwd=root, check=True)
        print("VERIFIED", name, flush=True)
    print("All requested CNN runs verified; no test set was evaluated.")


if __name__ == "__main__":
    main()
