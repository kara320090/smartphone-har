"""Save the exact local code and protocol before the formal three-seed run."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "reports/mlp_yoonseop"
    snapshot = out / "frozen_source"
    snapshot.mkdir(parents=True, exist_ok=False)
    source_digest = hashlib.sha256()
    files = []
    for folder, suffix in (("src", ".py"), ("configs", ".json"), ("scripts", ".py")):
        for path in sorted((root / folder).rglob(f"*{suffix}")):
            relative = path.relative_to(root)
            content = path.read_bytes()
            source_digest.update(relative.as_posix().encode())
            source_digest.update(content)
            files.append(path)
    files += sorted((root / "tests").rglob("*.py"))
    files += [root / "requirements.txt", root / ".gitignore",
              root / "docs/team-paper/contributions/02_mlp/training_protocol.md"]
    entries = []
    for path in files:
        relative = path.relative_to(root)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        entries.append({"path": relative.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
    record = {"date_korea": "2026-10-12", "stage": "before_formal_training", "contributor": "한윤섭",
              "base_commit": rev.stdout.strip(), "local_uncommitted_changes_in_snapshot": True,
              "code_sha256": source_digest.hexdigest(), "config_id": "M01", "seeds": [2026, 2027, 2028],
              "test_evaluated": False, "files": entries}
    (out / "protocol_freeze.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("FROZEN", source_digest.hexdigest(), len(entries), "files")


if __name__ == "__main__":
    main()
