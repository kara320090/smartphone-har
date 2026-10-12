"""Recover the team's original fit statistics from a verified public release.

No statistics are fitted and no test arrays are accessed. Run from repo root.
"""
import argparse
import hashlib
from pathlib import Path
import zipfile


SOURCE_URL = "https://github.com/kara320090/smartphone-har/releases/download/bongheon-lstm-v1/bongheon-lstm-v1-full.zip"
SOURCE_SHA256 = "f97fcdd145cbd0f6ac777ff68d576a2b7fbf832d6684e1fa36073d3b48f747a0"
STATS_SHA256 = "42b1b895d420a21ed7cce3b3636a4cbed96350953b27057c0670db9062d79fb2"


def recover_statistics(bundle_path, output_path):
    bundle_path, output_path = Path(bundle_path), Path(output_path)
    with bundle_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Release ZIP checksum differs from the frozen source")
    with zipfile.ZipFile(bundle_path) as archive:
        content = archive.read("data/preprocess.npz")
    if hashlib.sha256(content).hexdigest() != STATS_SHA256:
        raise ValueError("Original preprocessing checksum differs")
    if output_path.exists():
        if output_path.read_bytes() != content:
            raise FileExistsError("Different statistics already exist; use another output path")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content)
    return {"release_url": SOURCE_URL, "release_sha256": digest,
            "member": "data/preprocess.npz", "statistics_sha256": STATS_SHA256,
            "statistics_refitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/preprocess.npz"))
    args = parser.parse_args()
    print(recover_statistics(args.source_bundle, args.output))


if __name__ == "__main__":
    main()
