"""Download official UCI archive and extract safely; no credentials required."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
OFFICIAL_DOWNLOAD_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"


def extract_safe(archive, destination):
    destination = Path(destination).resolve()
    for item in archive.infolist():
        target = (destination / item.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Unsafe archive member: {item.filename}")
    archive.extractall(destination)


def download(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "uci-har-official.zip"
    if not archive_path.exists():
        temporary = archive_path.with_suffix(".partial")
        print(f"Downloading official UCI archive to {archive_path}", flush=True)
        request = urllib.request.Request(URL, headers={"User-Agent": "smartphone-har-course-project/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(archive_path)
    raw = archive_path.read_bytes()
    outer_hash = hashlib.sha256(raw).hexdigest()
    if outer_hash != OFFICIAL_DOWNLOAD_SHA256:
        raise ValueError(f"Download differs from plan SHA256: {outer_hash}; review source before proceeding")
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        inner_names = [name for name in outer.namelist() if name.endswith("UCI HAR Dataset.zip")]
        dataset_bytes = outer.read(inner_names[0]) if inner_names else raw
        actual = hashlib.sha256(dataset_bytes).hexdigest()
        with zipfile.ZipFile(io.BytesIO(dataset_bytes)) as inner:
            extract_safe(inner, destination)
        for name in outer.namelist():
            if name.endswith(".names"):
                (destination / Path(name).name).write_bytes(outer.read(name))
    provenance = {"url": URL, "dataset_archive_sha256": actual,
                  "outer_archive_sha256": outer_hash,
                  "doi": "10.24432/C54S4K", "license": "CC BY 4.0"}
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(destination.resolve() / "UCI HAR Dataset", flush=True)
    return destination / "UCI HAR Dataset"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    download(args.destination)
