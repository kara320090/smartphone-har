"""Verify all declared MLP release files using only the Python standard library."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def checked_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value or not path.parts:
        raise ValueError(f"Unsafe manifest path: {value}")
    return path.as_posix()


def stream_digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def verify_directory(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "MLP_RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        relative = checked_relative(entry["path"])
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or changed file: {relative}")
        with path.open("rb") as stream:
            if stream_digest(stream) != entry["sha256"]:
                raise ValueError(f"Checksum differs: {relative}")
    return {"passed": True, "files_verified": len(manifest["files"]),
            "source_commit": manifest["source_commit"], "release_tag": manifest["release_tag"]}


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        prefix = "smartphone-har/"
        manifest_name = prefix + "MLP_RELEASE_MANIFEST.json"
        manifest = json.loads(archive.read(manifest_name))
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Duplicate ZIP members")
        expected = {manifest_name}
        for entry in manifest["files"]:
            relative = checked_relative(entry["path"])
            name = prefix + relative
            expected.add(name)
            if archive.getinfo(name).file_size != entry["bytes"]:
                raise ValueError(f"File size differs: {relative}")
            with archive.open(name) as stream:
                if stream_digest(stream) != entry["sha256"]:
                    raise ValueError(f"Checksum differs: {relative}")
        actual = {member.filename for member in archive.infolist() if not member.is_dir()}
        if actual != expected:
            raise ValueError("ZIP contains missing or undeclared files")
        return {"passed": True, "files_verified": len(manifest["files"]),
                "source_commit": manifest["source_commit"], "release_tag": manifest["release_tag"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--directory", type=Path)
    source.add_argument("--archive", type=Path)
    args = parser.parse_args()
    result = verify_directory(args.directory) if args.directory else verify_archive(args.archive)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
