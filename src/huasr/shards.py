"""Tar shards and the manifest that describes them.

Raw audio is kept in uncompressed tar files ("shards") that each stay below
GitHub's file-size limit, next to a `manifest.csv` with one row per clip. The
tar files are written so that their bytes depend only on the member names and
contents (no time stamps, owners or modes that vary between machines).
`unpack` restores the clips and checks every SHA-256 against the manifest.
"""

from __future__ import annotations

import csv
import hashlib
import io
import tarfile
from pathlib import Path

MAX_SHARD_BYTES = 95_000_000  # GitHub rejects files of 100 MB and more
TAR_BLOCK = 512


def member_size(file_size: int) -> int:
    """Bytes one file takes in a tar: a header block plus the content padded to full blocks."""
    return TAR_BLOCK + -(-file_size // TAR_BLOCK) * TAR_BLOCK


def archive_size(member_bytes: int) -> int:
    """Tar size for members of this total size: two end blocks, padded to a full record."""
    return -(-(member_bytes + 2 * TAR_BLOCK) // tarfile.RECORDSIZE) * tarfile.RECORDSIZE


def tar_size(file_sizes: list[int]) -> int:
    """Size of an uncompressed tar holding files of these sizes, as written by `ShardWriter`."""
    return archive_size(sum(member_size(size) for size in file_sizes))


def is_plain_name(name: str) -> bool:
    """A file name without folder part and without characters that Windows refuses."""
    return bool(name) and name not in (".", "..") and not any(char in name for char in '/\\:*?"<>|')


class ShardWriter:
    """Writes files into `<folder>/<prefix>-000.tar`, `-001.tar`, ... and starts a new tar before the limit."""

    def __init__(self, folder: Path, prefix: str, max_bytes: int = MAX_SHARD_BYTES):
        self.folder, self.prefix, self.max_bytes = Path(folder), prefix, max_bytes
        self.shards: list[tuple[str, int]] = []  # (file name, size in bytes) of every closed shard
        self._tar, self._path, self._used = None, None, 0

    def add(self, name: str, data: bytes) -> str:
        """Append one file and return the name of the shard that holds it."""
        if not is_plain_name(name):
            raise ValueError(f"not a plain file name: {name!r}")
        size = member_size(len(data))
        if archive_size(size) >= self.max_bytes:
            raise ValueError(f"{name} ({len(data)} bytes) does not fit in a shard of {self.max_bytes} bytes")
        if self._tar is not None and archive_size(self._used + size) >= self.max_bytes:
            self._finish()
        if self._tar is None:
            self.folder.mkdir(parents=True, exist_ok=True)
            self._path = self.folder / f"{self.prefix}-{len(self.shards):03d}.tar"
            self._tar, self._used = tarfile.open(self._path, "w", format=tarfile.USTAR_FORMAT), 0
        info = tarfile.TarInfo(name)
        info.size, info.mtime, info.mode = len(data), 0, 0o644
        self._tar.addfile(info, io.BytesIO(data))
        self._used += size
        return self._path.name

    def _finish(self) -> None:
        self._tar.close()
        size = self._path.stat().st_size
        if size >= self.max_bytes:
            raise RuntimeError(f"{self._path.name} is {size} bytes, limit {self.max_bytes}")
        self.shards.append((self._path.name, size))
        self._tar = None

    def close(self) -> list[tuple[str, int]]:
        if self._tar is not None:
            self._finish()
        return self.shards


def write_manifest(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(folder: Path) -> list[dict[str, str]]:
    with (Path(folder) / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def unpack(folder: Path, out: Path) -> int:
    """Extract every clip of `folder/manifest.csv` to `out/<split>/<file>`, checking each SHA-256.

    Returns the number of clips. Stops at the first member that is not in the
    manifest or whose content does not match it.
    """
    folder, out = Path(folder), Path(out)
    by_shard: dict[str, dict[str, dict[str, str]]] = {}
    for row in read_manifest(folder):
        by_shard.setdefault(row["shard"], {})[row["file"]] = row
    count = 0
    for shard, expected in sorted(by_shard.items()):
        seen = set()
        with tarfile.open(folder / shard, "r:") as tar:
            for member in tar:
                row = expected.get(member.name)
                if row is None or not member.isfile() or not is_plain_name(member.name) \
                        or not is_plain_name(row["split"]):
                    raise ValueError(f"{shard}: unexpected member {member.name!r}")
                data = tar.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError(f"{shard}: {member.name} does not match the SHA-256 in the manifest")
                target = out / row["split"] / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                seen.add(member.name)
        if seen != set(expected):
            raise ValueError(f"{shard}: {len(set(expected) - seen)} clips of the manifest are missing")
        count += len(seen)
    return count
