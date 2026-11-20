"""Tests for the tar shards and for unpacking them against the manifest."""

import hashlib
import random
import tarfile
import tempfile
import unittest
from pathlib import Path

from huasr import shards
from huasr.shards import ShardWriter, tar_size, unpack, write_manifest

COLUMNS = ("clip_id", "split", "file", "shard", "sha256")


def blobs(count: int, seed: int = 0) -> dict[str, bytes]:
    rng = random.Random(seed)
    return {f"clip{number:03d}.flac": rng.randbytes(rng.randint(1, 5000)) for number in range(count)}


class TemporaryFolderTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def pack(self, files: dict[str, bytes], folder: Path, split: str = "train", max_bytes: int = 40_960):
        """Write `files` into shards and a manifest below `folder`."""
        writer = ShardWriter(folder / "clips", split, max_bytes)
        rows = [{"clip_id": name[:-5], "split": split, "file": name, "shard": f"clips/{writer.add(name, data)}",
                 "sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()]
        closed = writer.close()
        write_manifest(folder / "manifest.csv", COLUMNS, rows)
        return rows, closed


class ShardWriterTests(TemporaryFolderTestCase):
    def test_every_shard_is_below_the_limit_and_nothing_is_lost(self):
        files = blobs(120)
        rows, closed = self.pack(files, self.root)
        self.assertGreater(len(closed), 1)
        self.assertEqual([name for name, _ in closed], sorted(path.name for path in (self.root / "clips").iterdir()))
        packed = {}
        for name, size in closed:
            path = self.root / "clips" / name
            self.assertEqual(path.stat().st_size, size)
            self.assertLess(size, 40_960)
            with tarfile.open(path) as tar:
                content = {member.name: tar.extractfile(member).read() for member in tar}
            self.assertEqual(size, tar_size([len(data) for data in content.values()]))
            for member in content:
                self.assertEqual(next(row["shard"] for row in rows if row["file"] == member), f"clips/{name}")
            packed.update(content)
        self.assertEqual(packed, files)

    def test_same_input_gives_identical_shards(self):
        files = blobs(40)
        self.pack(files, self.root / "a")
        self.pack(files, self.root / "b")
        for path in sorted((self.root / "a" / "clips").iterdir()):
            self.assertEqual(path.read_bytes(), (self.root / "b" / "clips" / path.name).read_bytes())

    def test_a_file_larger_than_a_shard_is_an_error(self):
        writer = ShardWriter(self.root, "train", max_bytes=20_480)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            writer.add("big.flac", bytes(30_000))
        self.assertEqual(writer.close(), [])

    def test_only_plain_file_names_are_accepted(self):
        writer = ShardWriter(self.root, "train")
        for name in ("../x.flac", "a/b.flac", "a\\b.flac", "20:15.flac", ""):
            with self.assertRaisesRegex(ValueError, "plain file name"):
                writer.add(name, b"x")

    def test_default_limit_is_below_the_github_file_limit(self):
        self.assertLessEqual(shards.MAX_SHARD_BYTES, 95_000_000)

    def test_no_shard_is_written_for_no_clips(self):
        self.assertEqual(ShardWriter(self.root / "clips", "test").close(), [])
        self.assertFalse((self.root / "clips").exists())


class UnpackTests(TemporaryFolderTestCase):
    def test_unpack_restores_every_clip_under_its_split(self):
        files = blobs(30)
        self.pack(files, self.root / "raw", split="validation")
        self.assertEqual(unpack(self.root / "raw", self.root / "audio"), 30)
        for name, data in files.items():
            self.assertEqual((self.root / "audio" / "validation" / name).read_bytes(), data)

    def test_a_changed_shard_is_detected(self):
        self.pack(blobs(10), self.root / "raw")
        shard = sorted((self.root / "raw" / "clips").iterdir())[0]
        data = bytearray(shard.read_bytes())
        data[512] ^= 0xFF  # first content byte of the first clip
        shard.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            unpack(self.root / "raw", self.root / "audio")

    def test_a_member_that_is_not_in_the_manifest_is_detected(self):
        rows, _ = self.pack(blobs(6), self.root / "raw")
        write_manifest(self.root / "raw" / "manifest.csv", COLUMNS, rows[:-1])
        with self.assertRaisesRegex(ValueError, "unexpected member"):
            unpack(self.root / "raw", self.root / "audio")

    def test_a_missing_clip_is_detected(self):
        rows, _ = self.pack(blobs(6), self.root / "raw")
        ghost = {**rows[0], "clip_id": "ghost", "file": "ghost.flac"}
        write_manifest(self.root / "raw" / "manifest.csv", COLUMNS, [*rows, ghost])
        with self.assertRaisesRegex(ValueError, "missing"):
            unpack(self.root / "raw", self.root / "audio")


if __name__ == "__main__":
    unittest.main()
