"""Tests for the download module. Nothing here uses the network or a real key:
the HTTP session is replaced by a small fake.
"""

import contextlib
import hashlib
import io
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from huasr import download
from huasr.download import DownloadError, HubSource
from tests import REPOSITORY


class FakeResponse:
    def __init__(self, status_code=200, payload=None, body=b""):
        self.status_code, self.payload, self.body = status_code, payload, body

    def json(self):
        if self.payload is None:
            raise ValueError("no JSON")
        return self.payload

    def iter_content(self, chunk_size):
        for start in range(0, len(self.body), 7):  # small, uneven chunks
            yield self.body[start:start + 7]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Records every request and answers with the queued responses."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def _answer(self, method, url, **options):
        self.requests.append((method, url, options))
        return self.responses.pop(0)

    def get(self, url, **options):
        return self._answer("GET", url, **options)

    def post(self, url, **options):
        return self._answer("POST", url, **options)


def grant(**changes):
    payload = {"downloadUrl": "https://storage.invalid/archive?signature=abc",
               "sizeBytes": str(download.ARCHIVE_BYTES), "checksum": download.ARCHIVE_SHA256,
               "filename": download.ARCHIVE_NAME}
    return {**payload, **changes}


def run_main(arguments, session=None, environment=None):
    """Run the command line with a fake session and environment; return (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ, environment or {}, clear=True), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = download.main(arguments, session=session or FakeSession())
    return code, out.getvalue(), err.getvalue()


class TemporaryFolderTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)


class PinnedSourceTests(unittest.TestCase):
    def test_every_pinned_file_has_a_size_and_a_sha256(self):
        for source in (download.VOXPOPULI, download.FLEURS):
            self.assertRegex(source.revision, r"^[0-9a-f]{40}$")
            for path, (size, digest) in source.files.items():
                self.assertGreater(size, 0, path)
                self.assertRegex(digest, r"^[0-9a-f]{64}$", path)
        self.assertRegex(download.ARCHIVE_SHA256, r"^[0-9a-f]{64}$")

    def test_voxpopuli_files_are_the_hungarian_parquet_files_of_the_three_splits(self):
        names = sorted(download.local_name(path) for path in download.VOXPOPULI.files)
        self.assertEqual([name.split("-")[0] for name in names], ["test"] + ["train"] * 4 + ["validation"])
        self.assertTrue(all(path.startswith("hu/") and path.endswith(".parquet") for path in download.VOXPOPULI.files))
        self.assertEqual(download.VOXPOPULI.licence, "CC0-1.0")

    def test_fleurs_files_are_the_hungarian_test_transcripts_and_audio(self):
        self.assertEqual(sorted(download.local_name(path) for path in download.FLEURS.files),
                         ["test.tar.gz", "test.tsv"])
        self.assertTrue(all(path.startswith("data/hu_hu/") for path in download.FLEURS.files))
        self.assertEqual(download.FLEURS.licence, "CC BY 4.0")

    def test_urls_point_at_the_pinned_revision(self):
        source = download.VOXPOPULI
        self.assertEqual(source.url("hu/test-00000-of-00001.parquet"),
                         f"https://huggingface.co/datasets/facebook/voxpopuli/resolve/{source.revision}"
                         "/hu/test-00000-of-00001.parquet")

    def test_files_can_be_limited_to_splits(self):
        chosen = download.select_files(download.VOXPOPULI, ["test", "validation"])
        self.assertEqual(sorted(download.local_name(path) for path in chosen),
                         ["test-00000-of-00001.parquet", "validation-00000-of-00001.parquet"])
        self.assertEqual(download.select_files(download.VOXPOPULI, None), download.VOXPOPULI.files)
        with self.assertRaises(DownloadError):
            download.select_files(download.FLEURS, ["train"])


class VerifyTests(TemporaryFolderTestCase):
    def test_matching_file_passes_and_other_files_fail(self):
        path = self.root / "archive.tar.gz"
        path.write_bytes(b"hungarian" * 1000)
        digest = hashlib.sha256(b"hungarian" * 1000).hexdigest()
        self.assertEqual(download.sha256_of(path), digest)
        download.verify_file(path, 9000, digest)
        with self.assertRaisesRegex(DownloadError, "bytes"):
            download.verify_file(path, 9001, digest)
        with self.assertRaisesRegex(DownloadError, "SHA-256"):
            download.verify_file(path, 9000, "0" * 64)
        with self.assertRaisesRegex(DownloadError, "not a file"):
            download.verify_file(self.root / "missing.tar.gz", 9000, digest)


class FetchTests(TemporaryFolderTestCase):
    BODY = bytes(range(256)) * 4

    def test_download_from_the_start(self):
        session = FakeSession(FakeResponse(200, body=self.BODY))
        target = download.fetch(session, "https://storage.invalid/a", self.root / "sub" / "a.tar.gz", len(self.BODY))
        self.assertEqual(target.read_bytes(), self.BODY)
        self.assertFalse(target.with_name("a.tar.gz.part").exists())
        self.assertNotIn("Range", session.requests[0][2]["headers"])
        self.assertNotIn("Authorization", session.requests[0][2]["headers"])

    def test_interrupted_download_continues(self):
        target = self.root / "a.tar.gz"
        target.with_name("a.tar.gz.part").write_bytes(self.BODY[:300])
        session = FakeSession(FakeResponse(206, body=self.BODY[300:]))
        download.fetch(session, "https://storage.invalid/a", target, len(self.BODY))
        self.assertEqual(target.read_bytes(), self.BODY)
        self.assertEqual(session.requests[0][2]["headers"]["Range"], "bytes=300-")

    def test_server_without_range_support_starts_again(self):
        target = self.root / "a.tar.gz"
        target.with_name("a.tar.gz.part").write_bytes(b"old" * 100)
        session = FakeSession(FakeResponse(200, body=self.BODY))
        download.fetch(session, "https://storage.invalid/a", target, len(self.BODY))
        self.assertEqual(target.read_bytes(), self.BODY)

    def test_short_download_keeps_the_part_file(self):
        target = self.root / "a.tar.gz"
        session = FakeSession(FakeResponse(200, body=self.BODY[:500]))
        with self.assertRaisesRegex(DownloadError, "run again"):
            download.fetch(session, "https://storage.invalid/a", target, len(self.BODY))
        self.assertFalse(target.exists())
        self.assertEqual(target.with_name("a.tar.gz.part").stat().st_size, 500)

    def test_server_error(self):
        with self.assertRaisesRegex(DownloadError, "HTTP 403"):
            download.fetch(FakeSession(FakeResponse(403)), "https://storage.invalid/a", self.root / "a", 10)


class HubTests(TemporaryFolderTestCase):
    FILES = {"hu/train-00000-of-00001.parquet": b"train rows" * 50, "hu/test-00000-of-00001.parquet": b"test rows" * 30}

    def source(self, **changed) -> HubSource:
        files = {path: (len(body), hashlib.sha256(body).hexdigest()) for path, body in self.FILES.items()}
        files.update(changed)
        return HubSource("someone/corpus", "a" * 40, "CC0-1.0", files)

    def fetch(self, session, source=None, **options):
        with contextlib.redirect_stdout(io.StringIO()):
            return download.fetch_hub(session, source or self.source(), self.root, **options)

    def test_files_are_downloaded_flat_and_checked(self):
        session = FakeSession(*(FakeResponse(200, body=body) for body in self.FILES.values()))
        paths = self.fetch(session)
        self.assertEqual([path.name for path in paths], ["train-00000-of-00001.parquet", "test-00000-of-00001.parquet"])
        self.assertEqual([path.read_bytes() for path in paths], list(self.FILES.values()))
        self.assertEqual([url for _, url, _ in session.requests],
                         [f"https://huggingface.co/datasets/someone/corpus/resolve/{'a' * 40}/{path}"
                          for path in self.FILES])

    def test_files_that_are_already_here_are_checked_but_not_requested(self):
        for path, body in self.FILES.items():
            (self.root / download.local_name(path)).write_bytes(body)
        session = FakeSession()
        self.assertEqual(len(self.fetch(session)), 2)
        self.assertEqual(session.requests, [])

    def test_a_file_with_other_content_is_refused(self):
        path = "hu/test-00000-of-00001.parquet"
        wrong = self.source(**{path: (len(self.FILES[path]), "0" * 64)})
        session = FakeSession(FakeResponse(200, body=self.FILES[path]))
        with self.assertRaisesRegex(DownloadError, "SHA-256"):
            self.fetch(session, wrong, splits=["test"])

    def test_only_the_asked_splits_are_requested(self):
        session = FakeSession(FakeResponse(200, body=self.FILES["hu/test-00000-of-00001.parquet"]))
        self.assertEqual([path.name for path in self.fetch(session, splits=["test"])], ["test-00000-of-00001.parquet"])
        self.assertEqual(len(session.requests), 1)

    def test_dry_run_lists_the_pinned_files_without_any_request(self):
        for arguments in (["voxpopuli", "--dry-run"], ["fleurs", "--dry-run", "--out", str(self.root)]):
            session = FakeSession()
            code, out, _ = run_main(arguments, session)
            self.assertEqual((code, session.requests), (0, []))
            source = download.HUB_SOURCES[arguments[0]]
            self.assertIn(source.revision, out)
            for path, (size, digest) in source.files.items():
                self.assertIn(f"{path}  {size} bytes  sha256 {digest}", out)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_command_line_downloads_into_the_given_folder(self):
        source = self.source()
        session = FakeSession(FakeResponse(200, body=self.FILES["hu/test-00000-of-00001.parquet"]))
        with mock.patch.dict(download.HUB_SOURCES, {"voxpopuli": source}):
            code, out, err = run_main(["voxpopuli", "--splits", "test", "--out", str(self.root)], session)
        self.assertEqual((code, err), (0, ""))
        self.assertTrue((self.root / "test-00000-of-00001.parquet").is_file())
        self.assertIn("match the pinned values", out)


def make_archive(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


class CommonVoiceStaysUntrackedTests(TemporaryFolderTestCase):
    def test_tracked_folders_of_the_repository_are_refused(self):
        for folder in ("data/raw", "data/raw/commonvoice", "data/clean", "scratch", "src", "."):
            with self.assertRaisesRegex(DownloadError, "re-hosting"):
                download.require_untracked(REPOSITORY / folder)

    def test_the_ignored_folder_and_folders_outside_the_repository_are_accepted(self):
        inside = REPOSITORY / "external" / "common_voice"
        self.assertEqual(download.require_untracked(inside), inside.resolve())
        self.assertEqual(download.require_untracked(self.root / "cv"), (self.root / "cv").resolve())

    def test_gitignore_excludes_common_voice_and_keeps_the_data_folders(self):
        lines = [line.strip() for line in (REPOSITORY / ".gitignore").read_text(encoding="utf-8").split("\n")]
        patterns = [line for line in lines if line and not line.startswith("#")]
        for needed in ("external/", "*.mp3", "cv-corpus-*/", "common-voice-*", "scratch/"):
            self.assertIn(needed, patterns)
        self.assertFalse([pattern for pattern in patterns if "data" in pattern or pattern in ("*.tar", "*.flac", "*.csv")])

    def test_default_target_is_the_ignored_folder(self):
        with mock.patch.object(download, "common_voice", return_value=0) as called:
            run_main(["commonvoice"])
        self.assertEqual(called.call_args.args[0].out, Path("external") / "common_voice")

    def test_extraction_into_the_data_folder_is_refused_before_anything_is_written(self):
        archive = self.root / "cv.tar.gz"
        make_archive(archive, {"cv-corpus-0.0-synthetic/hu/test.tsv": b"client_id\n"})
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        target = REPOSITORY / "data" / "raw" / "commonvoice"
        with mock.patch.object(download, "ARCHIVE_BYTES", archive.stat().st_size), \
                mock.patch.object(download, "ARCHIVE_SHA256", digest):
            code, _, err = run_main(["commonvoice", "--archive", str(archive), "--extract", "--out", str(target)])
            with self.assertRaisesRegex(DownloadError, "re-hosting"):
                download.extract(archive, target)
        self.assertEqual(code, 1)
        self.assertIn("re-hosting", err)
        self.assertFalse(target.exists())

    def test_an_archive_lying_in_the_tracked_part_is_refused(self):
        code, _, err = run_main(["commonvoice", "--archive", str(REPOSITORY / "data" / "raw" / "cv.tar.gz"),
                                 "--out", str(self.root)])
        self.assertEqual(code, 1)
        self.assertIn("re-hosting", err)


class CommonVoiceApiTests(unittest.TestCase):
    def test_key_goes_into_the_authorization_header_only(self):
        session = FakeSession(FakeResponse(200, grant()))
        self.assertEqual(download.request_download(session, "test-key")["downloadUrl"], grant()["downloadUrl"])
        method, url, options = session.requests[0]
        self.assertEqual((method, url), ("POST", f"{download.API_URL}/datasets/{download.DATASET_ID}/download"))
        self.assertEqual(options["headers"]["Authorization"], "Bearer test-key")
        self.assertNotIn("test-key", url)

    def test_refusals_are_explained(self):
        cases = {401: download.KEY_VARIABLE, 403: "terms", 429: "limit", 500: "HTTP 500"}
        for status, expected in cases.items():
            session = FakeSession(FakeResponse(status, {"error": "refused"}))
            with self.assertRaisesRegex(DownloadError, expected):
                download.request_download(session, "test-key")

    def test_a_changed_file_on_the_server_is_refused(self):
        for changed in (grant(checksum="1" * 64), grant(sizeBytes="5"), {"sizeBytes": "5"}):
            with self.assertRaises(DownloadError):
                download.request_download(FakeSession(FakeResponse(200, changed)), "test-key")

    def test_public_description_needs_no_key(self):
        session = FakeSession(FakeResponse(200, {"name": "x", "sizeBytes": download.ARCHIVE_BYTES,
                                                 "checksum": download.ARCHIVE_SHA256}))
        info = download.dataset_info(session)
        download.check_published(info["sizeBytes"], info["checksum"])
        self.assertNotIn("Authorization", session.requests[0][2]["headers"])


class CommonVoiceCommandLineTests(TemporaryFolderTestCase):
    def test_without_a_key_the_manual_steps_are_printed_and_nothing_is_requested(self):
        session = FakeSession()
        code, _, err = run_main(["commonvoice", "--out", str(self.root)], session)
        self.assertEqual(code, 2)
        self.assertIn(download.KEY_VARIABLE, err)
        self.assertIn(download.DATASET_PAGE, err)
        self.assertIn("--archive", err)
        self.assertEqual(session.requests, [])

    def test_download_check_and_extraction_with_a_key_from_the_environment(self):
        # A tiny stand-in archive; the pinned size and checksum are replaced by its own for this test.
        make_archive(self.root / "served.tar.gz", {"cv-corpus-0.0-synthetic/hu/clips/a.mp3": b"audio",
                                                   "cv-corpus-0.0-synthetic/hu/test.tsv": b"client_id\n"})
        body = (self.root / "served.tar.gz").read_bytes()
        size, digest = len(body), hashlib.sha256(body).hexdigest()
        session = FakeSession(FakeResponse(200, grant(sizeBytes=str(size), checksum=digest)),
                              FakeResponse(200, body=body))
        out_folder = self.root / "external"
        with mock.patch.object(download, "ARCHIVE_BYTES", size), mock.patch.object(download, "ARCHIVE_SHA256", digest):
            code, out, err = run_main(["commonvoice", "--out", str(out_folder), "--extract"], session,
                                      {download.KEY_VARIABLE: "secret-key"})
        self.assertEqual((code, err), (0, ""))
        self.assertEqual((out_folder / download.ARCHIVE_NAME).read_bytes(), body)
        self.assertEqual((out_folder / "cv-corpus-0.0-synthetic" / "hu" / "clips" / "a.mp3").read_bytes(), b"audio")
        self.assertEqual(session.requests[0][2]["headers"]["Authorization"], "Bearer secret-key")
        self.assertNotIn("Authorization", session.requests[1][2]["headers"])  # the storage URL is pre-signed
        self.assertNotIn("secret-key", out)
        self.assertNotIn("signature", out)

    def test_a_wrong_local_archive_is_rejected(self):
        archive = self.root / "wrong.tar.gz"
        archive.write_bytes(b"something else")
        code, _, err = run_main(["commonvoice", "--archive", str(archive), "--extract", "--out",
                                 str(self.root / "out")])
        self.assertEqual(code, 1)
        self.assertIn("bytes", err)
        self.assertFalse((self.root / "out").exists())

    def test_member_outside_the_target_folder_is_refused(self):
        archive = self.root / "a.tar.gz"
        make_archive(archive, {"../escaped.txt": b"x"})
        with self.assertRaises(tarfile.FilterError):
            download.extract(archive, self.root / "out")
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_archive_without_split_files(self):
        archive = self.root / "a.tar.gz"
        make_archive(archive, {"readme.txt": b"x"})
        with self.assertRaisesRegex(DownloadError, "test.tsv"):
            download.extract(archive, self.root / "out")

    def test_info_compares_with_the_pinned_values(self):
        published = {"name": "Common Voice Scripted Speech 27.0 - Hungarian", "locale": "hu",
                     "license": "Creative Commons Zero v1.0 Universal (CC0-1.0)", "filename": download.ARCHIVE_NAME,
                     "sizeBytes": download.ARCHIVE_BYTES, "checksum": download.ARCHIVE_SHA256}
        code, out, _ = run_main(["commonvoice", "--info"], FakeSession(FakeResponse(200, published)))
        self.assertEqual(code, 0)
        self.assertIn("CC0-1.0", out)
        self.assertIn("match", out)
        code, _, err = run_main(["commonvoice", "--info"],
                                FakeSession(FakeResponse(200, {**published, "checksum": "f" * 64})))
        self.assertEqual(code, 1)
        self.assertIn("pinned", err)


if __name__ == "__main__":
    unittest.main()
