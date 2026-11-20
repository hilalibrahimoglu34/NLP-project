"""Download the source files of the three corpora and check them.

    python -m huasr.download voxpopuli [--splits train validation test] [--dry-run]
    python -m huasr.download fleurs [--dry-run]
    python -m huasr.download commonvoice --info
    python -m huasr.download commonvoice --archive <file.tar.gz> --extract
    python -m huasr.download commonvoice --extract            (needs MDC_API_KEY)

VoxPopuli (training, development and main test data) and FLEURS (extra test
set) come from the Hugging Face Hub without an account. Every file is pinned
to one dataset revision, and its size and SHA-256 are checked after the
download, so everyone builds from the same bytes. They go to `scratch/`,
which git ignores; `huasr.subset` turns them into the raw data of `data/raw/`.

Common Voice is used as an external test set only. Its dataset page forbids
re-hosting, so its files are never written into the tracked part of this
repository: the default target is `external/common_voice/` (ignored by git),
and any other target inside the repository is refused. Common Voice is
distributed by the Mozilla Data Collective (MDC). A download needs an MDC
account with a complete profile, and the terms of the dataset have to be
accepted on its web page; a script cannot do either (see MANUAL_STEPS). After
that the archive is either downloaded in the browser and passed with
--archive, or fetched with an API key from the environment variable
MDC_API_KEY. The key is read from the environment only; it is never printed
or written to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import requests

HEADERS = {"User-Agent": "huasr-download/0.1"}
TIMEOUT = (10, 60)  # seconds: connect, read
CHUNK = 1 << 20


class DownloadError(RuntimeError):
    """A file could not be obtained, is not the expected file, or may not be stored where asked."""


# ---------------------------------------------------------------- Hugging Face Hub (VoxPopuli, FLEURS)

@dataclass(frozen=True)
class HubSource:
    """Files of one dataset repository on the Hugging Face Hub, pinned to a revision."""

    repo: str
    revision: str
    licence: str
    files: dict[str, tuple[int, str]]  # path in the repository -> (size in bytes, SHA-256)

    def url(self, path: str) -> str:
        return f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{path}"


# Sizes and SHA-256 values as listed by the Hub for these revisions (tree API, field lfs.oid);
# the checksum of the small text file test.tsv was computed from the file of that revision.
VOXPOPULI = HubSource(
    repo="facebook/voxpopuli",
    revision="42f01879c780b4a2e90ec0b4f616c2ece526e4f1",
    licence="CC0-1.0",
    files={
        "hu/train-00000-of-00004.parquet":
            (2_595_897_654, "08e14edd2bdfb5f1c46d06553b9d62f3c105eb66d7f182ea3edadd94298f88cb"),
        "hu/train-00001-of-00004.parquet":
            (2_593_326_688, "4319b25926833a69d5109a97944faec8d9b41f0539aa5a607bd6789c63e393ef"),
        "hu/train-00002-of-00004.parquet":
            (2_613_856_059, "828ac5e9ce837737b6a020cddb9d8a3b01224bb8bc2b5ec5e19702c2cbb69675"),
        "hu/train-00003-of-00004.parquet":
            (2_572_374_907, "35619abd6771ea783346ae2fdac6f5acdcfb8ace5ed2f833bf098e3a1f3fd068"),
        "hu/validation-00000-of-00001.parquet":
            (608_942_228, "8088dc7e889a58e88cd25292f188b1934435558768432644acd3e0a9ea9eabeb"),
        "hu/test-00000-of-00001.parquet":
            (594_503_843, "cbb60430877088155443c0cf6e3719139bd90d2b0f9854e9e641c5c8d72a9b16"),
    },
)
FLEURS = HubSource(
    repo="google/fleurs",
    revision="70bb2e84b976b7e960aa89f1c648e09c59f894dd",
    licence="CC BY 4.0",
    files={
        "data/hu_hu/test.tsv": (561_505, "939accd545532c3b0eb34891742829314267efd6c67a4a5ac5e15aef23dec82f"),
        "data/hu_hu/audio/test.tar.gz":
            (542_250_479, "d2335311350b4c9a6a1811b3ca3faff772a97befe0a7106dbda1edf675b0cb8a"),
    },
)
HUB_SOURCES = {"voxpopuli": VOXPOPULI, "fleurs": FLEURS}


def local_name(path: str) -> str:
    """Hub files are stored flat: only the last part of their path is used locally."""
    return PurePosixPath(path).name


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    """Raise unless the file has this size and SHA-256."""
    if not Path(path).is_file():
        raise DownloadError(f"{path} is not a file")
    size = Path(path).stat().st_size
    if size != expected_bytes:
        raise DownloadError(f"{path} has {size} bytes, expected {expected_bytes}")
    found = sha256_of(path)
    if found != expected_sha256:
        raise DownloadError(f"{path} has SHA-256 {found}, expected {expected_sha256}")


def fetch(session, url: str, target: Path, total_bytes: int, show_progress: bool = False) -> Path:
    """Stream `url` to `target`; an interrupted download continues from its `.part` file."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > total_bytes:
        part.unlink()
        offset = 0
    if offset < total_bytes:
        headers = {**HEADERS, "Range": f"bytes={offset}-"} if offset else HEADERS
        with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as response:
            if response.status_code not in (200, 206):
                raise DownloadError(f"the server answered HTTP {response.status_code} for {target.name}")
            # 200 after a Range request: the server sent the whole file, so start again.
            resumed = response.status_code == 206
            with part.open("ab" if resumed else "wb") as handle:
                done, step = (offset if resumed else 0), max(total_bytes // 20, 1)
                for chunk in response.iter_content(chunk_size=CHUNK):
                    handle.write(chunk)
                    if show_progress and (done + len(chunk)) // step > done // step:  # about every 5%
                        print(f"  {(done + len(chunk)) / 1e9:.2f} of {total_bytes / 1e9:.2f} GB", flush=True)
                    done += len(chunk)
    if part.stat().st_size != total_bytes:
        raise DownloadError(f"{part} has {part.stat().st_size} of {total_bytes} bytes; run again to continue")
    part.replace(target)
    return target


def select_files(source: HubSource, splits: list[str] | None) -> dict[str, tuple[int, str]]:
    """The pinned files, or only those whose name starts with one of `splits`."""
    if not splits:
        return dict(source.files)
    chosen = {path: pinned for path, pinned in source.files.items()
              if any(local_name(path).startswith(split) for split in splits)}
    if not chosen:
        raise DownloadError(f"no pinned file of {source.repo} starts with {splits}")
    return chosen


def fetch_hub(session, source: HubSource, out: Path, splits: list[str] | None = None,
              show_progress: bool = False) -> list[Path]:
    """Download the pinned files of `source` into `out` and check size and SHA-256 of each one."""
    paths = []
    for path, (size, digest) in select_files(source, splits).items():
        target = Path(out) / local_name(path)
        if target.is_file():
            print(f"{target.name}: already here, checking")
        else:
            print(f"{target.name}: downloading {size} bytes")
            fetch(session, source.url(path), target, size, show_progress)
        verify_file(target, size, digest)
        print(f"{target.name}: size and SHA-256 match the pinned values")
        paths.append(target)
    return paths


# ---------------------------------------------------------------- Common Voice (external test set only)

API_URL = "https://mozilladatacollective.com/api"
DATASET_ID = "cmu5wq4u200dknq072xswzofb"
DATASET_PAGE = f"https://mozilladatacollective.com/datasets/{DATASET_ID}"
# Published for cv-corpus-27.0-2026-09-11, locale hu, in the release statistics
# (github.com/common-voice/cv-dataset) and by the MDC API (GET /datasets/<id>).
ARCHIVE_NAME = "common-voice-scripted-speech-27-0-hungar-9f5b5529.tar.gz"
ARCHIVE_BYTES = 3_846_979_882
ARCHIVE_SHA256 = "0912c72c5ecbd0bfb4a8721ed83b7f31626ea8dedb4c4fbc7319bce815d5f0e6"
KEY_VARIABLE = "MDC_API_KEY"
EXTERNAL = "external"  # folder of the repository that git ignores

MANUAL_STEPS = f"""\
Common Voice 27.0 Hungarian cannot be downloaded without an account. Do this yourself, once:
  1. Create an account at https://mozilladatacollective.com and complete the profile.
  2. Open {DATASET_PAGE}
     read the licence (CC0-1.0) and the conditions of use, and accept them there.
  3. Then either download {ARCHIVE_NAME}
     in the browser and run:   python -m huasr.download commonvoice --archive <file> --extract
     or create an API key in the MDC profile, set the environment variable {KEY_VARIABLE}
     to it and run:            python -m huasr.download commonvoice --extract
Do not write the key into any file of this repository."""


def repository_of(path: Path) -> Path | None:
    """The checkout of this project that contains `path`, if there is one."""
    for folder in (path, *path.parents):
        if (folder / "pyproject.toml").is_file() and (folder / "src" / "huasr").is_dir():
            return folder
    return None


def require_untracked(path: Path) -> Path:
    """Refuse a target for Common Voice files that lies in the tracked part of the repository.

    Allowed: any folder outside the repository, and `external/` inside it,
    which `.gitignore` excludes. The dataset page of Common Voice forbids
    re-hosting or re-sharing the dataset.
    """
    resolved = Path(path).resolve()
    root = repository_of(resolved)
    if root is not None and resolved.relative_to(root).parts[:1] != (EXTERNAL,):
        raise DownloadError(f"Common Voice files may not be stored in {resolved}: its terms forbid re-hosting, "
                            f"and only {root / EXTERNAL} is ignored by git. Use that folder or one outside the "
                            f"repository.")
    return resolved


def check_published(size_bytes, checksum) -> None:
    """Raise if MDC no longer serves the file this project is pinned to."""
    if int(size_bytes) != ARCHIVE_BYTES or checksum != ARCHIVE_SHA256:
        raise DownloadError(f"MDC now lists {size_bytes} bytes with SHA-256 {checksum} for this dataset; "
                            f"this project is pinned to {ARCHIVE_BYTES} bytes, {ARCHIVE_SHA256}")


def _message(response) -> str:
    try:
        body = response.json()
        return str(body.get("message") or body.get("error") or "")
    except ValueError:
        return ""


def dataset_info(session) -> dict:
    """Public description of the Common Voice dataset (name, size, checksum, licence); needs no key."""
    response = session.get(f"{API_URL}/datasets/{DATASET_ID}", headers=HEADERS, timeout=TIMEOUT)
    if response.status_code != 200:
        raise DownloadError(f"MDC answered HTTP {response.status_code} for the dataset description")
    return response.json()


def request_download(session, api_key: str) -> dict:
    """Ask MDC for a short-lived download URL of the archive (counts against the daily download limit)."""
    response = session.post(f"{API_URL}/datasets/{DATASET_ID}/download",
                            headers={**HEADERS, "Authorization": f"Bearer {api_key}"}, timeout=TIMEOUT)
    status = response.status_code
    if status == 401:
        raise DownloadError(f"MDC did not accept the API key in {KEY_VARIABLE}")
    if status == 403:
        raise DownloadError(f"MDC denied access ({_message(response) or 'no reason given'}). Complete your MDC "
                            f"profile and accept the dataset's terms on {DATASET_PAGE}")
    if status == 429:
        raise DownloadError("MDC's download limit for today is reached; try again after midnight UTC "
                            "or download in the browser and use --archive")
    if status != 200:
        raise DownloadError(f"MDC answered HTTP {status} {_message(response)}".rstrip())
    grant = response.json()
    if not grant.get("downloadUrl"):
        raise DownloadError("MDC's answer has no download URL")
    check_published(grant.get("sizeBytes", 0), grant.get("checksum"))
    return grant


def extract(archive: Path, out: Path) -> Path:
    """Unpack the Common Voice archive below `out` and return the folder that holds test.tsv."""
    out = require_untracked(out)
    out.mkdir(parents=True, exist_ok=True)
    release, count = None, 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            tar.extract(member, out, filter="data")  # the "data" filter refuses members that leave `out`
            count += 1
            if count % 20_000 == 0:
                print(f"  {count} files", flush=True)
            if PurePosixPath(member.name).name == "test.tsv":
                release = out / PurePosixPath(member.name).parent
    if release is None:
        raise DownloadError(f"{archive} has no test.tsv")
    return release


def common_voice(args, session) -> int:
    if args.info:
        info = dataset_info(session)
        for field in ("name", "locale", "license", "filename", "sizeBytes", "checksum", "datasetUrl"):
            print(f"{field}: {info.get(field)}")
        check_published(info.get("sizeBytes", 0), info.get("checksum"))
        print("size and checksum match the values pinned in huasr.download")
        return 0
    out = require_untracked(args.out)
    if args.archive:
        archive = require_untracked(args.archive)
    else:
        api_key = os.environ.get(KEY_VARIABLE, "").strip()
        if not api_key:
            print(MANUAL_STEPS, file=sys.stderr)
            return 2
        archive = out / ARCHIVE_NAME
        if not archive.exists():
            grant = request_download(session, api_key)
            print(f"downloading {ARCHIVE_NAME} to {out}")
            fetch(session, grant["downloadUrl"], archive, ARCHIVE_BYTES, show_progress=True)
    print(f"checking size and SHA-256 of {archive}")
    verify_file(archive, ARCHIVE_BYTES, ARCHIVE_SHA256)
    print("the archive is the published file")
    if args.extract:
        print(f"extracting to {out} (this takes a while)")
        print(f"release folder: {extract(archive, out)}")
    return 0


# ---------------------------------------------------------------- command line

def hub(args, session) -> int:
    source = HUB_SOURCES[args.corpus]
    files = select_files(source, getattr(args, "splits", None))
    print(f"{source.repo} at revision {source.revision} ({source.licence}), {len(files)} files, "
          f"{sum(size for size, _ in files.values())} bytes in total -> {args.out}")
    for path, (size, digest) in files.items():
        print(f"  {path}  {size} bytes  sha256 {digest}")
    if not args.dry_run:
        fetch_hub(session, source, args.out, getattr(args, "splits", None), show_progress=True)
    return 0


def main(argv: list[str] | None = None, session=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    corpora = parser.add_subparsers(dest="corpus", required=True)
    voxpopuli = corpora.add_parser("voxpopuli", help="VoxPopuli hu parquet files (Hugging Face Hub)")
    voxpopuli.add_argument("--splits", nargs="+", choices=("train", "validation", "test"),
                           help="download only these official splits")
    fleurs = corpora.add_parser("fleurs", help="FLEURS hu_hu test transcripts and audio (Hugging Face Hub)")
    for hub_parser, folder in ((voxpopuli, "scratch/voxpopuli"), (fleurs, "scratch/fleurs")):
        hub_parser.add_argument("--out", type=Path, default=Path(folder), help="target folder (not tracked by git)")
        hub_parser.add_argument("--dry-run", action="store_true", help="list the pinned files and stop")
    cv = corpora.add_parser("commonvoice", help="Common Voice 27.0 hu archive (external test set, never tracked)")
    cv.add_argument("--info", action="store_true",
                    help="print what MDC publishes about the dataset and compare it with the pinned values")
    cv.add_argument("--archive", type=Path, help="archive downloaded by hand from the dataset page")
    cv.add_argument("--out", type=Path, default=Path(EXTERNAL) / "common_voice",
                    help="target folder; inside the repository only external/ is accepted")
    cv.add_argument("--extract", action="store_true", help="unpack the archive after the check")
    args = parser.parse_args(argv)
    session = session or requests.Session()

    try:
        return common_voice(args, session) if args.corpus == "commonvoice" else hub(args, session)
    except DownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except requests.RequestException as error:
        # The message of a network error can contain a signed download URL, so only its type is shown.
        print(f"error: network request failed ({type(error).__name__}); run again to continue", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
