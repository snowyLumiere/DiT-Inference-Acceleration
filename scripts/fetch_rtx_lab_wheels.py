#!/usr/bin/env python3
"""Fetch or verify the exact public wheels for the offline RTX teaching labs.

Uses only Python's standard library. No package is installed and no GPU is used.
Existing files must match; a failed transfer leaves its .part file for inspection.
The caller chooses and checks the storage location before using this command.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HOSTS = {"download.pytorch.org", "download-r2.pytorch.org", "pypi.nvidia.com", "files.pythonhosted.org"}
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "requirements/rtx-lab-py310-cu128-wheels-public.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path):
    raw = path.read_bytes()
    obj = json.loads(raw.decode("utf-8"))
    rows = obj["wheels"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError("manifest must contain 1..100 wheels")
    seen = set()
    for row in rows:
        name = row["filename"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+%-]*\.whl", name) or name in seen:
            raise ValueError("unsafe or duplicate wheel filename")
        seen.add(name)
        if type(row["bytes"]) is not int or not 0 < row["bytes"] <= 10 * 1024**3:
            raise ValueError("invalid wheel size")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("invalid SHA256")
        url = validate_url(row["source_url"])
        if url.fragment and url.fragment != "sha256=" + row["sha256"]:
            raise ValueError("source URL hash fragment and manifest SHA256 disagree")
        if urllib.parse.unquote(url.path.rsplit("/", 1)[-1]) != name:
            raise ValueError("source URL and wheel filename disagree")
    if sum(row["bytes"] for row in rows) > 20 * 1024**3:
        raise ValueError("wheelhouse exceeds 20 GiB guard")
    return obj, hashlib.sha256(raw).hexdigest()


def validate_url(value):
    url = urllib.parse.urlsplit(value)
    if url.scheme != "https" or url.hostname not in HOSTS or url.port not in (None, 443) or url.username or url.password or url.query or (url.fragment and not re.fullmatch(r"sha256=[0-9a-f]{64}", url.fragment)):
        raise ValueError("wheel source must be an explicit official HTTPS URL")
    return url


class OfficialRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)  # Check before following each hop, not after it.
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_wheel(url):
    validate_url(url)
    return urllib.request.build_opener(OfficialRedirects()).open(url, timeout=60)


def verify_file(path, row):
    if path.is_symlink() or not path.is_file():
        raise ValueError("wheel must be a regular file: " + path.name)
    if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
        raise ValueError("wheel size/SHA256 mismatch: " + path.name)


def fetch(manifest, output, verify_only=False):
    obj, manifest_sha256 = read_manifest(manifest)
    if not output.is_absolute() or output.is_symlink():
        raise ValueError("output directory must be absolute and not a symlink")
    if not output.exists():
        if verify_only:
            raise ValueError("verify-only directory does not exist")
        output.mkdir()  # parent must already exist; no implicit storage selection.
    if not output.is_dir():
        raise ValueError("output is not a directory")
    missing = [row for row in obj["wheels"] if not (output / row["filename"]).exists()]
    if not verify_only and shutil.disk_usage(output).free < sum(row["bytes"] for row in missing) + 1024**3:
        raise ValueError("insufficient free space: wheels plus 1 GiB reserve required")
    reused = downloaded = 0
    for row in obj["wheels"]:
        dest = output / row["filename"]
        if dest.exists() or dest.is_symlink():
            verify_file(dest, row)
            reused += 1
            continue
        if verify_only:
            raise ValueError("missing wheel: " + dest.name)
        part = dest.with_name(dest.name + ".part")
        if part.exists() or part.is_symlink():
            raise FileExistsError("partial transfer exists; inspect it and choose a new directory: " + str(part))
        with part.open("xb") as stream, open_wheel(row["source_url"]) as response:
            validate_url(response.geturl())
            remaining = row["bytes"]
            while block := response.read(min(1024 * 1024, remaining + 1)):
                if len(block) > remaining:
                    raise ValueError("response exceeds manifest size")
                stream.write(block)
                remaining -= len(block)
        verify_file(part, row)
        # Same-directory hard link publishes the verified inode without copying
        # several GiB again. It fails if another process created the destination.
        os.link(part, dest)
        part.unlink()
        downloaded += 1
    return {"passed": True, "platform": obj.get("platform"), "manifest_sha256": manifest_sha256,
            "wheels": len(obj["wheels"]), "bytes": sum(r["bytes"] for r in obj["wheels"]),
            "verified_existing": reused, "downloaded": downloaded, "verify_only": verify_only,
            "packages_installed": False, "gpu_or_quality_assessed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = fetch(args.manifest, args.output_dir, args.verify_only)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
