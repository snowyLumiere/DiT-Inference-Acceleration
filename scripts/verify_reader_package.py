#!/usr/bin/env python3
"""Verify every file in an extracted reader package using the standard library.

Usage: python3 -B scripts/verify_reader_package.py [package_directory]
The manifest detects accidental changes; it is not a digital signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


MANIFEST = 'READER_MANIFEST.json'
SUMS = 'SHA256SUMS'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts or str(path) != relative:
        raise ValueError('unsafe package path: ' + relative)
    target = root.joinpath(*path.parts)
    if target.is_symlink() or target.resolve() != target.absolute():
        raise ValueError('symlink in package path: ' + relative)
    return target


def verify(root):
    root = root.resolve(strict=True)
    manifest_path = safe_path(root, MANIFEST)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['schema'] != 'dit.reader-package.v1':
        raise ValueError('wrong manifest schema')
    expected = {MANIFEST, SUMS}
    for row in manifest['files']:
        name = row['path']
        if name in expected:
            raise ValueError('duplicate or reserved manifest entry')
        expected.add(name)
        data = safe_path(root, name).read_bytes()
        if len(data) != row['bytes'] or digest(data) != row['sha256']:
            raise ValueError('file mismatch: ' + name)
    actual = {p.relative_to(root).as_posix()
              for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if actual != expected:
        raise ValueError('unexpected or missing package files')
    listed = {}
    for line in safe_path(root, SUMS).read_text(encoding='utf-8').splitlines():
        checksum, name = line.split('  ', 1)
        if name in listed:
            raise ValueError('duplicate checksum entry')
        listed[name] = checksum
    if set(listed) != expected - {SUMS}:
        raise ValueError('checksum coverage mismatch')
    for name, checksum in listed.items():
        if digest(safe_path(root, name).read_bytes()) != checksum:
            raise ValueError('checksum mismatch: ' + name)
    return {'passed': True, 'files': len(actual),
            'manifest_sha256': digest(manifest_path.read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.directory), indent=2))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f'Package verification failed: {exc}\n')


if __name__ == '__main__':
    main()
