import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fetch_rtx_lab_wheels as wheels


class Response(io.BytesIO):
    def geturl(self):
        return "https://files.pythonhosted.org/packages/test/demo-1-py3-none-any.whl"


class WheelhouseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.output = self.root / "wheels"
        self.payload = b"test wheel payload"
        self.row = {"filename": "demo-1-py3-none-any.whl", "bytes": len(self.payload),
                    "sha256": hashlib.sha256(self.payload).hexdigest(),
                    "source_url": "https://files.pythonhosted.org/packages/test/demo-1-py3-none-any.whl"}
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self, rows=None):
        self.manifest.write_text(json.dumps({"platform": "test", "wheels": rows or [self.row]}))

    def test_download_then_offline_verify(self):
        with patch.object(wheels, "open_wheel", return_value=Response(self.payload)):
            result = wheels.fetch(self.manifest, self.output)
        self.assertEqual(result["downloaded"], 1)
        self.assertFalse((self.output / (self.row["filename"] + ".part")).exists())
        with patch.object(wheels, "open_wheel", side_effect=AssertionError("network forbidden")):
            result = wheels.fetch(self.manifest, self.output, verify_only=True)
        self.assertEqual(result["verified_existing"], 1)

    def test_tampered_existing_file_is_preserved(self):
        self.output.mkdir()
        dest = self.output / self.row["filename"]
        dest.write_bytes(b"x" * len(self.payload))
        with patch.object(wheels, "open_wheel", side_effect=AssertionError("network forbidden")):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                wheels.fetch(self.manifest, self.output)
        self.assertEqual(dest.read_bytes(), b"x" * len(self.payload))

    def test_bad_download_keeps_partial_without_publishing(self):
        with patch.object(wheels, "open_wheel", return_value=Response(b"x" * len(self.payload))):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                wheels.fetch(self.manifest, self.output)
        self.assertFalse((self.output / self.row["filename"]).exists())
        self.assertTrue((self.output / (self.row["filename"] + ".part")).exists())

    def test_oversized_response_is_not_published(self):
        with patch.object(wheels, "open_wheel", return_value=Response(self.payload + b"x")):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                wheels.fetch(self.manifest, self.output)
        self.assertFalse((self.output / self.row["filename"]).exists())

    def test_partial_is_not_overwritten(self):
        self.output.mkdir()
        part = self.output / (self.row["filename"] + ".part")
        part.write_bytes(b"retained")
        with self.assertRaises(FileExistsError):
            wheels.fetch(self.manifest, self.output)
        self.assertEqual(part.read_bytes(), b"retained")

    def test_symlink_is_rejected(self):
        self.output.mkdir()
        (self.output / self.row["filename"]).symlink_to(self.manifest)
        with self.assertRaisesRegex(ValueError, "regular file"):
            wheels.fetch(self.manifest, self.output, verify_only=True)

    def test_manifest_traversal_is_rejected_before_writing(self):
        self.row["filename"] = "../escape.whl"
        self.save()
        with self.assertRaisesRegex(ValueError, "filename"):
            wheels.fetch(self.manifest, self.output)
        self.assertFalse(self.output.exists())

    def test_duplicate_and_nonofficial_source_are_rejected(self):
        self.save([self.row, self.row])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            wheels.read_manifest(self.manifest)
        self.row["source_url"] = "https://unknown.invalid/" + self.row["filename"]
        self.save()
        with self.assertRaisesRegex(ValueError, "official"):
            wheels.read_manifest(self.manifest)

    def test_missing_wheel_in_verify_mode_does_not_download(self):
        self.output.mkdir()
        with patch.object(wheels, "open_wheel", side_effect=AssertionError("network forbidden")):
            with self.assertRaisesRegex(ValueError, "missing wheel"):
                wheels.fetch(self.manifest, self.output, verify_only=True)

    def test_destination_created_during_download_is_preserved(self):
        def concurrent(_url):
            (self.output / self.row["filename"]).write_bytes(b"other writer")
            return Response(self.payload)
        with patch.object(wheels, "open_wheel", side_effect=concurrent):
            with self.assertRaises(FileExistsError):
                wheels.fetch(self.manifest, self.output)
        self.assertEqual((self.output / self.row["filename"]).read_bytes(), b"other writer")

    def test_receipt_hash_is_bound_to_the_read_snapshot(self):
        original = wheels.digest(self.manifest)
        def changed(_url):
            self.row["sha256"] = "f" * 64
            self.save()
            return Response(self.payload)
        with patch.object(wheels, "open_wheel", side_effect=changed):
            result = wheels.fetch(self.manifest, self.output)
        self.assertEqual(result["manifest_sha256"], original)
        self.assertNotEqual(result["manifest_sha256"], wheels.digest(self.manifest))

    def test_redirect_is_checked_before_following(self):
        handler = wheels.OfficialRedirects()
        req = wheels.urllib.request.Request(self.row["source_url"])
        for target in ("https://unknown.invalid/a.whl", "http://files.pythonhosted.org/a.whl",
                       "https://files.pythonhosted.org:444/a.whl", "https://name@files.pythonhosted.org/a.whl"):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "official"):
                handler.redirect_request(req, None, 302, "Found", {}, target)

    def test_redirect_between_official_hosts_is_allowed(self):
        handler = wheels.OfficialRedirects()
        req = wheels.urllib.request.Request(self.row["source_url"])
        target = "https://download.pytorch.org/whl/demo-1-py3-none-any.whl"
        redirected = handler.redirect_request(req, None, 302, "Found", {}, target)
        self.assertEqual(redirected.full_url, target)

    def test_official_index_hash_fragment_matches_manifest(self):
        self.row["source_url"] += "#sha256=" + self.row["sha256"]
        self.save()
        obj, _digest = wheels.read_manifest(self.manifest)
        self.assertEqual(obj["wheels"][0]["sha256"], self.row["sha256"])

    def test_conflicting_hash_fragment_is_rejected(self):
        self.row["source_url"] += "#sha256=" + "f" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "disagree"):
            wheels.read_manifest(self.manifest)


if __name__ == "__main__":
    unittest.main()
