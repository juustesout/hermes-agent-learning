"""Feature-import wordt nooit stil verborgen; context_features wordt echt gevuld."""
import builtins
import json
import os
import tempfile
import unittest
from unittest import mock

import features
import hermes_plugin
import opslog


_real_import = builtins.__import__


def _no_features_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Simuleert een runtime waarin features.py niet importeerbaar is,
    # zowel relatief (from .. import features) als absoluut (import features).
    if name == "features" or (fromlist and "features" in fromlist):
        raise ImportError("simulated: features not importable")
    return _real_import(name, globals, locals, fromlist, level)


class _StoreMixin:
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._patches = [
            mock.patch.object(opslog, "STORE_DIR", self._tmp),
            mock.patch.object(opslog, "OPSLOG_PATH", os.path.join(self._tmp, "operations.jsonl")),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()


class FeatureImportFailure(_StoreMixin, unittest.TestCase):
    def test_import_failure_is_recorded_as_health_error(self):
        with mock.patch.object(builtins, "__import__", _no_features_import):
            payload = hermes_plugin.build_operation(
                tool_name="web_extract",
                args={"url": "https://example.com/page"},
                status="ok",
                duration_ms=10,
            )
        self.assertIsNotNone(payload)
        self.assertIsNone(payload["context_features"])
        self.assertEqual(payload["metadata"]["import_source"], "none")
        health = opslog.health()
        self.assertFalse(health["ok"])
        self.assertIn("features import failed", health["last_error"])
        self.assertIn("relative", health["last_error"])
        self.assertIn("absolute", health["last_error"])

    def test_import_failure_is_not_silent_even_without_operation_write(self):
        with mock.patch.object(builtins, "__import__", _no_features_import):
            hermes_plugin.build_operation(tool_name="web_extract", args={"url": "https://a.nl"})
        with open(os.path.join(self._tmp, "opslog-health.json"), encoding="utf-8") as f:
            state = json.load(f)
        self.assertFalse(state["ok"])
        self.assertTrue(state["last_error"])


class FeatureImportSuccess(_StoreMixin, unittest.TestCase):
    def test_context_features_are_filled_with_pre_call_names(self):
        payload = hermes_plugin.build_operation(
            tool_name="web_extract",
            args={"url": "https://www.example.nl/a/b?x=1"},
            status="ok",
            duration_ms=10,
        )
        self.assertIn(payload["metadata"]["import_source"], {"relative", "absolute"})
        feats = payload["context_features"]
        self.assertIsInstance(feats, dict)
        self.assertEqual(set(feats), set(features.PRE_CALL_FEATURE_NAMES))
        self.assertGreater(feats["url_path_depth"], 0.0)
        self.assertEqual(feats["url_has_query"], 1.0)

    def test_no_health_error_when_import_works(self):
        hermes_plugin.build_operation(tool_name="web_extract", args={"url": "https://a.nl/x"})
        health = opslog.health()
        self.assertTrue(health["ok"])
        self.assertNotIn("features", health["last_error"] or "")


if __name__ == "__main__":
    unittest.main()
