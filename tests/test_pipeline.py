import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pro_pipeline
from klipy import KlipyPipeline
from pipeline.models import TimelineEntry
from pipeline.validation import validate_timeline
from pipeline.commands import MAIN_COMMAND_ALIASES, KLIPY_COMMAND_ALIASES
from pipeline.runtime import redact_secrets, project_logger
from pipeline import parsers
from pipeline import media


class ParserTests(unittest.TestCase):
    def test_main_module_uses_shared_parser_implementation(self):
        self.assertIs(pro_pipeline.parse_timecode, parsers.parse_timecode)
        self.assertIs(pro_pipeline.extract_sections, parsers.extract_sections)
        self.assertIs(pro_pipeline.parse_timeline_entries, parsers.parse_timeline_entries)

    def test_timecode_rejects_non_finite_and_negative_values(self):
        self.assertEqual(pro_pipeline.parse_timecode("00:42.50"), 42.5)
        for value in ("-1", "nan", "inf", "1:80.0", "1:2:3:4"):
            with self.assertRaises(ValueError):
                pro_pipeline.parse_timecode(value)

    def test_unknown_section_does_not_contaminate_previous_section(self):
        text = "### SHOT_ORDER\na\n### UNKNOWN\nb\n### EFFECTS\nc"
        sections = pro_pipeline.extract_sections(text)
        self.assertEqual(sections["SHOT_ORDER"], "a")
        self.assertEqual(sections["EFFECTS"], "c")
        self.assertNotIn("UNKNOWN", sections)

    def test_timeline_parser_and_validation(self):
        entries, warnings = pro_pipeline.parse_timeline_entries(
            "00:00.00 - 00:02.00 | vid_01 | intro\n"
            "00:01.50 - 00:03.00 | vid_02 | overlap"
        )
        self.assertEqual(warnings, [])
        errors = validate_timeline(entries, song_duration=3.0)
        self.assertTrue(any("překrytí" in error for error in errors))


class MediaTests(unittest.TestCase):
    def test_stringify_command_accepts_paths(self):
        self.assertEqual(media.stringify_command(["ffmpeg", Path("out.mp4")]), ["ffmpeg", "out.mp4"])

    @patch("pipeline.media.subprocess.check_output", return_value="12.5\n")
    def test_probe_duration_returns_finite_value(self, _check_output):
        self.assertEqual(media.probe_duration(Path("song.mp3")), 12.5)

    @patch("pipeline.media.subprocess.check_output", return_value="nan\n")
    def test_probe_duration_rejects_nan(self, _check_output):
        self.assertEqual(media.probe_duration(Path("song.mp3")), 0.0)


class RuntimeAndCliTests(unittest.TestCase):
    def test_shared_aliases_are_resolved(self):
        self.assertEqual(MAIN_COMMAND_ALIASES["vse"], "all")
        self.assertEqual(MAIN_COMMAND_ALIASES["zarovnej-rap"], "align-rap")
        self.assertEqual(KLIPY_COMMAND_ALIASES["order"], "2")

    def test_secrets_are_redacted(self):
        value = redact_secrets("api_key=secret-token authorization: Bearer xyz")
        self.assertNotIn("secret-token", value)
        self.assertNotIn("Bearer xyz", value)

    def test_project_logger_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger_a = project_logger(Path(tmp))
            logger_b = project_logger(Path(tmp))
            self.assertIs(logger_a, logger_b)
            logger_a.info("test")
            self.assertTrue((Path(tmp) / "EDIT_PROJECT" / "pipeline.log").exists())


class PipelineTests(unittest.TestCase):
    def test_settings_are_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = pro_pipeline.TemagenPipeline(Path(tmp))
            settings = pipeline._normalize_settings({"speed_min": "bad", "speed_max": -2})
            self.assertEqual(settings["speed_min"], 0.5)
            self.assertEqual(settings["speed_max"], 2.0)

    def test_klipy_naive_timeline_marks_estimates(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = KlipyPipeline(Path(tmp))
            lines, _ = pipeline._build_naive_timeline_from_order(
                ["rap_01", "vid_01"],
                {
                    "rap_01": {"duration_sec": 4.0, "group": "RAP", "text": "Ahoj"},
                    "vid_01": {"duration_sec": 0.0, "group": "VID", "obsah": "Ulice"},
                },
            )
            self.assertIn("[ODHAD DÉLKY]", lines[-1])


if __name__ == "__main__":
    unittest.main()
