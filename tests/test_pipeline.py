import json
import tempfile
import unittest
from unittest import mock
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
from pipeline import alignment
from pipeline import renderer
from pipeline import ai
from pipeline import orchestration
from pipeline import precision
from pipeline import visual_quality
from pipeline import output_quality
from pipeline import productivity
from pipeline import dramaturgy
from pipeline import visual_qa
from pipeline import lipsync
from pipeline import catalog_quality
from pipeline import motion
from pipeline import generative


class GenerativePromptTests(unittest.TestCase):
    def test_rap_prompt_places_exact_lyrics_in_lipsync_clause(self):
        lyric = "Tady a teď, brácho, žijeme tady a teď, (tady a teď!)"
        prompt = generative._build_rap_prompt(
            clip_id="rap_03",
            duration=3.25,
            lyric_text=lyric,
            mood_text="cinematic Czech rap, determined, nocturnal, urban",
        )
        self.assertIn('matching the Czech-rapped lyrics: "' + lyric + '"', prompt)
        self.assertIn("wearing a dark olive functional jacket over a distinctive hooded sweatshirt", prompt)
        self.assertIn("duration 3.25 seconds", prompt)
        self.assertNotIn("provided Czech rap line", prompt)

    def test_rap_prompt_sanitizes_nested_double_quotes(self):
        prompt = generative._build_rap_prompt(
            clip_id="rap_01",
            duration=2.50,
            lyric_text='Řekni "teď" a běž',
            mood_text="cinematic Czech rap",
        )
        self.assertIn('matching the Czech-rapped lyrics: "Řekni \'teď\' a běž"', prompt)


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


class MotionTests(unittest.TestCase):
    def test_explicit_glitch_and_whippan_tags_win(self):
        self.assertEqual(motion.motion_style("[GLITCH]", "chorus", 0.8), "glitch")
        self.assertEqual(motion.motion_style("[WHIPPAN]", "verse", 0.4), "whippan")

    def test_downbeat_high_energy_gets_impact(self):
        plan = motion.transition_plan({"section": "chorus", "energy": 0.9, "beat_is_downbeat": True, "duration": 2.0})
        self.assertEqual(plan["style"], "impact")
        self.assertTrue(any("eq=" in item for item in plan["filters"]))

    def test_unknown_motion_is_empty_and_safe(self):
        self.assertEqual(motion.motion_filters("unknown", 1.0), [])


class CatalogQualityTests(unittest.TestCase):
    def test_missing_media_is_reported(self):
        report = catalog_quality.build_catalog_quality({"VID_01": {"group": "VID"}}, Path("/tmp/nonexistent-input"))
        self.assertEqual(report["stats"]["missing_count"], 1)
        self.assertFalse(report["entries"]["VID_01"]["valid"])

    def test_invalid_quality_is_penalized_in_continuity(self):
        metadata = visual_quality.normalize_clip_metadata("bad", {"quality_score": 0.0, "valid": False})
        self.assertLess(visual_quality.continuity_score(metadata), 0.0)


class LipsyncTests(unittest.TestCase):
    def test_word_and_phoneme_manifest_uses_integer_ms(self):
        manifest = lipsync.build_lipsync_manifest([
            {"word": "Ahoj", "start": 1.25, "end": 1.75, "confidence": 0.92},
        ], song_duration=5.0, text_match_score=0.9)
        self.assertEqual(manifest["words"][0]["start_ms"], 1250)
        self.assertEqual(manifest["words"][0]["end_ms"], 1750)
        self.assertGreaterEqual(manifest["stats"]["phoneme_count"], 4)
        self.assertEqual(manifest["stats"]["low_confidence_word_count"], 0)

    def test_lipsync_drift_and_low_confidence(self):
        manifest = lipsync.build_lipsync_manifest([
            {"word": "test", "start": 1.0, "end": 1.5, "confidence": 0.2},
        ])
        report = lipsync.validate_manifest_against_ranges(
            manifest, [{"clip": "rap_01", "start": 0.0, "end": 2.0}], tolerance_ms=100
        )
        self.assertFalse(report["ok"])
        self.assertTrue(any("drift" in error for error in report["errors"]))
        self.assertTrue(any("nízkou confidence" in warning for warning in report["warnings"]))

    def test_czech_ch_is_single_phoneme_unit(self):
        self.assertEqual(lipsync.word_to_phonemes("chata")[0], "ch")


class VisualQATests(unittest.TestCase):
    def test_black_samples_are_errors(self):
        samples = [
            {"time": 0.1, "mean": 0.0, "variance": 0.0, "dark_ratio": 1.0},
            {"time": 1.0, "mean": 0.0, "variance": 0.0, "dark_ratio": 1.0},
        ]
        with mock.patch("pipeline.visual_qa.sample_video_frames", return_value=samples):
            report = visual_qa.audit_visual_quality(Path("clip.mp4"), 2.0, sample_count=2)
        self.assertFalse(report["ok"])
        self.assertIn("černých", report["errors"][0])

    def test_freeze_is_warning_not_error(self):
        samples = [
            {"time": 0.1, "mean": 50.0, "variance": 0.1, "dark_ratio": 0.1},
            {"time": 1.0, "mean": 50.05, "variance": 0.1, "dark_ratio": 0.1},
        ]
        with mock.patch("pipeline.visual_qa.sample_video_frames", return_value=samples):
            report = visual_qa.audit_visual_quality(Path("clip.mp4"), 2.0, sample_count=2)
        self.assertTrue(report["ok"])
        self.assertTrue(report["warnings"])


class DramaturgyTests(unittest.TestCase):
    def test_section_profiles_create_different_visual_energy(self):
        plan = dramaturgy.build_dramaturgy_plan([
            ("intro", 0.0, 8.0, "start"),
            ("chorus", 8.0, 20.0, "refrain"),
        ])
        self.assertEqual(plan[0]["key"], "intro")
        self.assertEqual(plan[1]["key"], "chorus")
        self.assertLess(plan[0]["cut_density"], plan[1]["cut_density"])
        self.assertLess(plan[0]["energy"], plan[1]["energy"])
        self.assertEqual(plan[1]["start_ms"], 8000)

    def test_czech_section_alias_and_time_lookup(self):
        plan = dramaturgy.build_dramaturgy_plan([("refrén", 0.0, 10.0, "")])
        self.assertEqual(plan[0]["key"], "chorus")
        self.assertEqual(dramaturgy.section_at_time(plan, 3.0)["key"], "chorus")
        self.assertEqual(dramaturgy.section_at_time(plan, 20.0)["key"], "unknown")


class ProductivityTests(unittest.TestCase):
    def test_seed_is_persistent_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            first = productivity.ensure_seed(project)
            second = productivity.ensure_seed(project)
        self.assertEqual(first, second)

    def test_preview_report_counts_repeats_and_locks(self):
        timeline = "00:00.000 - 00:02.000 | vid_01 | [LOCK]\n00:02.000 - 00:04.000 | vid_01 | [SPEED=1.2x]"
        report = productivity.build_preview_report(timeline, seed=7, locks={"vid_02": {"reason": "manual"}})
        self.assertEqual(report["segments"], 2)
        self.assertEqual(report["repeated_assets"], {"vid_01": 2})
        self.assertEqual(report["locked_segments"], 1)
        self.assertEqual(report["speed_max"], 1.2)
        self.assertIn("vid_02", report["locked_assets"])

    def test_invalid_timeline_lines_are_ignored(self):
        report = productivity.build_preview_report("not a timeline\n00:00.000 - 00:01.000 | pic_01 | scene")
        self.assertEqual(report["segments"], 1)


class OutputQualityTests(unittest.TestCase):
    def test_final_profile_is_conservative_and_normalized(self):
        profile = output_quality.profile_for("final", "fullhd", 30)
        self.assertEqual(profile.preset, "slow")
        self.assertEqual(profile.pixel_format, "yuv420p")
        self.assertTrue(profile.loudnorm)
        self.assertIn("-r", profile.video_encoder_args)
        self.assertIn("-ar", profile.audio_encoder_args)

    def test_draft_profile_is_faster_without_loudnorm(self):
        profile = output_quality.profile_for("draft", "draft", 24)
        self.assertEqual(profile.preset, "veryfast")
        self.assertFalse(profile.loudnorm)
        self.assertIsNone(output_quality.loudness_filter(profile))

    def test_loudness_audit_rejects_true_peak(self):
        stderr = 'prefix\n{"input_i": "-14.0", "input_tp": "0.5"}\n'
        with mock.patch("pipeline.output_quality.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stderr=stderr)
            report = output_quality.run_loudness_audit(Path("output.mp4"))
        self.assertFalse(report["ok"])
        self.assertIn("clipping", report["errors"][0])


class VisualQualityTests(unittest.TestCase):
    def test_metadata_and_continuity_score(self):
        catalog = {
            "vid_01": {"group": "VID", "obsah": "noční město", "location": "město", "energy": 0.8},
            "vid_02": {"group": "VID", "obsah": "les v mlze", "location": "les", "energy": 0.2},
        }
        ranked = visual_quality.rank_candidates(["vid_01", "vid_02"], catalog, "noční město", previous_id="vid_01")
        self.assertEqual(ranked[0], "vid_01")
        self.assertEqual(visual_quality.normalize_clip_metadata("vid_01", catalog["vid_01"]).location, "město")

    def test_recent_clip_is_penalized(self):
        catalog = {
            "vid_01": {"group": "VID", "obsah": "město", "energy": 0.5},
            "vid_02": {"group": "VID", "obsah": "město", "energy": 0.5},
        }
        ranked = visual_quality.rank_candidates(["vid_01", "vid_02"], catalog, "město", recent_ids=["vid_01"])
        self.assertEqual(ranked[0], "vid_02")

    def test_enriched_beats_mark_downbeats_and_phrases(self):
        beats = visual_quality.enrich_beats([0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertTrue(beats[0]["is_downbeat"])
        self.assertTrue(beats[0]["is_phrase_start"])
        self.assertTrue(beats[4]["is_downbeat"])
        anchor = visual_quality.nearest_sync_point(1.02, beats, prefer_downbeat=True)
        self.assertEqual(anchor["time_ms"], 1000)


class PrecisionTests(unittest.TestCase):
    def test_integer_millisecond_model(self):
        self.assertEqual(precision.seconds_to_ms("1.2345"), 1234)
        self.assertEqual(precision.ms_to_seconds(1235), 1.235)
        self.assertEqual(precision.duration_drift_ms(2.001, 2.0), 1)
        self.assertEqual(precision.duration_drift_ms(2.021, 2.0), 21)

    def test_drift_tolerance(self):
        self.assertIsNone(precision.validate_duration_drift("rap_01", 2.02, 2.0, 20))
        issue = precision.validate_duration_drift("rap_01", 2.021, 2.0, 20)
        self.assertIsNotNone(issue)
        self.assertIn("21 ms", issue.message())

    def test_ffprobe_qa_rejects_invalid_output(self):
        with mock.patch("pipeline.precision.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=1, stderr="invalid media")
            report = precision.ffprobe_media_qa(Path("output.mp4"))
        self.assertFalse(report["ok"])
        self.assertIn("invalid media", report["errors"][0])


class OrchestrationTests(unittest.TestCase):
    def test_none_is_legacy_success(self):
        result = orchestration.execute_step("legacy", lambda: None)
        self.assertTrue(result.ok)

    def test_false_is_failure(self):
        result = orchestration.execute_step("validation", lambda: False)
        self.assertFalse(result.ok)
        self.assertIn("validation", result.errors[0])

    def test_exception_is_captured(self):
        result = orchestration.execute_step("broken", lambda: 1 / 0)
        self.assertFalse(result.ok)
        self.assertIn("ZeroDivisionError", result.errors[0])

    def test_sequence_stops_after_failure(self):
        calls = []
        result = orchestration.execute_sequence([
            ("first", lambda: calls.append("first")),
            ("second", lambda: False),
            ("third", lambda: calls.append("third")),
        ])
        self.assertFalse(result.ok)
        self.assertEqual(calls, ["first"])


class AITests(unittest.TestCase):
    def test_provider_normalization_and_limits(self):
        self.assertEqual(ai.normalize_provider(" GROQ "), "groq")
        self.assertEqual(ai.normalize_provider("unknown"), "local")
        self.assertEqual(ai.finite_positive_int("999", 1, maximum=10), 10)
        self.assertEqual(ai.finite_temperature("9"), 2.0)

    def test_local_plan_uses_stream_call(self):
        calls = []
        result = ai.dispatch_text(
            [], "plan", {"text_ai_provider": "groq"}, "local-model", "groq-model",
            lambda **kwargs: calls.append(("local", kwargs)) or "unused",
            lambda *args, **kwargs: calls.append(("stream", kwargs)) or "plan-text",
            lambda *args, **kwargs: calls.append(("groq", kwargs)) or "unused",
        )
        self.assertEqual(result.text, "plan-text")
        self.assertEqual(calls[0][0], "stream")
        self.assertEqual(result.provider, "local")

    def test_scenario_uses_groq_callback(self):
        calls = []
        result = ai.dispatch_text(
            [], "scenario", {"text_ai_provider": "groq"}, "local-model", "groq-model",
            lambda **kwargs: "unused", lambda **kwargs: "unused",
            lambda *args, **kwargs: calls.append(kwargs) or "scenario-text",
        )
        self.assertEqual(result.text, "scenario-text")
        self.assertEqual(result.model, "groq-model")
        self.assertEqual(result.provider, "groq")
        self.assertEqual(calls[0]["max_tokens"], 3000)


class RendererTests(unittest.TestCase):
    def test_video_commands_differ_for_image_and_video(self):
        image_cmd = renderer.video_segment_command(Path("in.png"), Path("out.mp4"), 2.0, "scale=640:360", True)
        video_cmd = renderer.video_segment_command(Path("in.mp4"), Path("out.mp4"), 2.0, "scale=640:360", False)
        self.assertIn("-loop", image_cmd)
        self.assertNotIn("-loop", video_cmd)

    def test_concat_manifest_and_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = [root / "one.mp4", root / "two's.mp4"]
            manifest = renderer.concat_manifest(parts, root / "concat.txt")
            content = manifest.read_text(encoding="utf-8")
            self.assertIn("file '", content)
            self.assertEqual(renderer.concat_command(manifest, root / "out.mp4")[0], "ffmpeg")

    def test_fade_duration_is_clamped_to_half_video(self):
        command = renderer.fade_command(Path("in.mp4"), Path("out.mp4"), 1.0, 1.5)
        self.assertIn("d=0.5", " ".join(command))


class AlignmentTests(unittest.TestCase):
    def test_speed_for_slot_is_clamped(self):
        self.assertEqual(alignment.speed_for_slot(8, 2, 0.5, 2.0), 2.0)
        self.assertEqual(alignment.speed_for_slot(2, 8, 0.5, 2.0), 0.5)

    def test_distribute_gap_is_contiguous(self):
        slots = alignment.distribute_gap(0, 9, [1, 2, 3])
        self.assertEqual(slots, [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)])

    def test_alignment_ranges_detect_overlap(self):
        errors = alignment.validate_alignment_ranges([(0, 2), (1.5, 3)])
        self.assertTrue(any("překrývá" in error for error in errors))


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


class SocialExperimentObservabilityTests(unittest.TestCase):
    def test_thumbnail_image_metrics_reject_empty_and_score_real_image(self):
        from PIL import Image
        from pipeline import social, experiments, observability
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "thumb.png"
            Image.new("RGB", (64, 64), (120, 80, 40)).save(path)
            metrics = social.analyze_thumbnail_image(path)
            self.assertTrue(metrics["valid"])
            self.assertIn("sharpness", metrics)
            self.assertGreaterEqual(social.thumbnail_score(metrics), 0.0)
            self.assertFalse(social.analyze_thumbnail_image(Path(temp) / "missing.png")["valid"])

    def test_variant_manifest_is_deterministic_and_applies_overrides(self):
        from pipeline import experiments
        variants = experiments.build_variants(42)
        self.assertEqual([v.seed for v in variants], [v.seed for v in experiments.build_variants(42)])
        plans = experiments.build_variant_plans([{"cut_density": 0.8}], variants)
        self.assertGreater(plans["faster_cuts"][0]["cut_density"], plans["control"][0]["cut_density"])
        self.assertEqual(plans["cleaner_motion"][0]["motion_intensity"], 0.75)

    def test_empty_qa_is_unknown_and_failure_is_registered(self):
        from pipeline import observability
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            self.assertEqual(observability.build_qa_summary(project)["status"], "UNKNOWN")
            observability.append_render_failure(project, mode="final", resolution="fullhd", error="render failed")
            records = observability.read_render_registry(project)
            self.assertEqual(records[0]["status"], "failed")
            self.assertEqual(observability.build_qa_summary(project)["status"], "FAIL")


class ABWorkflowTests(unittest.TestCase):
    def test_ab_comparison_prefers_passing_variant(self):
        from pipeline.experiments import compare_variant_qa
        result = compare_variant_qa({
            "control": {"ok": False, "errors": ["black frame"], "warnings": []},
            "faster_cuts": {"ok": True, "errors": [], "warnings": ["minor"]},
            "cleaner_motion": {"ok": True, "errors": [], "warnings": []},
        })
        self.assertEqual(result["recommended"], "cleaner_motion")

    def test_ab_workflow_isolates_outputs_and_writes_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            pipeline = pro_pipeline.TemagenPipeline(project)
            outputs = {}
            def fake_render(**kwargs):
                variant = kwargs["variant_name"]
                target = kwargs["output_dir"] / f"{variant}.mp4"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"video")
                target.with_suffix(target.suffix + ".qa.json").write_text('{"ok": true, "errors": [], "warnings": []}')
                outputs[variant] = target
                return target
            with patch.object(pipeline, "render_video", side_effect=fake_render):
                result = pipeline.run_ab_render_workflow(base_seed=11)
            self.assertEqual(set(result["outputs"]), {"control", "faster_cuts", "cleaner_motion"})
            self.assertEqual(result["recommended"], "cleaner_motion")
            self.assertTrue((project / "EDIT_PROJECT" / "ab_comparison.json").exists())
            self.assertEqual(len(outputs), 3)


class RapQualityTests(unittest.TestCase):
    def test_phoneme_locked_qa_reports_plosive_drift_and_rolls(self):
        from pipeline import rap_quality
        manifest = {"song_duration_ms": 5000, "phonemes": [{"phoneme": "p", "start_ms": 1000, "end_ms": 1040}, {"phoneme": "a", "start_ms": 1040, "end_ms": 1120}]}
        report = rap_quality.phoneme_locked_qa(manifest, [{"clip": "rap_01", "start_ms": 1000, "end_ms": 1120}], tolerance_ms=35)
        self.assertTrue(report["ok"])
        self.assertEqual(report["pre_roll_ms"], 100)
        self.assertIn("max_plosive_drift_ms", report)

    def test_fallback_and_local_timewarp_are_bounded(self):
        from pipeline import rap_quality
        candidate = rap_quality.choose_fallback_candidate([{"id": "a", "quality_score": 0.9}, {"id": "b", "quality_score": 0.7}], used_ids={"a"})
        self.assertEqual(candidate["id"], "b")
        plan = rap_quality.local_timewarp_plan(0.5, 2.0)
        self.assertGreaterEqual(plan["core_speed"], 0.85)
        self.assertLessEqual(plan["core_speed"], 1.18)

    def test_rap_continuity_and_summary(self):
        from pipeline import rap_quality
        score = rap_quality.rap_continuity_score({"face_scale": 1.0, "mouth_x": 0.5}, {"face_scale": 2.0, "mouth_x": 0.5})
        self.assertLess(score, 1.0)
        summary = rap_quality.build_rap_qa_summary([{"valid": True}], {"ok": True}, [score])
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["clip_count"], 1)


class CharacterMotionTests(unittest.TestCase):
    def test_beak_motion_is_relative_to_root_and_phoneme_aligned(self):
        from pipeline import character_motion
        observations = [
            {"time_ms": 0, "visible": True, "tip": [0.5, 0.4], "root": [0.4, 0.4], "aspect_ratio": 2.0},
            {"time_ms": 100, "visible": True, "tip": [0.6, 0.4], "root": [0.4, 0.4], "aspect_ratio": 2.0},
        ]
        motion = character_motion.track_beak_motion(observations)
        self.assertGreater(motion["motion_energy"][0]["motion_energy"], 0.0)
        alignment = character_motion.align_beak_motion_to_phonemes(motion, [{"phoneme": "p", "start_ms": 100}])
        self.assertTrue(alignment["ok"])

    def test_beak_integrity_flags_low_visibility_and_builds_report(self):
        from pipeline import character_motion
        motion = {"visible_ratio": 0.2, "max_geometry_jump": 0.3, "motion_peaks": []}
        integrity = character_motion.audit_beak_integrity(motion)
        report = character_motion.build_character_lipsync_qa(motion, {"ok": True}, integrity)
        self.assertFalse(integrity["ok"])
        self.assertEqual(report["status"], "FAIL")


class GenerativePackageTests(unittest.TestCase):
    def test_new_song_package_selects_limited_rap_and_writes_prompts(self):
        from pipeline.generative import build_generation_package

        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            (project / "INPUT").mkdir()
            (project / "EDIT_PROJECT").mkdir()
            (project / "INPUT" / "lyrics.txt").write_text(
                "Tohle je signál z budoucnosti\n"
                "Míň stresu víc vaty připravenej na ten fame\n"
                "Každá ta prohra v prachu byla lístek na vrchol\n"
                "Vidíme se na vrcholu signál ukončen\n",
                encoding="utf-8",
            )
            package = build_generation_package(project, max_rap_passages=3)
            self.assertEqual(package["character_type"], "masked_bird_stork_rapper")
            self.assertEqual(len(package["rap_passages"]), 3)
            self.assertTrue(all(2.5 <= item["duration"] <= 6.0 for item in package["rap_passages"]))
            rap_clips = [item for item in package["clips"] if item["type"] == "rap_lipsync"]
            broll_clips = [item for item in package["clips"] if item["type"] == "broll"]
            self.assertEqual(len(rap_clips), 3)
            self.assertEqual(len(broll_clips), 30)
            self.assertEqual([item["clip_id"] for item in broll_clips], [f"vid_{index:02d}" for index in range(1, 31)])
            for item in rap_clips:
                self.assertIn(f'matching the Czech-rapped lyrics: "{item["text"]}"', item["prompt"])
            self.assertTrue((project / "EDIT_PROJECT" / "generation_manifest.json").exists())
            self.assertTrue((project / "Prompts" / "scenario.txt").exists())
            self.assertTrue((project / "Prompts" / "generation_prompts.md").exists())


class RapTranscriptionGuardTests(unittest.TestCase):
    def test_completed_rap_transcription_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            pipeline = pro_pipeline.TemagenPipeline(project)
            (project / "gen_rap").mkdir(parents=True, exist_ok=True)
            (project / "gen_rap" / "rap_01.mp4").write_bytes(b"placeholder")
            (pipeline.edit_dir / "rap_alignment.json").write_text(
                '{"rap_01": {"transcript_raw": "test phrase", '
                '"transcript_fixed": "test phrase", '
                '"words_raw": [{"word": "test", "start": 0.0, "end": 0.2}], '
                '"transcript_empty": false}}',
                encoding="utf-8",
            )
            with mock.patch.object(
                pipeline, "_groq_ready", side_effect=AssertionError("provider check should be skipped")
            ):
                pipeline.transcribe_rap_clips()


class LipsyncExportTests(unittest.TestCase):
    def test_lipsync_export_falls_back_to_text_anchored_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            pipeline = pro_pipeline.TemagenPipeline(project)
            pipeline.timeline_file.parent.mkdir(parents=True, exist_ok=True)
            pipeline.timeline_file.write_text(
                "00:00.000 - 00:02.000 | vid_01 | broll\n", encoding="utf-8"
            )
            audio = project / "song.mp3"
            audio.write_bytes(b"audio")
            (pipeline.edit_dir / "rap_alignment.json").write_text(
                '{"rap_01": {"transcript_fixed": "Tohle je test", '
                '"song_match": {"song_start": 12.5, "song_end": 15.2}}}',
                encoding="utf-8",
            )

            def fake_ffmpeg(command):
                Path(command[-1]).write_bytes(b"wav")
                return True

            with mock.patch.object(pipeline, "validate_transcription_integrity", return_value=True), \
                 mock.patch.object(pipeline, "find_audio", return_value=audio), \
                 mock.patch("pro_pipeline.run_ffmpeg", side_effect=fake_ffmpeg):
                self.assertTrue(pipeline.export_lipsync_audio_segments())
            manifest = json.loads(
                (project / "LIPSYNC_AUDIO" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["segments"][0]["source"], "rap_alignment.song_match")
            self.assertEqual(manifest["segments"][0]["start"], 12.5)
            self.assertEqual(manifest["segments"][0]["end"], 15.2)


class PartialRapTranscriptionTests(unittest.TestCase):
    def test_partial_alignment_transcribes_only_missing_clips(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            pipeline = pro_pipeline.TemagenPipeline(project)
            rap_dir = project / "gen_rap"
            rap_dir.mkdir(parents=True, exist_ok=True)
            (rap_dir / "rap_01.mp4").write_bytes(b"one")
            (rap_dir / "rap_02.mp4").write_bytes(b"two")
            (pipeline.edit_dir / "rap_alignment.json").write_text(
                '{"rap_01": {"transcript_raw": "already done", '
                '"transcript_fixed": "already done", '
                '"words_raw": [{"word": "already", "start": 0.0, "end": 0.2}], '
                '"transcript_empty": false}}',
                encoding="utf-8",
            )
            pipeline._transcribe_media_json = lambda source, whisper, tmp: {
                "segments": [{"words": [{"word": "missing", "start": 0.0, "end": 0.2}]}]
            }
            pipeline._whisper_segments_to_words = lambda data: [
                {"word": "missing", "start": 0.0, "end": 0.2}
            ]
            with mock.patch.object(
                pipeline, "load_settings", return_value={"transcription_provider": "groq"}
            ), mock.patch.object(pipeline, "_groq_ready", return_value=True), mock.patch.object(
                pipeline, "_load_lyrics_text", return_value="already done missing"
            ), mock.patch.object(pipeline, "_load_song_segments", return_value=[]), mock.patch.object(
                pipeline, "_load_timeline_rap_ranges", return_value={}
            ), mock.patch.object(
                pipeline, "_best_lyrics_window_scored", return_value=("missing", 1.0)
            ), mock.patch.object(
                pipeline, "_align_words_to_lyrics", side_effect=lambda words, text: words
            ), mock.patch.object(
                pipeline,
                "_best_song_match",
                return_value={"song_start": 3.0, "song_end": 4.0, "score": 1.0},
            ), mock.patch.object(pro_pipeline, "probe_duration", return_value=1.0):
                pipeline.transcribe_rap_clips()
            saved = json.loads(
                (pipeline.edit_dir / "rap_alignment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(saved), {"rap_01", "rap_02"})
            self.assertEqual(saved["rap_01"]["transcript_raw"], "already done")
            self.assertEqual(saved["rap_02"]["transcript_raw"], "missing")
