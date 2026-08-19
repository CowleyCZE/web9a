from pathlib import Path

from pipeline.broll import fit_mode, is_loopable_broll, validate_source_for_slot
from pipeline.renderer import video_segment_command


def test_short_vid_is_loopable():
    source = Path("vid_01.mp4")
    assert is_loopable_broll(source)
    assert fit_mode(source, 2.0, 8.0) == "loop"
    assert validate_source_for_slot(source, 2.0, 8.0) == (True, "loop")


def test_long_vid_is_trimmed_at_render():
    source = Path("vid_01.mp4")
    assert fit_mode(source, 12.0, 8.0) == "trim"
    cmd = video_segment_command(source, Path("out.mp4"), 8.0, "null")
    assert "-stream_loop" in cmd
    assert "-1" in cmd
    assert "-t" in cmd
    assert "8.000" in cmd


def test_short_non_loopable_media_is_mismatch():
    source = Path("rap_01.mp4")
    assert fit_mode(source, 2.0, 8.0) == "mismatch"
    ok, reason = validate_source_for_slot(source, 2.0, 8.0)
    assert not ok
    assert reason == "short_non_loopable_source"
