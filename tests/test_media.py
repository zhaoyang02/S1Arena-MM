import numpy as np
import pytest

from conftest import make_mp4, make_png, make_video, make_wav
from jevarena_mm import media

pytest.importorskip("av")
pytest.importorskip("PIL")


def test_make_ref_hashes_and_probes(workspace):
    make_video(workspace / "media" / "toy" / "clip.mkv", seconds=4, fps=10)
    ref = media.make_ref("video", "toy/clip.mkv", window=(1.0, 3.0))
    assert ref.sha256 == media.sha256_file(workspace / "media" / "toy" / "clip.mkv")
    assert ref.info["fps"] == 10 and ref.info["width"] == 32 and ref.info["has_audio"] is False
    assert media.check(ref) is None
    (workspace / "media" / "toy" / "clip.mkv").write_bytes(b"changed")
    assert media.check(ref) == "sha256 mismatch"
    (workspace / "media" / "toy" / "clip.mkv").unlink()
    assert media.check(ref) == "missing"


def test_frames_follow_the_window_and_the_adapters_policy(workspace):
    make_video(workspace / "media" / "toy" / "clip.mkv", seconds=4, fps=10)
    ref = media.make_ref("video", "toy/clip.mkv", window=(1.0, 3.0))

    every = media.frames(ref)
    assert len(every) == 20 and every[0].t == pytest.approx(1.0) and every[-1].t == pytest.approx(2.9)
    assert every[0].array.shape == (32, 32, 3) and every[0].array.dtype == np.uint8

    two_fps = media.frames(ref, fps=2)
    assert [f.t for f in two_fps] == pytest.approx([1.0, 1.5, 2.0, 2.5])
    assert int(two_fps[1].array.mean()) == pytest.approx(8 * 15, abs=3)  # frame 15 of the file

    capped = media.frames(ref, fps=10, max_frames=5)
    assert len(capped) == 5 and capped[0].t == pytest.approx(1.0) and capped[-1].t == pytest.approx(2.9)

    centres = media.frames(ref, max_frames=4)
    assert [f.t for f in centres] == pytest.approx([1.2, 1.7, 2.2, 2.7])

    nearest = media.frames(ref, timestamps=[1.04, 2.96, 9.0])
    assert [f.t for f in nearest] == pytest.approx([1.0, 2.9])  # 2.96 and 9.0 both map to the last frame in the window

    dense = media.frames(ref, fps=40)
    assert len(dense) == 20  # never the same frame twice


def test_whole_file_without_window(workspace):
    make_video(workspace / "media" / "toy" / "clip.mkv", seconds=2, fps=10)
    ref = media.make_ref("video", "toy/clip.mkv")
    assert len(media.frames(ref, fps=1)) == 2


def test_audio_window_and_resampling(workspace):
    make_wav(workspace / "media" / "toy" / "tone.wav", seconds=2, rate=8000)
    ref = media.make_ref("audio", "toy/tone.wav", window=(0.5, 1.5))
    assert ref.info["sample_rate"] == 8000 and ref.info["channels"] == 1
    samples, rate = media.audio(ref, sample_rate=16000)
    assert rate == 16000 and samples.dtype == np.float32 and samples.ndim == 1
    assert abs(len(samples) - 16000) <= 64
    assert 0.3 < float(np.abs(samples).max()) < 0.6


def test_images_from_files_and_from_memory(workspace):
    make_png(workspace / "media" / "toy" / "a.png", size=16, value=200)
    ref = media.make_ref("image", "toy/a.png")
    assert ref.info == {"width": 16, "height": 16}
    assert media.image(ref).getpixel((0, 0)) == (200, 200, 200)

    live = media.inline_image(np.zeros((8, 12, 3), dtype=np.uint8))
    assert live.uri.startswith("mem:") and live.info == {"width": 12, "height": 8}
    assert media.image(live).size == (12, 8)
    with pytest.raises(ValueError):
        media.path_of(live)


def test_marks_are_drawn_on_the_marked_frames(workspace):
    import json

    from jevarena_mm.schema import MediaRef

    make_video(workspace / "media" / "toy" / "clip.mkv", seconds=2, fps=10, size=32)
    marks_path = workspace / "media" / "marks" / "clip.json"
    marks_path.parent.mkdir(parents=True)
    marks_path.write_text(json.dumps({"boxes": [[5, 4, 4, 20, 20]]}))
    plain = media.make_ref("video", "toy/clip.mkv", window=(0.4, 0.7))
    marked = MediaRef(**{**plain.model_dump(), "marks": {"uri": "marks/clip.json", "sha256": media.sha256_file(marks_path)}})
    assert media.check(marked) is None
    frames = {round(f.t, 1): f.array for f in media.frames(marked)}
    assert sorted(frames) == [0.4, 0.5, 0.6]
    assert tuple(frames[0.5][4, 10]) == (255, 0, 0) and tuple(frames[0.5][10, 10]) != (255, 0, 0)  # frame 5: outline only
    assert tuple(frames[0.4][4, 10]) != (255, 0, 0)  # frame 4 has no box
    marks_path.write_text("{}")
    assert media.check(marked) == "marks sha256 mismatch"


def test_a_window_rounded_past_the_first_frame_still_includes_it(workspace):
    # Windows are stored to the microsecond: a start of 0.083008 for a first frame at 0.0830078125 must keep that frame.
    make_video(workspace / "media" / "toy" / "clip.mkv", seconds=1, fps=10)
    ref = media.make_ref("video", "toy/clip.mkv", window=(0.1000004, 0.5))
    assert [round(f.t, 1) for f in media.frames(ref)] == [0.1, 0.2, 0.3, 0.4]


def test_sparse_targets_seek_to_the_frames_a_full_decode_finds(workspace):
    make_mp4(workspace / "media" / "toy" / "long.mp4", seconds=60, fps=10, gop=25)
    ref = media.make_ref("video", "toy/long.mp4", window=(3.0, 57.0))
    stamps = [3.0 + 54.0 * k / 8 + 0.04 for k in range(8)]  # 6.75 s apart, between frames
    fast = media._seek_select(ref, None, None, stamps)
    full = media._stream_select(ref, None, None, stamps)
    assert fast is not None and [f.t for f in fast] == [f.t for f in full]
    assert all(np.array_equal(a.array, b.array) for a, b in zip(fast, full))
    assert [f.t for f in media.frames(ref, timestamps=stamps)] == [f.t for f in full]
    assert media._seek_select(ref, None, None, [3.0 + 0.5 * k for k in range(40)]) is None  # dense: whole stream
    make_video(workspace / "media" / "toy" / "long.mkv", seconds=60, fps=10)
    assert media._seek_select(media.make_ref("video", "toy/long.mkv", window=(3.0, 57.0)), None, None, stamps) is None


def test_sparse_targets_of_a_marked_video_get_their_marks(workspace):
    import json

    from jevarena_mm.schema import MediaRef

    make_mp4(workspace / "media" / "toy" / "long.mp4", seconds=60, fps=10, gop=25)
    stamps = [5.0, 20.0, 35.1, 50.2]
    marks_path = workspace / "media" / "marks" / "long.json"
    marks_path.parent.mkdir(parents=True)
    marks_path.write_text(json.dumps({"boxes": [[round(t * 10), 4, 4, 20, 20] for t in stamps[1:3]]}))
    plain = media.make_ref("video", "toy/long.mp4", window=(0.0, 60.0))
    marked = MediaRef(**{**plain.model_dump(), "marks": {"uri": "marks/long.json", "sha256": media.sha256_file(marks_path)}})
    fast = media._seek_select(marked, None, None, stamps)
    full = media._stream_select(marked, None, None, stamps)
    assert fast is not None and [f.t for f in fast] == [f.t for f in full]
    assert all(np.array_equal(a.array, b.array) for a, b in zip(fast, full))
    assert [tuple(f.array[4, 10]) == (255, 0, 0) for f in fast] == [False, True, True, False]


def test_open_media_reads_metadata_leniently(monkeypatch):
    """av.open is asked to replace undecodable tag bytes; a PyAV without that argument (19+) is opened plainly."""
    import av

    calls = []

    def fake_open(path, **kw):
        calls.append(kw)
        if "metadata_errors" in kw and len(calls) == 1 and path == "new.mp4":
            raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
        return path

    monkeypatch.setattr(av, "open", fake_open)
    assert media.open_media("old.mp4") == "old.mp4" and calls == [{"metadata_errors": "replace"}]
    calls.clear()
    assert media.open_media("new.mp4") == "new.mp4" and calls == [{"metadata_errors": "replace"}, {}]
