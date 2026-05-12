"""Postprocess tests: normalization, matching, interpolation, line assembly."""

from __future__ import annotations

import pytest

from karaoke.models import AlignedWord
from karaoke.pipeline.postprocess import (
    PipelineError,
    _cap_durations,
    _compute_word_cap,
    _greedy_match,
    _interpolate,
    assemble_lines,
    match_rate,
    normalize_hebrew,
)


# -------- normalize_hebrew --------

def test_normalize_strips_niqqud():
    # שָׁלוֹם (with niqqud) -> שלום -> שלומ (final ם folded to medial מ)
    assert normalize_hebrew("שָׁלוֹם") == "שלומ"


def test_normalize_final_letters_equivalent():
    # ם <-> מ, ן <-> נ, ץ <-> צ, ף <-> פ, ך <-> כ
    assert normalize_hebrew("שלום") == normalize_hebrew("שלומ")
    assert normalize_hebrew("בן") == normalize_hebrew("בנ")
    assert normalize_hebrew("ארץ") == normalize_hebrew("ארצ")


def test_normalize_strips_punctuation():
    assert normalize_hebrew("שלום,") == "שלומ"  # final ם folded
    assert normalize_hebrew("«אהבה»") == "אהבה"
    assert normalize_hebrew("yes!") == "yes"


def test_normalize_collapses_to_empty_for_punct_only():
    assert normalize_hebrew("---") == ""


# -------- _greedy_match --------

def test_greedy_match_perfect():
    matches = _greedy_match(["א", "ב", "ג"], ["א", "ב", "ג"])
    assert matches == [0, 1, 2]


def test_greedy_match_skips_hallucinated_word():
    # Whisper inserted a spurious word between two real ones.
    matches = _greedy_match(["א", "ב", "ג"], ["א", "xxx", "ב", "ג"])
    assert matches == [0, 2, 3]


def test_greedy_match_missing_word():
    # Whisper dropped the middle word entirely.
    matches = _greedy_match(["א", "ב", "ג"], ["א", "ג"])
    assert matches[0] == 0
    assert matches[1] is None
    assert matches[2] == 1


def test_greedy_match_never_reuses():
    matches = _greedy_match(["א", "א", "א"], ["א"])
    assert matches[0] == 0
    assert matches[1] is None
    assert matches[2] is None


# -------- _interpolate --------

def test_interpolate_single_missing():
    t = [(0.0, 1.0), None, (3.0, 4.0)]
    out = _interpolate(t, duration=10.0, word_cap=10.0)
    # The gap is [1.0, 3.0], one word, gets the full 2.0s slice.
    assert out[0] == (0.0, 1.0)
    assert out[2] == (3.0, 4.0)
    s, e = out[1]
    assert s == pytest.approx(1.0)
    assert e == pytest.approx(3.0)


def test_interpolate_multiple_missing_split_evenly():
    t = [(0.0, 1.0), None, None, None, (4.0, 5.0)]
    out = _interpolate(t, duration=10.0, word_cap=10.0)
    # Gap from 1.0 to 4.0 split evenly into 3 -> each 1.0s.
    assert out[1] == pytest.approx((1.0, 2.0))
    assert out[2] == pytest.approx((2.0, 3.0))
    assert out[3] == pytest.approx((3.0, 4.0))


def test_interpolate_missing_at_start_falls_back_to_zero():
    t = [None, None, (4.0, 5.0)]
    out = _interpolate(t, duration=10.0, word_cap=10.0)
    # prev_end defaults to 0.0; 2 words across [0, 4] -> 2.0s each.
    assert out[0] == pytest.approx((0.0, 2.0))
    assert out[1] == pytest.approx((2.0, 4.0))


def test_interpolate_trailing_uses_word_cap_not_duration():
    # Previously this stretched to `duration`; now it must use `word_cap` per word.
    t = [(0.0, 1.0), None, None]
    out = _interpolate(t, duration=5.0, word_cap=0.8)
    assert out[1] == pytest.approx((1.0, 1.8))
    assert out[2] == pytest.approx((1.8, 2.6))


def test_interpolate_all_missing_uses_word_cap_at_tail():
    # No anchors at all — all words are part of one big trailing run from 0.0.
    t = [None, None]
    out = _interpolate(t, duration=4.0, word_cap=1.0)
    assert out[0] == pytest.approx((0.0, 1.0))
    assert out[1] == pytest.approx((1.0, 2.0))


# -------- assemble_lines --------

def _aw(text: str, start: float, end: float, conf: float = 1.0) -> AlignedWord:
    return AlignedWord(text=text, start=start, end=end, confidence=conf)


def test_assemble_lines_basic():
    lyrics = "שלום עולם\nאיך הולך"
    aligned = [
        _aw("שלום", 0.0, 0.5),
        _aw("עולם", 0.5, 1.0),
        _aw("איך", 2.0, 2.3),
        _aw("הולך", 2.3, 2.8),
    ]
    lines = assemble_lines(lyrics, aligned, duration=10.0)
    assert len(lines) == 2
    assert lines[0].is_empty is False
    assert lines[0].start == 0.0
    assert lines[0].end == 1.0
    assert [w.text for w in lines[0].words] == ["שלום", "עולם"]
    assert [w.text for w in lines[1].words] == ["איך", "הולך"]


def test_assemble_lines_preserves_empty_line_breaks():
    lyrics = "שלום עולם\n\nאיך הולך"
    aligned = [
        _aw("שלום", 0.0, 0.5),
        _aw("עולם", 0.5, 1.0),
        _aw("איך", 2.0, 2.3),
        _aw("הולך", 2.3, 2.8),
    ]
    lines = assemble_lines(lyrics, aligned, duration=10.0)
    assert len(lines) == 3
    assert lines[1].is_empty is True


def test_assemble_lines_preserves_original_token_text():
    # User wrote with niqqud + punctuation; output must echo it.
    lyrics = "שָׁלוֹם, עוֹלָם!"
    aligned = [
        _aw("שלום", 0.0, 0.5),
        _aw("עולם", 0.5, 1.0),
    ]
    lines = assemble_lines(lyrics, aligned, duration=10.0)
    assert [w.text for w in lines[0].words] == ["שָׁלוֹם,", "עוֹלָם!"]


def test_assemble_lines_interpolates_dropped_word_silently():
    lyrics = "שלום עולם גדול"
    aligned = [
        _aw("שלום", 0.0, 1.0),
        # "עולם" is missing
        _aw("גדול", 3.0, 4.0),
    ]
    lines = assemble_lines(lyrics, aligned, duration=10.0)
    words = lines[0].words
    assert [w.text for w in words] == ["שלום", "עולם", "גדול"]
    # Interpolated word should fall between 1.0 and 3.0.
    assert 1.0 <= words[1].start <= words[1].end <= 3.0
    # OutputWord schema must not leak `confidence` or `interpolated`.
    assert set(words[0].model_dump().keys()) == {"text", "start", "end"}


def test_assemble_lines_low_confidence_treated_as_missing():
    lyrics = "שלום עולם"
    aligned = [
        _aw("שלום", 0.0, 1.0, conf=1.0),
        _aw("עולם", 1.5, 2.0, conf=0.1),  # below floor -> interpolate
    ]
    lines = assemble_lines(lyrics, aligned, duration=10.0)
    # 2nd word's timestamps should NOT match aligned[1] directly.
    assert lines[0].words[1].start != 1.5


# -------- match_rate --------

def test_match_rate_full():
    aligned = [_aw("שלום", 0, 1), _aw("עולם", 1, 2)]
    assert match_rate("שלום עולם", aligned) == 1.0


def test_match_rate_partial():
    aligned = [_aw("שלום", 0, 1)]
    # 1 of 2 words match
    assert match_rate("שלום עולם", aligned) == 0.5


def test_match_rate_empty_lyrics():
    assert match_rate("", []) == 1.0


# -------- BOM handling --------

def test_normalize_strips_bom():
    # U+FEFF glued to the front of the first word.
    assert normalize_hebrew("﻿שלום") == normalize_hebrew("שלום")
    # BOM inside the token, too.
    assert normalize_hebrew("של﻿ום") == normalize_hebrew("שלום")


def test_assemble_lines_strips_bom_from_first_word():
    lyrics = "﻿שלום עולם"
    aligned = [_aw("שלום", 0.0, 0.5), _aw("עולם", 0.5, 1.0)]
    lines = assemble_lines(lyrics, aligned, duration=10.0)
    # Output text must not contain BOM.
    assert "﻿" not in lines[0].words[0].text
    assert lines[0].words[0].text == "שלום"


# -------- empty-input guards --------

def test_assemble_lines_raises_on_empty_lyrics():
    with pytest.raises(PipelineError):
        assemble_lines("", [], duration=10.0)


def test_assemble_lines_raises_on_punctuation_only_lyrics():
    with pytest.raises(PipelineError):
        assemble_lines("--- !!! ,,,", [], duration=10.0)


# -------- word cap / tail stretching --------

def test_compute_word_cap_uses_median():
    # Median of [0.4, 0.5, 0.6] = 0.5, so cap = max(0.6, 1.0) = 1.0.
    t = [(0.0, 0.4), (1.0, 1.5), (2.0, 2.6)]
    assert _compute_word_cap(t) == pytest.approx(1.0)


def test_compute_word_cap_floor():
    # Tiny durations => floor kicks in.
    t = [(0.0, 0.1), (1.0, 1.1), (2.0, 2.1)]
    assert _compute_word_cap(t) == pytest.approx(0.6)


def test_compute_word_cap_fallback_when_few_anchors():
    assert _compute_word_cap([(0.0, 1.0)]) == pytest.approx(1.5)
    assert _compute_word_cap([None, None]) == pytest.approx(1.5)


def test_interpolate_trailing_does_not_stretch_to_duration():
    # 1 anchor, then 2 trailing missing words, audio is much longer than singing.
    t = [(0.0, 1.0), None, None]
    out = _interpolate(t, duration=180.0, word_cap=0.8)
    # Without the fix, words 1 and 2 would span from 1.0 -> 180.0 (89s each).
    # With the fix, each gets `word_cap` = 0.8s.
    assert out[1] == pytest.approx((1.0, 1.8))
    assert out[2] == pytest.approx((1.8, 2.6))


def test_cap_durations_clamps_long_words():
    out = _cap_durations([(0.0, 0.5), (1.0, 50.0)], word_cap=1.0)
    assert out[0] == (0.0, 0.5)
    assert out[1] == (1.0, 2.0)


def test_assemble_lines_tail_does_not_stretch_to_audio_end():
    # Whisper anchored only the first line; second line went unmatched.
    # Audio is 200s. Final line must NOT end at ~200s.
    lyrics = "שלום עולם\nאיך הולך לך"
    aligned = [_aw("שלום", 0.0, 0.5), _aw("עולם", 0.5, 1.0)]
    lines = assemble_lines(lyrics, aligned, duration=200.0)
    assert lines[-1].is_empty is False
    # Last line should end well before the 200s mark — within a few seconds of
    # the last anchor, capped by word_cap per word.
    assert lines[-1].end < 10.0


def test_assemble_lines_caps_stretched_aligner_word():
    # Whisper returned a final word with a wildly stretched end timestamp.
    lyrics = "שלום עולם"
    aligned = [
        _aw("שלום", 0.0, 0.5),
        _aw("עולם", 1.0, 60.0),  # 59s for one word — pathological
    ]
    lines = assemble_lines(lyrics, aligned, duration=180.0)
    # Cap should keep the last word's duration well under the original 59s.
    last = lines[0].words[-1]
    assert last.end - last.start <= 1.6  # 2 * median(0.5) = 1.0, but floor is 0.6 — generous bound
