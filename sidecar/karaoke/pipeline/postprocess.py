"""Hebrew text normalization, robust line assembly, and silent interpolation.

The original prototype assumed `len(line.split()) == number_of_aligned_words`,
which breaks whenever Whisper drops, merges, or hallucinates a word. We replace
that with a sequential greedy Levenshtein-window match, plus linear
interpolation for any original word that didn't get a confident alignment.

Output `OutputLine` / `OutputWord` carry the ORIGINAL user-supplied tokens, not
the cleaned ones, and never expose `confidence` / `interpolated`.
"""

from __future__ import annotations

import re

from Levenshtein import distance as lev_distance

from karaoke.models import AlignedWord, OutputLine, OutputWord


class PipelineError(ValueError):
    """Raised for degenerate input (empty lyrics, empty alignment, etc.)."""


# Hebrew niqqud (cantillation + vowel points): U+0591..U+05C7
_NIQQUD_RE = re.compile(r"[֑-ׇ]")
# Final-letter -> medial form, so מ/ם, נ/ן, צ/ץ, פ/ף, כ/ך match for similarity.
_FINAL_MAP = str.maketrans("םןץףך", "מנצפכ")
# Keep Hebrew letters (֐-׿), latin word chars, digits; drop punctuation.
_KEEP_RE = re.compile(r"[^\w֐-׿]+", re.UNICODE)

# Match threshold: similarity >= this to accept a Whisper word as the original.
_MATCH_THRESHOLD = 0.55
# Look this many aligned words ahead when greedy-matching a single original word.
_LOOKAHEAD = 5
# Whisper words with probability below this are treated as un-anchored (interpolate).
_CONFIDENCE_FLOOR = 0.4
# Floor for the per-word duration cap; the cap is max(this, 2 * median anchored dur).
_WORD_CAP_FLOOR_S = 0.6
# Fallback word cap when we have too few anchored words to compute a median.
_WORD_CAP_FALLBACK_S = 1.5


def normalize_hebrew(word: str) -> str:
    """Strip niqqud, normalize final letters, strip punctuation, casefold.

    Also strips any U+FEFF (BOM) characters anywhere in the token; these sneak in
    when lyrics files are saved as UTF-8-with-BOM and the BOM ends up glued to
    the first word.
    """
    w = word.replace("﻿", "")
    w = _NIQQUD_RE.sub("", w)
    w = w.translate(_FINAL_MAP)
    w = _KEEP_RE.sub("", w)
    return w.casefold()


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    d = lev_distance(a, b)
    return 1.0 - d / max(len(a), len(b))


def _tokenize_lyrics(lyrics_text: str) -> list[list[tuple[str, str]]]:
    """Split lyrics into lines. Each line is a list of (original_token, cleaned_token).

    Empty lines become `[]` to mark visual breaks.
    """
    # Strip any leading BOM (some editors save Hebrew text as UTF-8-with-BOM).
    text = lyrics_text.lstrip("﻿").replace("\r\n", "\n")
    lines: list[list[tuple[str, str]]] = []
    for raw in text.split("\n"):
        stripped = raw.strip().lstrip("﻿")
        if not stripped:
            lines.append([])
            continue
        toks = stripped.split()
        # Strip BOM from each token defensively, then preserve the cleaned-of-BOM
        # form as the "original" — we never want ﻿ in user-visible output.
        cleaned_toks = [t.replace("﻿", "") for t in toks if t.replace("﻿", "")]
        # Drop tokens whose normalized form is empty (pure punctuation, dashes, etc.).
        entries = [(t, normalize_hebrew(t)) for t in cleaned_toks]
        entries = [(o, c) for (o, c) in entries if c]
        lines.append(entries)
    # Drop trailing blank lines so the output doesn't end with empty separators.
    while lines and lines[-1] == []:
        lines.pop()
    return lines


def _greedy_match(
    originals_cleaned: list[str],
    aligned_cleaned: list[str],
    lookahead: int = _LOOKAHEAD,
    threshold: float = _MATCH_THRESHOLD,
) -> list[int | None]:
    """For each original word, the index into `aligned` that matched it (or None).

    Walks left-to-right and never reuses an aligned word. If a clearly better
    match for the current original lies a few positions ahead, we skip over
    intervening aligned words (which were likely hallucinations or merges).
    """
    matches: list[int | None] = []
    ai = 0
    n = len(aligned_cleaned)
    for oword in originals_cleaned:
        if ai >= n:
            matches.append(None)
            continue
        best_j: int | None = None
        best_score = -1.0
        for j in range(ai, min(ai + lookahead, n)):
            s = _similarity(oword, aligned_cleaned[j])
            if s > best_score:
                best_score = s
                best_j = j
                if s == 1.0:
                    break
        if best_j is not None and best_score >= threshold:
            matches.append(best_j)
            ai = best_j + 1
        else:
            matches.append(None)
    return matches


def _compute_word_cap(timings: list[tuple[float, float] | None]) -> float:
    """`max(_WORD_CAP_FLOOR_S, 2 * median anchored-word duration)`.

    Anchored words = those with a confident alignment match (non-None entries).
    Falls back to `_WORD_CAP_FALLBACK_S` when too few anchors exist to be useful.
    """
    durations = [e - s for t in timings if t is not None for (s, e) in [t] if e > s]
    if len(durations) < 3:
        return _WORD_CAP_FALLBACK_S
    durations.sort()
    median = durations[len(durations) // 2]
    return max(_WORD_CAP_FLOOR_S, 2.0 * median)


def _interpolate(
    timings: list[tuple[float, float] | None],
    duration: float,
    word_cap: float,
) -> list[tuple[float, float]]:
    """Fill in `None` entries.

    For interior None-runs (anchors on both sides), spread linearly across the gap.
    For trailing None-runs (no next anchor), give each word `word_cap` seconds
    rather than stretching to `duration` — this prevents the final line from
    inheriting all the dead air at the end of the track.
    For leading None-runs (no previous anchor), spread linearly from 0.0.
    """
    n = len(timings)
    out: list[tuple[float, float]] = [(0.0, 0.0)] * n
    i = 0
    while i < n:
        if timings[i] is not None:
            out[i] = timings[i]  # type: ignore[assignment]
            i += 1
            continue
        prev_i = i - 1
        while prev_i >= 0 and timings[prev_i] is None:
            prev_i -= 1
        prev_end = timings[prev_i][1] if prev_i >= 0 else 0.0  # type: ignore[index]

        next_i = i
        while next_i < n and timings[next_i] is None:
            next_i += 1
        gap_count = next_i - i

        if next_i < n:
            next_start = timings[next_i][0]  # type: ignore[index]
            gap_dur = max(0.0, next_start - prev_end)
            per = gap_dur / gap_count if gap_count > 0 else 0.0
        else:
            # Trailing run: cap per word instead of stretching to `duration`.
            per = word_cap

        for k in range(gap_count):
            s = prev_end + per * k
            e = prev_end + per * (k + 1) if per > 0 else s
            out[i + k] = (s, e)
        i = next_i
    return out


def _cap_durations(
    words: list[tuple[float, float]], word_cap: float
) -> list[tuple[float, float]]:
    """Clamp each word's end so `end - start <= word_cap`. Leaves gaps when capped."""
    capped: list[tuple[float, float]] = []
    for s, e in words:
        if e - s > word_cap:
            e = s + word_cap
        capped.append((s, e))
    return capped


def assemble_lines(
    lyrics_text: str,
    aligned: list[AlignedWord],
    duration: float,
) -> list[OutputLine]:
    """Convert (lyrics, aligned-words) into the final per-line output structure."""
    structured = _tokenize_lyrics(lyrics_text)

    flat: list[tuple[int, int, str, str]] = []  # (line_idx, word_idx_in_line, original, cleaned)
    for li, line in enumerate(structured):
        for wi, (orig, cleaned) in enumerate(line):
            flat.append((li, wi, orig, cleaned))

    if not flat:
        raise PipelineError(
            "lyrics produced zero tokens after normalization — file is empty or "
            "contains only whitespace/punctuation"
        )

    aligned_cleaned = [normalize_hebrew(w.text) for w in aligned]
    matches = _greedy_match([c for _, _, _, c in flat], aligned_cleaned)

    timings: list[tuple[float, float] | None] = []
    for j in matches:
        if j is None:
            timings.append(None)
            continue
        aw = aligned[j]
        if aw.confidence < _CONFIDENCE_FLOOR:
            timings.append(None)
        else:
            timings.append((aw.start, aw.end))

    word_cap = _compute_word_cap(timings)
    resolved = _interpolate(timings, duration, word_cap)
    resolved = _cap_durations(resolved, word_cap)

    out_lines: list[OutputLine] = []
    pos = 0
    for line in structured:
        if not line:
            out_lines.append(OutputLine(is_empty=True))
            continue
        words_out: list[OutputWord] = []
        for orig, _cleaned in line:
            s, e = resolved[pos]
            words_out.append(OutputWord(text=orig, start=s, end=e))
            pos += 1
        out_lines.append(
            OutputLine(
                is_empty=False,
                start=words_out[0].start,
                end=words_out[-1].end,
                words=words_out,
            )
        )
    return out_lines


def match_rate(
    lyrics_text: str,
    aligned: list[AlignedWord],
) -> float:
    """Fraction of original words that got a confident alignment match.

    Used by the orchestrator to decide whether to retry with VAD-trimmed audio.
    """
    structured = _tokenize_lyrics(lyrics_text)
    cleaned_originals = [c for line in structured for (_o, c) in line]
    if not cleaned_originals:
        return 1.0
    aligned_cleaned = [normalize_hebrew(w.text) for w in aligned]
    matches = _greedy_match(cleaned_originals, aligned_cleaned)
    confident = 0
    for j in matches:
        if j is None:
            continue
        if aligned[j].confidence >= _CONFIDENCE_FLOOR:
            confident += 1
    return confident / len(cleaned_originals)
