"""formatter.py – heuristic conversion of plain LLM output into Markdown

This module provides lightweight, dependency-free functions that post-process a raw
string (e.g. an LLM answer) and converts patterns that *look* like lists or tables
into explicit GitHub-flavoured Markdown.

The heuristics purposefully err on the side of *under-formatting* — if we are not
confident, we leave the text unchanged so the user can still read it.  False
positives would silently corrupt content, while a false negative only loses
formatting niceties.

The entrypoint is ``format_markdown(text)``.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List

__all__ = [
    "format_markdown",
]

# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------
_bullet_starters = {
    "using",
    "employing",
    "incorporating",
    "developing",
    "adopting",
    "leveraging",
    "utilising",
    "utilizing",
    "creating",
    "implementing",
}

# Regex pre-compiled for speed
#: any_line_with_colon_and_commas = re.compile(r":\s*[^\n]*[,;][^\n]*")
_colon_list_rx = re.compile(r":\s*[^\n]*[,;][^\n]*")
_table_row_rx = re.compile(r"^\s*\S+\s+\d+(?:\.\d+)?(?:\s+\d+(?:\.\d+)?)+\s*$")
_key_value_row_rx = re.compile(r"^\s*[^:]+:\s*.+$")
_end_punct_rx = re.compile(r"[.!?]$")
_first_word_rx = re.compile(r"^(\w+)")


# ---------------------------------------------------------------------------
# List detection
# ---------------------------------------------------------------------------

def _paragraph_as_list(lines: List[str]) -> str | None:
    """Return bullet-ised version of *lines* if they *look* like a list.

    The caller guarantees *lines* is a non-empty sequence of text lines that
    belonged to a paragraph (i.e. separated by blank lines originally).
    """
    if len(lines) < 2:
        return None  # a list needs at least two items

    # Strategy 1 – colon + comma/semicolon separated list (single line)
    if len(lines) == 1 and _colon_list_rx.search(lines[0]):
        # split on comma or semicolon
        after_colon = lines[0].split(":", 1)[1]
        parts = re.split(r"[;,]", after_colon)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            return "\n".join(f"- {p}" for p in parts)

    # Strategy 2 – multiple lines, none ends with terminal punctuation
    if all(not _end_punct_rx.search(l.strip()) for l in lines):
        return "\n".join(f"- {l.strip()}" for l in lines)

    # Strategy 3 – repeated lexical start pattern
    first_words = [_first_word_rx.match(l.strip().lower()).group(1) if _first_word_rx.match(l.strip()) else "" for l in lines]
    counts = Counter(first_words)
    # number of repeats of most common first word
    most_common_word, freq = counts.most_common(1)[0]
    if most_common_word and freq >= 2:
        return "\n".join(f"- {l.strip()}" for l in lines)

    return None  # not recognised as list


# ---------------------------------------------------------------------------
# Table detection
# ---------------------------------------------------------------------------

def _paragraph_as_table(lines: List[str]) -> str | None:
    """Return Markdown table of *lines* if they match simple key/value or numeric pattern."""
    if len(lines) < 2:
        return None

    # Strategy 1 – key: value pairs (colon-separated)
    if all(_key_value_row_rx.match(l) for l in lines):
        headers = ["Key", "Value"]
        sep = ["---", "---"]
        body = []
        for l in lines:
            key, val = l.split(":", 1)
            body.append(f"| {key.strip()} | {val.strip()} |")
        return "| " + " | ".join(headers) + " |\n| " + " | ".join(sep) + " |\n" + "\n".join(body)

    # Strategy 2 – whitespace-separated numeric table (at least 2 numeric cols)
    if all(_table_row_rx.match(l) for l in lines):
        # derive column count
        col_counts = [len(re.split(r"\s+", l.strip())) for l in lines]
        if len(set(col_counts)) != 1:
            return None  # not rectangular
        n_cols = col_counts[0]
        headers = [f"Col{i+1}" for i in range(n_cols)]
        sep = ["---"] * n_cols
        body_rows = []
        for l in lines:
            cells = re.split(r"\s+", l.strip())
            body_rows.append("| " + " | ".join(cells) + " |")
        return "| " + " | ".join(headers) + " |\n| " + " | ".join(sep) + " |\n" + "\n".join(body_rows)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_markdown(text: str) -> str:
    """Return *text* with implicit lists/tables converted to explicit Markdown.

    The algorithm walks paragraph by paragraph (split on blank lines).  Each
    paragraph is inspected by the detectors above; if none triggers, it is left
    untouched.  Finally, paragraphs are re-joined with blank lines.
    """
    paragraphs: List[str] = re.split(r"\n{2,}", text.strip())
    out_paragraphs: List[str] = []

    for para in paragraphs:
        # preserve internal line breaks for analysis
        lines = para.split("\n")
        table_md = _paragraph_as_table(lines)
        if table_md:
            out_paragraphs.append(table_md)
            continue

        list_md = _paragraph_as_list(lines)
        if list_md:
            out_paragraphs.append(list_md)
            continue

        out_paragraphs.append(para)

    return "\n\n".join(out_paragraphs) + "\n" 