"""Apply exported page-content JSON to ``docs/index.html``."""

from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from typing import Any

from lce.pipeline.paths import REPO_ROOT

# Marker that immediately follows each editable block in docs/index.html.
_EDITABLE_BOUNDARY: dict[str, str] = {
    "title": 'data-editable="header-copy"',
    "header-copy": '<p class="meta-links edit-entry">',
    "plot-copy": '<p class="meta-links edit-entry">',
}


def replace_editable_inner(html: str, editable_id: str, new_inner: str) -> str:
    """Replace inner HTML of a ``data-editable`` element."""
    marker = f'data-editable="{editable_id}"'
    pos = html.find(marker)
    if pos == -1:
        raise KeyError(f"missing editable block {editable_id!r}")

    open_start = html.rfind("<", 0, pos)
    if open_start == -1:
        raise ValueError(f"no opening tag for {editable_id!r}")

    open_end = html.find(">", open_start) + 1
    tag_match = re.match(r"<(\w+)", html[open_start:open_end])
    if not tag_match:
        raise ValueError(f"could not parse tag for {editable_id!r}")
    tag = tag_match.group(1).lower()
    close_tag = f"</{tag}>"

    boundary = _EDITABLE_BOUNDARY.get(editable_id)
    if boundary:
        next_pos = html.find(boundary, open_end)
        if next_pos == -1:
            raise ValueError(f"missing boundary after {editable_id!r}")
        close_start = html.rfind(close_tag, open_end, next_pos)
        if close_start == -1:
            raise ValueError(f"missing closing tag for {editable_id!r}")
        close_end = close_start + len(close_tag)
        return html[:open_end] + new_inner + html[close_end:]

    # Generic depth-based fallback for other tags.
    depth = 1
    i = open_end
    close_end = None
    open_needle = f"<{tag}"
    while i < len(html) and depth:
        next_open = html.find(open_needle, i)
        next_close = html.find(close_tag, i)
        if next_close == -1:
            raise ValueError(f"unclosed tag for {editable_id!r}")
        if next_open != -1 and next_open < next_close:
            depth += 1
            i = next_open + len(open_needle)
        else:
            depth -= 1
            if depth == 0:
                close_start = next_close
                close_end = next_close + len(close_tag)
                break
            i = next_close + len(close_tag)
    if close_end is None:
        raise ValueError(f"failed to close tag for {editable_id!r}")
    return html[:open_end] + new_inner + html[close_end:]


def apply_page_content(payload: dict[str, Any], *, index_path: Path | None = None) -> Path:
    """Write editable blocks from export JSON into ``docs/index.html``."""
    index_path = index_path or (REPO_ROOT / "docs" / "index.html")
    blocks = payload.get("blocks")
    if not isinstance(blocks, dict):
        raise ValueError("payload must include a blocks object")

    html = index_path.read_text(encoding="utf-8")
    if "title" in blocks:
        plain_title = unescape(str(blocks["title"]).replace("&nbsp;", " "))
        html = re.sub(r"<title>.*?</title>", f"<title>{plain_title}</title>", html, count=1)
        html = replace_editable_inner(html, "title", str(blocks["title"]))

    for block_id in ("header-copy", "plot-copy"):
        if block_id in blocks:
            html = replace_editable_inner(html, block_id, str(blocks[block_id]))

    index_path.write_text(html, encoding="utf-8")
    return index_path


def apply_page_content_json(raw: str, *, index_path: Path | None = None) -> Path:
    payload = json.loads(raw)
    return apply_page_content(payload, index_path=index_path)
