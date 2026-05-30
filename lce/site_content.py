"""Apply exported page-content JSON to ``docs/index.html``."""

from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from typing import Any

from lce.pipeline.paths import REPO_ROOT


def replace_div_inner(html: str, editable_id: str, new_inner: str) -> str:
    marker = f'data-editable="{editable_id}"'
    pos = html.find(marker)
    if pos == -1:
        raise KeyError(f"missing editable block {editable_id!r}")
    open_start = html.rfind("<div", 0, pos)
    open_end = html.find(">", open_start) + 1
    i = open_end
    depth = 1
    while i < len(html) and depth:
        next_open = html.find("<div", i)
        next_close = html.find("</div>", i)
        if next_close == -1:
            raise ValueError(f"unclosed div for {editable_id!r}")
        if next_open != -1 and next_open < next_close:
            depth += 1
            i = next_open + 4
        else:
            depth -= 1
            if depth == 0:
                close_start = next_close
                break
            i = next_close + 6
    else:
        raise ValueError(f"failed to close div for {editable_id!r}")
    return html[:open_end] + new_inner + html[close_start:]


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
        html = replace_div_inner(html, "title", str(blocks["title"]))

    for block_id in ("header-copy", "plot-copy"):
        if block_id in blocks:
            html = replace_div_inner(html, block_id, str(blocks[block_id]))

    index_path.write_text(html, encoding="utf-8")
    return index_path


def apply_page_content_json(raw: str, *, index_path: Path | None = None) -> Path:
    payload = json.loads(raw)
    return apply_page_content(payload, index_path=index_path)
