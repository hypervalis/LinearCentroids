from pathlib import Path

from lce.site_content import apply_page_content, replace_editable_inner


def test_replace_editable_blocks_use_known_boundaries(tmp_path: Path) -> None:
    html = """<!doctype html>
<html><body>
<h1 data-editable="title">Old title</h1>
<div data-editable="header-copy"><p>Header</p></div>
<p class="meta-links edit-entry">edit</p>
<section>
<div data-editable="plot-copy"><p>Plot</p></div>
<p class="meta-links edit-entry">edit</p>
</section>
</body></html>"""

    html = replace_editable_inner(html, "title", "New title")
    html = replace_editable_inner(html, "header-copy", "<p>New header</p>")
    html = replace_editable_inner(html, "plot-copy", "<p>New plot</p><p>Second paragraph</p>")

    assert 'data-editable="title">New title</h1>' in html
    assert 'data-editable="header-copy"><p>New header</p></div>' in html
    assert "Second paragraph" in html
    assert html.count('<p class="meta-links edit-entry">') == 2


def test_apply_page_content_writes_all_blocks(tmp_path: Path) -> None:
    index_path = tmp_path / "index.html"
    index_path.write_text(
        """<html><head><title>Old</title></head><body>
<h1 data-editable="title">Old</h1>
<div data-editable="header-copy">H</div>
<p class="meta-links edit-entry"></p>
<div data-editable="plot-copy">P</div>
<p class="meta-links edit-entry"></p>
</body></html>""",
        encoding="utf-8",
    )

    apply_page_content(
        {
            "blocks": {
                "title": "T",
                "header-copy": "<p>H2</p>",
                "plot-copy": "<p>P2</p><p>Closing extra</p>",
            }
        },
        index_path=index_path,
    )

    out = index_path.read_text(encoding="utf-8")
    assert "<title>T</title>" in out
    assert "Closing extra" in out
