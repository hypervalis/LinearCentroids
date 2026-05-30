from pathlib import Path

from lce.site_content import apply_page_content, replace_editable_inner

# Mirrors the real docs/index.html structure: editable blocks bounded by
# structural anchors (no edit-only markup left in the deployed page).
SAMPLE = """<!doctype html>
<html><head><title>Old</title></head><body>
<main>
<header class="page-header">
<h1 data-editable="title">Old title</h1>
<div data-editable="header-copy"><p>Header</p></div>
</header>
<section class="page-section">
<div data-editable="plot-copy"><p>Plot</p><div>Nested</div></div>
</section>
<footer class="site-footer"><p>footer</p></footer>
</main>
</body></html>"""


def test_replace_editable_blocks_use_structural_boundaries() -> None:
    html = SAMPLE
    html = replace_editable_inner(html, "title", "New title")
    html = replace_editable_inner(html, "header-copy", "<p>New header</p>")
    html = replace_editable_inner(
        html, "plot-copy", "<p>New plot</p><p>Closing extra</p>"
    )

    assert 'data-editable="title">New title</h1>' in html
    assert 'data-editable="header-copy"><p>New header</p></div>' in html
    assert "Closing extra" in html
    # Footer and surrounding structure are preserved.
    assert html.count("<footer") == 1
    assert "footer" in html


def test_apply_page_content_writes_all_blocks(tmp_path: Path) -> None:
    index_path = tmp_path / "index.html"
    index_path.write_text(SAMPLE, encoding="utf-8")

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
    assert "<footer" in out
