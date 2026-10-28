"""HTML → PDF via headless Chromium (Playwright).

Plotly charts are drawn by JavaScript, so a real browser is needed to print
them. Playwright is a development dependency (requirements-dev.txt); after
installing it once run ``python -m playwright install chromium``.
"""
from __future__ import annotations

from pathlib import Path

from src.utils.logging import get_logger

log = get_logger(__name__)

# A4 at 96 dpi with 12 mm side margins → ~703 px printable width. Rendering the
# page at this width means charts are laid out at exactly the printed size.
_A4_PRINT_WIDTH_PX = 703
# Print at 85 % so two typical analyses fit on one page; charts are vectors so
# they stay sharp. The page is laid out at width/scale so nothing is clipped.
_PRINT_SCALE = 0.85

_FOOTER = (
    "<div style='width:100%;font-size:8px;color:#64748b;padding:0 12mm;"
    "font-family:Segoe UI,Helvetica,Arial,sans-serif;display:flex;justify-content:space-between'>"
    "<span>{title}</span><span>Page <span class='pageNumber'></span> of <span class='totalPages'></span></span></div>"
)


class PdfToolMissing(RuntimeError):
    pass


def html_to_pdf(html_path: Path | str, pdf_path: Path | str, title: str) -> Path:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise PdfToolMissing(
            "Playwright is required to build PDF reports. Run:\n"
            "  pip install -r requirements-dev.txt\n  python -m playwright install chromium"
        ) from exc

    html_path, pdf_path = Path(html_path).resolve(), Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # pragma: no cover
            raise PdfToolMissing("Chromium for Playwright is not installed. Run: "
                                 "python -m playwright install chromium") from exc
        page = browser.new_page(viewport={"width": int(_A4_PRINT_WIDTH_PX / _PRINT_SCALE), "height": 1000})
        page.goto(html_path.as_uri(), wait_until="load")
        # Wait until every Plotly container has drawn its SVG.
        page.wait_for_function(
            "() => [...document.querySelectorAll('.plotly-graph-div')].every(d => d.querySelector('.main-svg'))",
            timeout=120_000,
        )
        page.wait_for_timeout(500)
        page.emulate_media(media="print")
        page.pdf(
            path=str(pdf_path), format="A4", print_background=True, scale=_PRINT_SCALE,
            margin={"top": "12mm", "bottom": "14mm", "left": "12mm", "right": "12mm"},
            display_header_footer=True, header_template="<span></span>",
            footer_template=_FOOTER.format(title=title),
        )
        browser.close()
    log.info("Wrote %s", pdf_path)
    return pdf_path


def markdown_to_html(md_text: str) -> str:
    """Render Markdown (tables, fenced code) for the documentation PDF."""
    import markdown

    return markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
