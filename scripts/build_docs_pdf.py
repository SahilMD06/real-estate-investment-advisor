"""Bundle the project documentation into one PDF for submission.

Usage (from the project root):
    python -m scripts.build_docs_pdf

Reads the Markdown files in docs/ (the editable source, rendered on GitHub)
and writes reports/Project_Documentation.pdf. The Mermaid data-flow diagram is
replaced by its pre-rendered image, docs/architecture_dataflow.svg.
"""
from __future__ import annotations

import html
import re
import sys

from src.reporting.pdf import PdfToolMissing, html_to_pdf, markdown_to_html
from src.reporting.template import page
from src.utils.config import DOCS_DIR, REPORTS_DIR
from src.utils.logging import get_logger

log = get_logger("build_docs_pdf")

# (file, short title) in reading order.
DOCS = [
    ("METHODOLOGY.md", "Methodology"),
    ("ARCHITECTURE.md", "Architecture & data flow"),
    ("DATA_DICTIONARY.md", "Data dictionary"),
    ("DEPLOYMENT.md", "Deployment"),
    ("DEMO_SCRIPT.md", "Demo script"),
]
DIAGRAM = "architecture_dataflow.svg"
_MERMAID = re.compile(r"```mermaid\n.*?```", re.S)
# Links between docs point at .md files; inside the PDF they become plain text.
_DOC_LINK = re.compile(r"\[([^\]]+)\]\((?:docs/)?[A-Z_]+\.md(?:#[^)]*)?\)")


def prepare(md_text: str) -> str:
    if (DOCS_DIR / DIAGRAM).exists():
        md_text = _MERMAID.sub(f"![Data flow](../docs/{DIAGRAM})", md_text)
    return _DOC_LINK.sub(r"\1", md_text)


def build() -> str:
    present = [(f, t) for f, t in DOCS if (DOCS_DIR / f).exists()]
    toc = "".join(f"<li>{i + 1}. {html.escape(t)}</li>" for i, (_, t) in enumerate(present))
    parts = [f"<div class='summary'><h2>Contents</h2><ol style='list-style:none;padding-left:0'>{toc}</ol>"
             "<p style='font-size:13px;color:#475569'>Companion reports: <b>EDA_Report.pdf</b> (20 analyses) and "
             "<b>Model_Evaluation_Report.pdf</b> (metrics, charts, MLflow evidence), both in <code>reports/</code>."
             "</p></div>"]
    for i, (fname, _) in enumerate(present):
        body = markdown_to_html(prepare((DOCS_DIR / fname).read_text(encoding="utf-8")))
        parts.append(f"<section class='md{' page-break' if i else ''}'>{body}</section>")
    return page("Project Documentation", "Methodology · architecture · data dictionary · deployment · demo",
                "".join(parts), footer="PropWise AI is an educational tool. It is not investment advice.")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS_DIR / "project_documentation.html"
    html_path.write_text(build(), encoding="utf-8")
    try:
        html_to_pdf(html_path, REPORTS_DIR / "Project_Documentation.pdf", "PropWise AI · Project Documentation")
    except PdfToolMissing as exc:
        log.warning("PDF not built: %s", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
