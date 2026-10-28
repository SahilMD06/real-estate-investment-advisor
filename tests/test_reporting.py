"""Report building blocks (no browser needed)."""
from scripts.build_docs_pdf import prepare
from src.reporting.pdf import markdown_to_html
from src.reporting.template import kpis, page, table


def test_markdown_renders_tables_and_code():
    out = markdown_to_html("| a | b |\n|---|---|\n| 1 | 2 |\n\n```python\nx = 1\n```\n")
    assert "<table>" in out and "<td>1</td>" in out
    assert "<code" in out and "x = 1" in out


def test_table_escapes_and_highlights():
    out = table(["Model", "AUC"], [["<b>x</b>", "0.9"], ["y", "0.8"]], numeric={1}, highlight=1)
    assert "&lt;b&gt;x&lt;/b&gt;" in out
    assert out.count("class='hl'") == 1
    assert "class='num'" in out


def test_page_is_self_contained():
    html = page("Title", "Sub", kpis([("Rows", "10", "note")]))
    assert html.startswith("<!doctype html>") and "<style>" in html and "Rows" in html


def test_doc_links_become_text_and_mermaid_is_replaced():
    md = "See [Methodology](docs/METHODOLOGY.md).\n\n```mermaid\nflowchart TB\nA-->B\n```\n"
    out = prepare(md)
    assert "METHODOLOGY.md" not in out and "See Methodology." in out
    assert "```mermaid" not in out and "architecture_dataflow.svg" in out
