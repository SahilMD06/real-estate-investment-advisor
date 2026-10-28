"""Shared HTML template for PropWise reports (screen + A4 print).

Reports are written as self-contained HTML (interactive Plotly charts) and
then printed to PDF by :mod:`src.reporting.pdf`. One stylesheet serves both:
print rules keep cards, tables and figures from splitting across pages.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

import plotly.io as pio

from src.utils.config import APP_NAME

CSS = """
:root{--ink:#0f172a;--ink2:#475569;--muted:#64748b;--line:#e2e8f0;--bg:#f6f8fb;--card:#fff;--accent:#0f8a65;--navy:#0b1f3a}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.55}
header.cover{background:var(--navy);color:#fff;padding:36px 28px 30px}
header.cover .wrap{max-width:1080px;margin:auto}
header.cover .eyebrow{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#6ee7b7;margin:0 0 6px;font-weight:600}
header.cover h1{margin:0 0 6px;font-size:26px;font-weight:650;letter-spacing:-.01em}
header.cover p{margin:0;color:#cbd5e1;font-size:13.5px}
main{max-width:1080px;margin:auto;padding:24px 16px 48px}
.summary,.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin-bottom:16px}
.summary h2,.card h2{font-size:17px;margin:0 0 10px}
.summary li{margin:5px 0;color:var(--ink2);font-size:13.5px}
.card p,.card li{font-size:13.5px;color:var(--ink2)}
h2.part{font-size:20px;margin:30px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--accent)}
h3.group{font-size:11.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:28px 0 10px}
nav.toc{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 22px}
nav.toc a{font-size:12px;color:var(--ink2);text-decoration:none;border:1px solid var(--line);background:#fff;border-radius:999px;padding:3px 10px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 22px;margin-top:10px}
.grid div{font-size:13px;color:var(--ink2)}
.grid b{display:block;font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:2px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0 4px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f1f5f9;color:var(--ink);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.hl td{background:#e7f5ef;font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:4px 0 16px}
.kpi{background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.kpi .l{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}
.kpi .v{font-size:19px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
.kpi .s{font-size:11.5px;color:var(--ink2)}
.note{border-left:3px solid var(--accent);background:#eef8f4;padding:10px 14px;border-radius:6px;font-size:13px;color:#0b5a43;margin:10px 0}
.warn{border-left:3px solid #b45309;background:#fff7e6;padding:10px 14px;border-radius:6px;font-size:13px;color:#7a3e05;margin:10px 0}
.figrow{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.figrow>div{min-width:0}
figure.shot{margin:8px 0 4px}
figure.shot img{width:100%;max-height:560px;object-fit:contain;object-position:left top;border:1px solid var(--line);border-radius:8px}
figure.shot figcaption{font-size:12px;color:var(--muted);margin-top:4px}
.md h1{font-size:22px;margin:4px 0 10px}.md h2{font-size:17px;margin:22px 0 8px}.md h3{font-size:14.5px;margin:16px 0 6px}
.md h4{font-size:13.5px;margin:14px 0 6px}
.md p,.md li{font-size:13px;color:var(--ink2)}
.md code{font-family:"JetBrains Mono",Consolas,monospace;font-size:12px;background:#f1f5f9;padding:1px 4px;border-radius:4px}
.md pre{background:#0f172a;color:#e2e8f0;padding:10px 12px;border-radius:8px;overflow:auto;font-size:11.5px;white-space:pre-wrap}
.md pre code{background:none;color:inherit;padding:0}
.md blockquote{border-left:3px solid var(--accent);margin:8px 0;padding:4px 12px;background:#eef8f4;color:#0b5a43}
.md img{max-width:100%;max-height:820px;display:block;margin:6px auto}
footer{max-width:1080px;margin:auto;padding:0 16px 32px;font-size:11.5px;color:var(--muted)}
@media(max-width:760px){.grid,.kpis{grid-template-columns:1fr 1fr}.figrow{grid-template-columns:1fr}}
@media print{
  body{background:#fff}
  main{padding:14px 0 0;max-width:none}
  header.cover{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .card,.summary,figure.shot,table,.kpis,.note,.warn{break-inside:avoid}
  h2.part{break-before:page}
  h2.part.first{break-before:auto}
  .page-break{break-before:page}
  h2,h3,h4{break-after:avoid}
  nav.toc{display:none}
  .md pre{white-space:pre-wrap}
}
"""


class FigureEmbedder:
    """Embeds Plotly figures, including plotly.js only once per document."""

    def __init__(self) -> None:
        self._first = True

    def __call__(self, fig) -> str:
        out = pio.to_html(fig, full_html=False, include_plotlyjs=self._first,
                          config={"displaylogo": False, "responsive": True, "staticPlot": False})
        self._first = False
        return out


def page(title: str, subtitle: str, body: str, eyebrow: str = APP_NAME,
         footer: str = "Findings describe this dataset only and are not investment advice.") -> str:
    generated = f"{datetime.now(timezone.utc):%d %b %Y %H:%M} UTC"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        f"<header class='cover'><div class='wrap'><p class='eyebrow'>{html.escape(eyebrow)}</p>"
        f"<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)} · generated {generated}</p></div></header>"
        f"<main>{body}</main><footer>{html.escape(footer)}</footer></body></html>"
    )


def table(headers: list[str], rows: list[list], numeric: set[int] | None = None,
          highlight: int | None = None) -> str:
    numeric = numeric or set()
    th = "".join(f"<th class='{'num' if i in numeric else ''}'>{html.escape(str(h))}</th>" for i, h in enumerate(headers))
    body = []
    for r_i, row in enumerate(rows):
        tds = "".join(f"<td class='{'num' if i in numeric else ''}'>{html.escape(str(v))}</td>" for i, v in enumerate(row))
        body.append(f"<tr class='{'hl' if r_i == highlight else ''}'>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def kpis(items: list[tuple[str, str, str]]) -> str:
    return "<div class='kpis'>" + "".join(
        f"<div class='kpi'><div class='l'>{html.escape(l)}</div><div class='v'>{html.escape(v)}</div>"
        f"<div class='s'>{html.escape(s)}</div></div>" for l, v, s in items) + "</div>"
