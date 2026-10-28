"""Design tokens and the single stylesheet injected on every page.

Most theming lives in .streamlit/config.toml (colours, font, radii, sidebar).
This CSS only styles our own ``pw-*`` component classes, plus two layout tweaks
on stable ``data-testid`` hooks, so it does not depend on Streamlit's internal
class names (which change between versions).
"""
from __future__ import annotations

import streamlit as st

TOKENS = {
    "ink": "#0f172a",
    "ink2": "#475569",
    "muted": "#64748b",
    "line": "#e2e8f0",
    "surface": "#ffffff",
    "canvas": "#f6f8fb",
    "accent": "#0f8a65",
    "accent_soft": "#e7f5ef",
    "navy": "#0b1f3a",
    "warn": "#b45309",
    "warn_soft": "#fff7e6",
    "danger": "#b42318",
    "danger_soft": "#fdecea",
    "info": "#1d4ed8",
    "info_soft": "#eef4ff",
}

_CSS = """
<style>
:root{
  --pw-ink:%(ink)s;--pw-ink2:%(ink2)s;--pw-muted:%(muted)s;--pw-line:%(line)s;--pw-surface:%(surface)s;
  --pw-accent:%(accent)s;--pw-accent-soft:%(accent_soft)s;--pw-navy:%(navy)s;
  --pw-warn:%(warn)s;--pw-warn-soft:%(warn_soft)s;--pw-danger:%(danger)s;--pw-danger-soft:%(danger_soft)s;
  --pw-info:%(info)s;--pw-info-soft:%(info_soft)s;
}
[data-testid="stMainBlockContainer"]{max-width:1240px;padding-top:2.2rem;padding-bottom:3rem}
/* Responsive rows: wrap instead of squeezing on medium-width screens.
   Card / chart rows live in keyed containers (st-key-pwr-*) with larger minimums. */
[data-testid="stHorizontalBlock"]{flex-wrap:wrap;row-gap:.75rem}
[data-testid="stColumn"]{min-width:min(120px,100%%)}
[class*="st-key-pwr-card"] [data-testid="stColumn"]{min-width:min(230px,100%%)}
[class*="st-key-pwr-chart"] [data-testid="stColumn"]{min-width:min(360px,100%%)}
[data-testid="stSidebarHeader"] img{height:2.4rem;max-width:100%%}

.pw-eyebrow{font-size:.72rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--pw-accent);margin:0 0 .25rem}
.pw-title{font-size:1.65rem;font-weight:650;letter-spacing:-.015em;color:var(--pw-ink);margin:0;line-height:1.25}
.pw-subtitle{font-size:.95rem;color:var(--pw-ink2);margin:.35rem 0 0;max-width:780px}
.pw-header{margin-bottom:1.1rem}

.pw-section{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin:1.4rem 0 .5rem}
.pw-section h3{font-size:1.02rem;font-weight:620;color:var(--pw-ink);margin:0;padding:0}
.pw-section p{font-size:.82rem;color:var(--pw-muted);margin:0}

.pw-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:.25rem 0 .5rem}
.pw-kpi{background:var(--pw-surface);border:1px solid var(--pw-line);border-radius:12px;padding:14px 16px;min-height:96px;display:flex;flex-direction:column;justify-content:flex-start;gap:2px}
.pw-kpi .l{font-size:.74rem;font-weight:550;color:var(--pw-muted);text-transform:uppercase;letter-spacing:.05em}
.pw-kpi .v{font-size:1.45rem;font-weight:650;color:var(--pw-ink);font-variant-numeric:tabular-nums;letter-spacing:-.01em;line-height:1.2;margin-top:4px}
.pw-kpi .s{font-size:.78rem;color:var(--pw-ink2);margin-top:2px}
.pw-kpi.good .v{color:var(--pw-accent)}
.pw-kpi.bad .v{color:var(--pw-warn)}

.pw-callout{display:flex;gap:12px;align-items:flex-start;border-radius:10px;padding:12px 14px;border:1px solid;margin:.4rem 0 .8rem;font-size:.88rem;line-height:1.5}
.pw-callout svg{flex:0 0 18px;margin-top:2px}
.pw-callout b{display:block;margin-bottom:2px;font-weight:620}
.pw-callout.info{background:var(--pw-info-soft);border-color:#c7d7fe;color:#1e3a8a}
.pw-callout.warn{background:var(--pw-warn-soft);border-color:#fcd9a0;color:#7a3e05}
.pw-callout.danger{background:var(--pw-danger-soft);border-color:#f5c2bd;color:#7a1a12}
.pw-callout.success{background:var(--pw-accent-soft);border-color:#b7e2d0;color:#0b5a43}
.pw-callout.neutral{background:#f8fafc;border-color:var(--pw-line);color:var(--pw-ink2)}
.pw-callout ul{margin:.3rem 0 0 1rem;padding:0}

.pw-verdict{border-radius:14px;padding:20px 22px;border:1px solid var(--pw-line);background:var(--pw-surface);display:flex;gap:18px;align-items:center}
.pw-verdict .badge{flex:0 0 52px;height:52px;border-radius:12px;display:flex;align-items:center;justify-content:center}
.pw-verdict.good .badge{background:var(--pw-accent-soft);color:var(--pw-accent)}
.pw-verdict.bad .badge{background:var(--pw-warn-soft);color:var(--pw-warn)}
.pw-verdict .k{font-size:.74rem;text-transform:uppercase;letter-spacing:.07em;color:var(--pw-muted);font-weight:600}
.pw-verdict .t{font-size:1.35rem;font-weight:650;color:var(--pw-ink);margin:2px 0}
.pw-verdict .d{font-size:.9rem;color:var(--pw-ink2)}

.pw-card{background:var(--pw-surface);border:1px solid var(--pw-line);border-radius:12px;padding:16px 18px;height:100%%}
.pw-card h4{font-size:.95rem;font-weight:620;margin:0 0 .35rem;color:var(--pw-ink)}
.pw-card p{font-size:.86rem;color:var(--pw-ink2);margin:0}
.pw-scen{background:var(--pw-surface);border:1px solid var(--pw-line);border-radius:12px;padding:14px 16px}
.pw-scen .n{font-size:.78rem;font-weight:600;color:var(--pw-muted);text-transform:uppercase;letter-spacing:.05em;display:flex;align-items:center;gap:6px}
.pw-scen .dot{width:9px;height:9px;border-radius:50%%;display:inline-block}
.pw-scen .v{font-size:1.4rem;font-weight:650;color:var(--pw-ink);font-variant-numeric:tabular-nums;margin-top:4px}
.pw-scen .s{font-size:.8rem;color:var(--pw-ink2)}

.pw-list{list-style:none;margin:0;padding:0}
.pw-list li{display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-bottom:1px solid var(--pw-line);font-size:.88rem;color:var(--pw-ink2)}
.pw-list li:last-child{border-bottom:0}
.pw-list svg{flex:0 0 16px;margin-top:2px}

.pw-pill{display:inline-flex;align-items:center;gap:6px;font-size:.75rem;font-weight:600;border-radius:999px;padding:3px 10px;border:1px solid var(--pw-line);color:var(--pw-ink2);background:#fff}
.pw-pill.good{color:var(--pw-accent);border-color:#b7e2d0;background:var(--pw-accent-soft)}
.pw-pill.warn{color:var(--pw-warn);border-color:#fcd9a0;background:var(--pw-warn-soft)}

.pw-empty{text-align:center;border:1px dashed #cbd5e1;border-radius:12px;padding:36px 16px;background:#fff;color:var(--pw-ink2)}
.pw-empty b{display:block;color:var(--pw-ink);font-size:1rem;margin:8px 0 4px}

.pw-brand{display:flex;align-items:center;gap:10px;padding:2px 0 10px}
.pw-brand .mark{width:34px;height:34px;border-radius:9px;background:#0f8a65;display:flex;align-items:center;justify-content:center}
.pw-brand .name{font-weight:650;font-size:1.05rem;color:#ffffff;letter-spacing:-.01em;line-height:1.1}
.pw-brand .tag{font-size:.72rem;color:#9fb3c8}
.pw-side-note{font-size:.74rem;color:#9fb3c8;line-height:1.5;border-top:1px solid #1f3a5f;padding-top:10px;margin-top:6px}

.pw-footer{margin-top:2.5rem;padding-top:12px;border-top:1px solid var(--pw-line);font-size:.75rem;color:var(--pw-muted)}
@media (max-width: 640px){
  .pw-title{font-size:1.35rem}
  .pw-verdict{flex-direction:column;align-items:flex-start}
  .pw-kpi .v{font-size:1.25rem}
}
</style>
""" % TOKENS


def inject_css() -> None:
    from src.ui.components import icon_css
    st.html(_CSS.replace("</style>", icon_css() + "</style>"))
