"""Reusable presentational components.

All components render escaped text into our own ``pw-*`` classes, so a user
typing HTML into a form can never inject markup.
"""
from __future__ import annotations

import base64
from html import escape

import streamlit as st

from src.utils.config import APP_NAME

# Minimal inline icon set (Lucide-style strokes) so icons are consistent and
# never emojis.
_ICONS = {
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    "warn": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
    "danger": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/>',
    "success": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "neutral": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    "up": '<path d="m5 12 7-7 7 7M12 19V5"/>',
    "down": '<path d="m19 12-7 7-7-7M12 5v14"/>',
    "dot": '<circle cx="12" cy="12" r="3"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "tag": '<path d="M12.6 2.6A2 2 0 0 0 11.2 2H4a2 2 0 0 0-2 2v7.2a2 2 0 0 0 .6 1.4l8.7 8.7a2.4 2.4 0 0 0 3.4 0l6.6-6.6a2.4 2.4 0 0 0 0-3.4z"/><circle cx="7.5" cy="7.5" r=".5"/>',
    "building": '<rect x="4" y="2" width="16" height="20" rx="2"/><path d="M9 22v-4h6v4M8 6h.01M16 6h.01M12 6h.01M12 10h.01M12 14h.01M16 10h.01M16 14h.01M8 10h.01M8 14h.01"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
}


# Each icon has one fixed colour; st.html strips <svg> and <img>, so icons are
# CSS background images declared in a stylesheet (style blocks are allowed).
_ICON_COLOR = {"info": "#1d4ed8", "warn": "#b45309", "danger": "#b42318", "success": "#0f8a65",
               "neutral": "#64748b", "up": "#0f8a65", "down": "#b45309", "dot": "#64748b",
               "check": "#0f8a65", "tag": "#b45309", "search": "#94a3b8", "building": "#ffffff"}


def icon_css() -> str:
    rules = []
    for name, color in _ICON_COLOR.items():
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
               f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>')
        data = base64.b64encode(svg.encode()).decode()
        rules.append(f'.pw-i-{name}{{background-image:url("data:image/svg+xml;base64,{data}")}}')
    return ".pw-i{display:inline-block;flex:0 0 auto;background-repeat:no-repeat;background-size:contain;margin-top:2px}" \
           ".pw-i.s16{width:16px;height:16px}.pw-i.s18{width:18px;height:18px}.pw-i.s26{width:26px;height:26px;margin:0}" \
           + "".join(rules)


def icon(name: str, size: int = 18, color: str | None = None) -> str:  # noqa: ARG001 (color fixed per icon)
    cls = {16: "s16", 18: "s18", 26: "s26"}.get(size, "s18")
    return f'<span class="pw-i pw-i-{name} {cls}" aria-hidden="true"></span>'


def row(kind: str, name: str):
    """Keyed container whose columns wrap responsively.

    kind: "card" (min 230px per column) or "chart" (min 360px per column).
    Use as ``with ui.row("chart", "dash-1"): a, b = st.columns(2)``.
    """
    return st.container(key=f"pwr-{kind}-{name}")


def page_header(title: str, subtitle: str | None = None, eyebrow: str | None = None) -> None:
    parts = ['<div class="pw-header">']
    if eyebrow:
        parts.append(f'<p class="pw-eyebrow">{escape(eyebrow)}</p>')
    parts.append(f'<h1 class="pw-title">{escape(title)}</h1>')
    if subtitle:
        parts.append(f'<p class="pw-subtitle">{escape(subtitle)}</p>')
    parts.append("</div>")
    st.html("".join(parts))


def section(title: str, caption: str | None = None) -> None:
    cap = f"<p>{escape(caption)}</p>" if caption else ""
    st.html(f'<div class="pw-section"><h3>{escape(title)}</h3>{cap}</div>')


def kpi_row(items: list[dict]) -> None:
    """items: [{label, value, sub?, tone?: 'good'|'bad'|None}]"""
    cards = []
    for it in items:
        tone = it.get("tone") or ""
        sub = f'<div class="s">{escape(str(it["sub"]))}</div>' if it.get("sub") else ""
        cards.append(f'<div class="pw-kpi {tone}"><div class="l">{escape(it["label"])}</div>'
                     f'<div class="v">{escape(str(it["value"]))}</div>{sub}</div>')
    st.html(f'<div class="pw-kpis">{"".join(cards)}</div>')


def callout(kind: str, title: str, body: str | None = None, bullets: list[str] | None = None) -> None:
    content = f"<b>{escape(title)}</b>"
    if body:
        content += escape(body)
    if bullets:
        content += "<ul>" + "".join(f"<li>{escape(b)}</li>" for b in bullets) + "</ul>"
    st.html(f'<div class="pw-callout {kind}">{icon(kind)}<div>{content}</div></div>')


def verdict(good: bool, title: str, detail: str, kicker: str = "Value assessment") -> None:
    tone = "good" if good else "bad"
    st.html(
        f'<div class="pw-verdict {tone}"><div class="badge">{icon("check" if good else "tag", 26, "#0f8a65" if good else "#b45309")}</div>'
        f'<div><div class="k">{escape(kicker)}</div><div class="t">{escape(title)}</div>'
        f'<div class="d">{escape(detail)}</div></div></div>'
    )


def factor_list(items: list[str], kind: str) -> None:
    """kind: 'up' (supports value), 'down' (weighs against), 'dot' (context)."""
    color = {"up": "#0f8a65", "down": "#b45309", "dot": "#64748b"}[kind]
    if not items:
        st.html('<p style="font-size:.85rem;color:#64748b;margin:.3rem 0">None identified.</p>')
        return
    lis = "".join(f"<li>{icon(kind, 16, color)}<span>{escape(t)}</span></li>" for t in items)
    st.html(f'<ul class="pw-list">{lis}</ul>')


def scenario_cards(results, colors: list[str]) -> None:
    with row("card", "scenarios"):
        cols = st.columns(len(results))
    for col, res, color in zip(cols, results, colors):
        with col:
            st.html(
                f'<div class="pw-scen"><div class="n"><span class="dot" style="background:{color}"></span>'
                f'{escape(res.name)} · {res.annual_rate:.1%}/yr</div>'
                f'<div class="v">₹{res.future_value:,.1f} L</div>'
                f'<div class="s">+₹{res.gain:,.1f} L ({res.gain_pct:+.1f}%) over {res.years} years</div></div>'
            )


def pill(text: str, tone: str = "") -> str:
    return f'<span class="pw-pill {tone}">{escape(text)}</span>'


def empty_state(title: str, body: str) -> None:
    st.html(f'<div class="pw-empty">{icon("search", 26, "#94a3b8")}<b>{escape(title)}</b>{escape(body)}</div>')


def card(title: str, body: str) -> None:
    st.html(f'<div class="pw-card"><h4>{escape(title)}</h4><p>{escape(body)}</p></div>')


def sidebar_brand() -> None:
    st.html(
        f'<div class="pw-brand"><div class="mark">{icon("building", 20, "#ffffff")}</div>'
        f'<div><div class="name">{escape(APP_NAME)}</div><div class="tag">Property value analytics</div></div></div>'
    )


def footer() -> None:
    st.html('<div class="pw-footer">PropWise AI is an educational analytics tool built on a static, '
            'synthetic-looking dataset. It is not financial, legal or tax advice. Verify any property '
            'independently and consult a registered adviser before investing.</div>')


def lakhs(v: float) -> str:
    return f"₹{v:,.1f} L"


def inr(v: float) -> str:
    return f"₹{v:,.0f}"
