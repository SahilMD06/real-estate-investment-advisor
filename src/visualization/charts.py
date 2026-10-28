"""Plotly chart library shared by the EDA report and the Streamlit app.

Design rules (validated with the dataviz palette checker):
* Categorical order is fixed: emerald, violet, amber. Amber is below 3:1 on
  white, so any chart using it also carries direct labels or a legend.
* Single-series charts use emerald; "context" bars use a muted slate.
* Sequential = one emerald ramp, light → dark. Diverging (correlations) =
  blue ↔ red with a neutral grey midpoint.
* Charts receive pre-aggregated data where possible so the browser never
  receives 250k raw points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

INK = "#0f172a"
INK_2 = "#475569"
MUTED = "#94a3b8"
GRID = "#e8ecf1"
SURFACE = "#ffffff"
EMERALD = "#0f8a65"
VIOLET = "#6250d6"
AMBER = "#eda100"
SLATE = "#c7d0db"
NAVY = "#0b1f3a"
CATEGORICAL = [EMERALD, VIOLET, AMBER]
SEQUENTIAL = ["#eaf6f1", "#c3e6d8", "#8fd0b6", "#55b48f", "#0f8a65", "#0b6a4e", "#074a37"]
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]
FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"

pio.templates["propwise"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, size=13, color=INK_2),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", y=0.97),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=CATEGORICAL,
        margin=dict(l=8, r=16, t=48, b=8),
        hoverlabel=dict(bgcolor=NAVY, bordercolor=NAVY, font=dict(family=FONT, color="#ffffff", size=12)),
        xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False, ticks="", automargin=True,
                   title=dict(font=dict(size=12, color=INK_2))),
        yaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False, ticks="", automargin=True,
                   title=dict(font=dict(size=12, color=INK_2))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=12), title=dict(text="")),
        bargap=0.25,
    )
)
pio.templates.default = "propwise"


def finish(fig: go.Figure, title: str | None, height: int = 360, subtitle: str | None = None) -> go.Figure:
    """Apply title/height; reserve room so a top legend never collides with the title."""
    legend_on = fig.layout.showlegend is not False and sum(
        1 for t in fig.data if t.showlegend is not False) > 1
    top = (48 if title else 16) + (26 if legend_on else 0) + (20 if subtitle and title else 0)
    title_cfg = None
    if title:
        title_cfg = dict(text=title, y=1, yref="container", yanchor="top", pad=dict(t=14))
        if subtitle:
            title_cfg["subtitle"] = dict(text=subtitle, font=dict(size=11.5, color=INK_2))
    fig.update_layout(title=title_cfg, height=height, margin=dict(t=top))
    if legend_on and title:
        fig.update_layout(legend=dict(y=1, yref="paper", yanchor="bottom", x=0, xanchor="left"))
    return fig


def _inr(v: float) -> str:
    return f"₹{v:,.0f}"


def mean_ci(df: pd.DataFrame, by: str, value: str, order: list | None = None) -> pd.DataFrame:
    g = df.groupby(by, observed=True)[value].agg(["mean", "std", "size", "median"]).reset_index()
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["size"])
    if order is not None:
        g[by] = pd.Categorical(g[by], categories=order, ordered=True)
        g = g.sort_values(by)
    return g


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
def histogram(series: pd.Series, title: str, x_title: str, bins: int = 30,
              color: str = EMERALD, marker_line: float | None = None, marker_label: str | None = None,
              height: int = 340) -> go.Figure:
    s = series.dropna().to_numpy()
    counts, edges = np.histogram(s, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(
        x=centers, y=counts, width=np.diff(edges) * 0.92, marker=dict(color=color, cornerradius=3),
        customdata=np.stack([edges[:-1], edges[1:]], axis=1),
        hovertemplate="%{customdata[0]:,.0f} – %{customdata[1]:,.0f}<br><b>%{y:,} properties</b><extra></extra>",
    ))
    med = float(np.median(s)) if len(s) else 0.0
    line_at = marker_line if marker_line is not None else med
    fig.add_vline(x=line_at, line=dict(color=NAVY, width=1.5, dash="dot"))
    fig.add_annotation(x=line_at, y=1, yref="paper", yanchor="bottom", showarrow=False,
                       text=marker_label or f"Median {med:,.0f}", font=dict(size=11, color=INK))
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="Properties")
    return finish(fig, title, height)


def price_distribution(df: pd.DataFrame) -> go.Figure:
    return histogram(df["Price_in_Lakhs"], "Distribution of asking prices", "Price (₹ lakhs)", bins=25)


def size_distribution(df: pd.DataFrame) -> go.Figure:
    return histogram(df["Size_in_SqFt"], "Distribution of property sizes", "Size (sqft)", bins=25)


def pps_outliers(df: pd.DataFrame) -> go.Figure:
    pps = df["Price_per_SqFt_INR"]
    q1, q3 = np.percentile(pps, [25, 75])
    fence = q3 + 1.5 * (q3 - q1)
    n_out = int((pps > fence).sum())
    return histogram(pps.clip(upper=60000), "Price per sqft with IQR outlier fence", "₹ per sqft (values above 60,000 shown at 60,000)",
                     bins=60, marker_line=fence,
                     marker_label=f"IQR fence ₹{fence:,.0f} · {n_out:,} above ({n_out / len(pps):.1%})")


def size_vs_price(df: pd.DataFrame, n: int = 4000, seed: int = 42) -> go.Figure:
    s = df.sample(min(n, len(df)), random_state=seed)
    fig = go.Figure(go.Scattergl(
        x=s["Size_in_SqFt"], y=s["Price_in_Lakhs"], mode="markers",
        marker=dict(size=5, color=EMERALD, opacity=0.35, line=dict(width=0)),
        hovertemplate="%{x:,} sqft<br>₹%{y:,.1f} L<extra></extra>",
    ))
    bins = pd.cut(df["Size_in_SqFt"], 18)
    trend = df.groupby(bins, observed=True)["Price_in_Lakhs"].mean()
    fig.add_trace(go.Scatter(
        x=[iv.mid for iv in trend.index], y=trend.values, mode="lines",
        line=dict(color=NAVY, width=2), name="Mean price by size bin",
        hovertemplate="Mean ₹%{y:,.1f} L<extra></extra>",
    ))
    r = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])
    fig.update_layout(showlegend=False)
    sub = f"Pearson r = {r:+.3f} · {len(s):,}-point sample · line = binned mean"
    fig.update_xaxes(title="Size (sqft)")
    fig.update_yaxes(title="Price (₹ lakhs)", rangemode="tozero")
    return finish(fig, "Size vs asking price", 380, subtitle=sub)


def box_from_quantiles(df: pd.DataFrame, by: str, value: str, title: str, y_title: str,
                       order: list | None = None, height: int = 360) -> go.Figure:
    """Box plot from precomputed quartiles (no raw points sent to the browser)."""
    cats = order or sorted(df[by].dropna().unique())
    fig = go.Figure()
    q = df.groupby(by, observed=True)[value].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    q = q.reindex(cats)
    fig.add_trace(go.Box(
        x=[str(c) for c in cats], q1=q[0.25], median=q[0.5], q3=q[0.75], lowerfence=q[0.05],
        upperfence=q[0.95], marker_color=EMERALD, fillcolor="rgba(15,138,101,0.18)",
        line=dict(color=EMERALD, width=1.5), name="", boxpoints=False,
        hoverinfo="y",
    ))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title=y_title, rangemode="tozero")
    return finish(fig, title, height, subtitle="Box = quartiles · whiskers = 5th–95th percentile")


# ---------------------------------------------------------------------------
# Category comparisons
# ---------------------------------------------------------------------------
def bar_mean_ci(df: pd.DataFrame, by: str, value: str, title: str, value_title: str,
                order: list | None = None, horizontal: bool = False, height: int = 340,
                fmt: str = ",.0f", highlight: str | None = None) -> go.Figure:
    g = mean_ci(df, by, value, order)
    labels = g[by].astype(str)
    colors = [EMERALD if (highlight is None or lab == highlight) else SLATE for lab in labels]
    common = dict(marker=dict(color=colors, cornerradius=3),
                  customdata=np.stack([g["size"], g["ci"]], axis=1))
    if horizontal:
        fig = go.Figure(go.Bar(
            y=labels, x=g["mean"], orientation="h",
            error_x=dict(type="data", array=g["ci"], color=INK_2, thickness=1, width=3),
            hovertemplate=f"%{{y}}<br><b>%{{x:{fmt}}}</b> ± %{{customdata[1]:{fmt}}} (95% CI)<br>n = %{{customdata[0]:,}}<extra></extra>",
            **common,
        ))
        fig.update_xaxes(title=value_title, rangemode="tozero")
        fig.update_yaxes(autorange="reversed")
    else:
        fig = go.Figure(go.Bar(
            x=labels, y=g["mean"],
            error_y=dict(type="data", array=g["ci"], color=INK_2, thickness=1, width=4),
            hovertemplate=f"%{{x}}<br><b>%{{y:{fmt}}}</b> ± %{{customdata[1]:{fmt}}} (95% CI)<br>n = %{{customdata[0]:,}}<extra></extra>",
            **common,
        ))
        fig.update_yaxes(title=value_title, rangemode="tozero")
    spread = (g["mean"].max() / g["mean"].min() - 1) * 100 if g["mean"].min() > 0 else float("nan")
    sub = f"Range across groups: {spread:.1f}% · bars = mean ± 95% CI"
    return finish(fig, title, height, subtitle=sub)


def count_bar(df: pd.DataFrame, by: str, title: str, order: list | None = None, height: int = 300) -> go.Figure:
    vc = df[by].value_counts()
    if order is not None:
        vc = vc.reindex(order)
    share = vc / vc.sum() * 100
    fig = go.Figure(go.Bar(
        x=vc.index.astype(str), y=vc.values, marker=dict(color=EMERALD, cornerradius=3),
        text=[f"{s:.1f}%" for s in share], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK_2, size=12),
        hovertemplate="%{x}<br><b>%{y:,} properties</b><extra></extra>",
    ))
    fig.update_yaxes(title="Properties", rangemode="tozero")
    return finish(fig, title, height)


def line_mean_ci(df: pd.DataFrame, by: str, value: str, title: str, x_title: str, value_title: str,
                 height: int = 320) -> go.Figure:
    g = mean_ci(df, by, value).sort_values(by)
    x = g[by].to_numpy()
    fig = go.Figure([
        go.Scatter(x=np.concatenate([x, x[::-1]]),
                   y=np.concatenate([g["mean"] + g["ci"], (g["mean"] - g["ci"])[::-1]]),
                   fill="toself", fillcolor="rgba(15,138,101,0.14)", line=dict(width=0),
                   hoverinfo="skip", showlegend=False),
        go.Scatter(x=x, y=g["mean"], mode="lines+markers", line=dict(color=EMERALD, width=2),
                   marker=dict(size=8, color=EMERALD, line=dict(color=SURFACE, width=2)),
                   customdata=g["size"], showlegend=False,
                   hovertemplate=f"{x_title} %{{x}}<br><b>₹%{{y:,.0f}}</b><br>n = %{{customdata:,}}<extra></extra>"),
    ])
    r = df[by].corr(df[value])
    fig.update_xaxes(title=x_title, dtick=1)
    fig.update_yaxes(title=value_title, rangemode="tozero")
    return finish(fig, title, height, subtitle=f"r = {r:+.3f} · band = 95% CI")


def ranked_bar(labels: pd.Series, values: pd.Series, title: str, value_title: str,
               customdata=None, hover: str | None = None, height: int | None = None,
               fmt: str = ",.0f") -> go.Figure:
    height = height or max(300, 22 * len(labels) + 80)
    fig = go.Figure(go.Bar(
        y=labels.astype(str), x=values, orientation="h", marker=dict(color=EMERALD, cornerradius=3),
        customdata=customdata,
        hovertemplate=hover or f"%{{y}}<br><b>%{{x:{fmt}}}</b><extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=value_title, rangemode="tozero")
    return finish(fig, title, height)


def heatmap(z: pd.DataFrame, title: str, colorbar_title: str, fmt: str = ".1f", diverging: bool = False,
            height: int | None = None, zmid: float | None = None) -> go.Figure:
    height = height or max(320, 20 * len(z.index) + 120)
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[str(c) for c in z.columns], y=[str(i) for i in z.index],
        colorscale=DIVERGING if diverging else [[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
        zmid=zmid, xgap=2, ygap=2,
        colorbar=dict(title=dict(text=colorbar_title, side="right"), thickness=10, outlinewidth=0),
        hovertemplate=f"%{{y}} · %{{x}}<br><b>%{{z:{fmt}}}</b><extra></extra>",
        # Plain list comprehension: np.vectorize can raise spurious FP-flag
        # RuntimeWarnings on some CPUs while formatting strings.
        text=[["" if pd.isna(v) else format(v, fmt) for v in row] for row in z.values] if z.size <= 150 else None,
        texttemplate="%{text}" if z.size <= 150 else None,
        textfont=dict(size=11),
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom")
    return finish(fig, title, height)


def correlation_heatmap(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    corr = df[cols].corr().round(3)
    return heatmap(corr, "Correlation among numeric variables (Pearson r)", "r", fmt="+.2f",
                   diverging=True, zmid=0, height=520)


def bhk_by_city(df: pd.DataFrame, height: int = 900) -> go.Figure:
    tab = pd.crosstab(df["City"], df["BHK"], normalize="index") * 100
    tab = tab.sort_index()
    fig = heatmap(tab, "BHK mix by city (% of each city's listings)", "% of city", fmt=".0f", height=height)
    fig.update_xaxes(title="BHK")
    return fig


def amenities_effect(df: pd.DataFrame) -> go.Figure:
    rows = []
    for col in ["has_clubhouse", "has_garden", "has_gym", "has_playground", "has_pool"]:
        with_ = df.loc[df[col] == 1, "Price_per_SqFt_INR"]
        without = df.loc[df[col] == 0, "Price_per_SqFt_INR"]
        diff = with_.median() / without.median() - 1
        rows.append({"Amenity": col.replace("has_", "").title(), "diff": diff * 100,
                     "n_with": len(with_)})
    g = pd.DataFrame(rows).sort_values("diff")
    fig = go.Figure(go.Bar(
        y=g["Amenity"], x=g["diff"], orientation="h",
        marker=dict(color=[EMERALD if d >= 0 else VIOLET for d in g["diff"]], cornerradius=3),
        text=[f"{d:+.1f}%" for d in g["diff"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK_2), customdata=g["n_with"],
        hovertemplate="%{y}<br><b>%{x:+.2f}%</b> median ₹/sqft vs without<br>n with = %{customdata:,}<extra></extra>",
    ))
    lim = max(5, g["diff"].abs().max() * 1.6)
    fig.update_xaxes(title="Median ₹/sqft difference vs properties without it (%)", range=[-lim, lim],
                     zeroline=True, zerolinecolor=MUTED)
    return finish(fig, "Price per sqft premium by individual amenity", 300)


# ---------------------------------------------------------------------------
# Analyzer & scenario charts
# ---------------------------------------------------------------------------
def cohort_position(values: np.ndarray, property_pps: float, median: float, title: str | None = None) -> go.Figure:
    v = np.asarray(values, dtype=float)
    hi = max(np.percentile(v, 97), property_pps * 1.1)
    counts, edges = np.histogram(np.clip(v, None, hi), bins=30)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [EMERALD if c <= median else SLATE for c in centers]
    fig = go.Figure(go.Bar(x=centers, y=counts, width=np.diff(edges) * 0.9,
                           marker=dict(color=colors, cornerradius=2),
                           hovertemplate="₹%{x:,.0f}/sqft<br>%{y} comparable listings<extra></extra>"))
    below = property_pps <= median
    fig.add_vline(x=median, line=dict(color=INK_2, width=1.5, dash="dot"))
    fig.add_annotation(x=median, y=1, yref="paper", yanchor="bottom", showarrow=False,
                       xanchor="left" if below else "right", xshift=4 if below else -4,
                       text=f"Cohort median {_inr(median)}", font=dict(size=11, color=INK_2))
    fig.add_vline(x=property_pps, line=dict(color=NAVY, width=2.5))
    fig.add_annotation(x=property_pps, y=1, yref="paper", yanchor="bottom", showarrow=False,
                       xanchor="right" if below else "left", xshift=-4 if below else 4,
                       text=f"<b>This property {_inr(property_pps)}</b>", font=dict(size=12, color=NAVY))
    fig.update_xaxes(title="₹ per sqft (comparable training listings)")
    fig.update_yaxes(title="Listings", showgrid=False)
    fig.update_layout(showlegend=False)
    return finish(fig, title, 300)


def scenario_trajectory(traj: pd.DataFrame, height: int = 320) -> go.Figure:
    fig = go.Figure()
    order = list(dict.fromkeys(traj["Scenario"]))
    for i, name in enumerate(order):
        d = traj[traj["Scenario"] == name]
        color = CATEGORICAL[i % len(CATEGORICAL)] if len(order) <= 3 else EMERALD
        fig.add_trace(go.Scatter(
            x=d["Year"], y=d["Value"], name=name, mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=8, color=color, line=dict(color=SURFACE, width=2)),
            hovertemplate=f"{name} · year %{{x}}<br><b>₹%{{y:,.1f}} L</b><extra></extra>",
        ))
        last = d.iloc[-1]
        fig.add_annotation(x=last["Year"], y=last["Value"], text=f"{name} ₹{last['Value']:,.1f} L",
                           showarrow=False, xanchor="left", xshift=8, font=dict(size=11, color=INK))
    fig.update_xaxes(title="Years from today", dtick=1, range=[-0.2, traj["Year"].max() + 1.9])
    fig.update_yaxes(title="Value (₹ lakhs)")
    fig.update_layout(hovermode="x unified")
    return finish(fig, None, height)


# ---------------------------------------------------------------------------
# Model-insight charts
# ---------------------------------------------------------------------------
def roc_chart(curves: dict[str, dict], title: str = "ROC curves (test split)") -> go.Figure:
    fig = go.Figure()
    for i, (name, c) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=name,
                                 line=dict(color=CATEGORICAL[i % 3], width=2),
                                 hovertemplate=f"{name}<br>FPR %{{x:.3f}} · TPR %{{y:.3f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                             hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return finish(fig, title, 380)


def calibration_chart(curves: dict[str, dict], title: str = "Calibration (test split)") -> go.Figure:
    fig = go.Figure()
    for i, (name, c) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(x=c["mean_pred"], y=c["frac_pos"], mode="lines+markers", name=name,
                                 line=dict(color=CATEGORICAL[i % 3], width=2), marker=dict(size=8),
                                 hovertemplate=f"{name}<br>predicted %{{x:.2f}} · observed %{{y:.2f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                             hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title="Mean predicted probability", range=[0, 1])
    fig.update_yaxes(title="Observed share 'good value'", range=[0, 1])
    return finish(fig, title, 380)


def confusion_chart(cm: list[list[int]], title: str) -> go.Figure:
    z = np.array(cm)
    labels = ["Above comparables", "Good value"]
    pct = z / z.sum() * 100
    text = [[f"{z[i, j]:,}<br>{pct[i, j]:.1f}%" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, text=text, texttemplate="%{text}", textfont=dict(size=13),
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)], showscale=False,
        xgap=3, ygap=3, hovertemplate="Actual %{y}<br>Predicted %{x}<br><b>%{z:,}</b><extra></extra>",
    ))
    fig.update_xaxes(title="Predicted", showgrid=False, side="bottom")
    fig.update_yaxes(title="Actual (rule label)", autorange="reversed", showgrid=False)
    return finish(fig, title, 340)


def importance_chart(records: list[dict], title: str, top: int = 12) -> go.Figure:
    d = pd.DataFrame(records).head(top)
    d["label"] = d["feature"].str.replace("_", " ")
    fig = go.Figure(go.Bar(
        y=d["label"], x=d["importance"], orientation="h", marker=dict(color=EMERALD, cornerradius=3),
        error_x=dict(type="data", array=d["std"], color=INK_2, thickness=1, width=3),
        hovertemplate="%{y}<br>AUC drop when shuffled: <b>%{x:.4f}</b><extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Mean drop in ROC-AUC when the field is shuffled")
    return finish(fig, title, max(300, 26 * len(d) + 90))


def metric_dots(table: pd.DataFrame, metric: str, title: str, x_title: str) -> go.Figure:
    """Dot plot comparing one metric across models (one axis, no dual scales)."""
    d = table.sort_values(metric)
    fig = go.Figure(go.Scatter(
        x=d[metric], y=d["Model"], mode="markers+text", text=[f"{v:.3f}" for v in d[metric]],
        textposition="middle right", textfont=dict(color=INK_2, size=12),
        marker=dict(size=12, color=EMERALD, line=dict(color=SURFACE, width=2)),
        hovertemplate="%{y}<br><b>%{x:.4f}</b><extra></extra>",
    ))
    lo, hi = float(d[metric].min()), float(d[metric].max())
    pad = max((hi - lo) * 0.35, 0.01)
    fig.update_xaxes(title=x_title, range=[lo - pad, hi + pad * 1.6])
    return finish(fig, title, max(240, 44 * len(d) + 100))


def actual_vs_predicted(actual, predicted, title: str) -> go.Figure:
    a, p = np.asarray(actual), np.asarray(predicted)
    lo, hi = float(min(a.min(), p.min())), float(max(a.max(), p.max()))
    fig = go.Figure([
        go.Scattergl(x=a, y=p, mode="markers", marker=dict(size=5, color=EMERALD, opacity=0.4),
                     hovertemplate="Actual ₹%{x:,.1f} L<br>Predicted ₹%{y:,.1f} L<extra></extra>", showlegend=False),
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                   hoverinfo="skip", showlegend=False),
    ])
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
                       text="Dotted line = perfect prediction", font=dict(size=11, color=INK_2))
    fig.update_xaxes(title="Actual price (₹ lakhs)")
    fig.update_yaxes(title="Predicted price (₹ lakhs)")
    return finish(fig, title, 380)


def residual_hist(counts: list[int], edges: list[float], title: str) -> go.Figure:
    e = np.asarray(edges)
    fig = go.Figure(go.Bar(x=(e[:-1] + e[1:]) / 2, y=counts, width=np.diff(e) * 0.9,
                           marker=dict(color=EMERALD, cornerradius=2),
                           hovertemplate="Residual ≈ ₹%{x:,.0f} L<br>%{y:,} properties<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=NAVY, width=1.5, dash="dot"))
    fig.update_xaxes(title="Residual: actual − predicted (₹ lakhs)")
    fig.update_yaxes(title="Properties")
    return finish(fig, title, 340)
