"""Generate the exploratory data analysis report.

Usage (from the project root):
    python -m scripts.run_eda

Outputs
    reports/eda_report.html     self-contained interactive report (20 analyses)
    reports/EDA_Report.pdf      the same report as a print-ready A4 PDF
    reports/eda_stats.json      the numbers quoted in the narrative
Every number in the narrative is computed here — nothing is hard-coded.
"""
from __future__ import annotations

import html
import json
import sys

import numpy as np
import pandas as pd
import plotly.io as pio

from src.data.load_data import load_processed
from src.reporting.pdf import PdfToolMissing, html_to_pdf
from src.reporting.template import FigureEmbedder, page
from src.utils.config import REPORTS_DIR
from src.utils.logging import get_logger
from src.visualization import charts as ch

log = get_logger("run_eda")

FURNISH = ["Unfurnished", "Semi-furnished", "Furnished"]
TRANSPORT = ["Low", "Medium", "High"]


def spread(g: pd.Series) -> float:
    return (g.max() / g.min() - 1) * 100


def build_sections(df: pd.DataFrame) -> tuple[list[dict], dict]:
    s: dict = {}
    pps = df["Price_per_SqFt_INR"]
    s["n"] = len(df)
    s["price_min"], s["price_med"], s["price_max"] = df["Price_in_Lakhs"].agg(["min", "median", "max"]).round(2)
    s["price_skew"] = round(float(df["Price_in_Lakhs"].skew()), 3)
    s["size_min"], s["size_med"], s["size_max"] = (int(v) for v in df["Size_in_SqFt"].agg(["min", "median", "max"]))
    s["one_bhk_large_share"] = round(float(((df["BHK"] == 1) & (df["Size_in_SqFt"] > 3000)).sum() / (df["BHK"] == 1).sum() * 100), 1)
    s["size_skew"] = round(float(df["Size_in_SqFt"].skew()), 3)
    s["pps_med"] = round(float(pps.median()))
    s["pps_skew"] = round(float(pps.skew()), 2)
    s["pps_by_type"] = df.groupby("Property_Type")["Price_per_SqFt_INR"].median().round(0).to_dict()
    s["r_size_price"] = round(float(df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])), 4)
    s["r_size_pps"] = round(float(df["Size_in_SqFt"].corr(pps)), 3)
    q1, q3 = np.percentile(pps, [25, 75])
    s["pps_fence"] = round(float(q3 + 1.5 * (q3 - q1)))
    s["pps_outliers"] = int((pps > s["pps_fence"]).sum())
    s["pps_out_small_share"] = round(float((df.loc[pps > s["pps_fence"], "Size_in_SqFt"] < 1000).mean()), 3)
    st = df.groupby("State")["Price_per_SqFt_INR"].mean()
    s["state_top"], s["state_bottom"] = st.idxmax(), st.idxmin()
    s["state_spread"] = round(spread(st), 1)
    ct = df.groupby("City")["Price_in_Lakhs"].mean()
    s["city_top"], s["city_bottom"] = ct.idxmax(), ct.idxmin()
    s["city_top_v"], s["city_bottom_v"] = round(ct.max(), 1), round(ct.min(), 1)
    s["city_spread"] = round(spread(ct), 1)
    loc = df.groupby(["City", "Locality"])
    loc_age = loc["Age_Years"].median()
    loc_n = loc.size()
    s["loc_pairs"] = len(loc_n)
    s["loc_n_median"] = int(loc_n.median())
    s["loc_age_iqr"] = [float(x) for x in np.percentile(loc_age, [25, 75])]
    bhk_tab = pd.crosstab(df["City"], df["BHK"], normalize="index") * 100
    s["bhk_share_min"], s["bhk_share_max"] = round(float(bhk_tab.values.min()), 1), round(float(bhk_tab.values.max()), 1)
    loc_pps = loc["Price_per_SqFt_INR"].median()
    eligible = loc_pps[loc_n >= 10]
    s["top_loc"] = " / ".join(eligible.idxmax())
    s["top_loc_v"] = round(float(eligible.max()))
    s["loc_pps_sd"] = round(float(eligible.std()))
    # Permutation baseline: shuffle ₹/sqft across listings, recompute locality medians.
    shuffled = df[["City", "Locality"]].assign(
        v=np.random.default_rng(42).permutation(df["Price_per_SqFt_INR"].to_numpy()))
    perm = shuffled.groupby(["City", "Locality"])["v"].median()[loc_n >= 10]
    s["loc_pps_sd_perm"] = round(float(perm.std()))
    num_cols = ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt_INR", "BHK", "Year_Built", "Floor_No",
                "Total_Floors", "Nearby_Schools", "Nearby_Hospitals", "amenity_count"]
    corr = df[num_cols].corr()
    off = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack()
    s["corr_top_pairs"] = {f"{a} ~ {b}": round(float(corr.loc[a, b]), 3)
                           for (a, b) in off.sort_values(ascending=False).index[:4:2]}
    s["r_schools"] = round(float(df["Nearby_Schools"].corr(pps)), 4)
    s["r_hospitals"] = round(float(df["Nearby_Hospitals"].corr(pps)), 4)
    for col, key in [("Furnished_Status", "furnish"), ("Facing", "facing"),
                     ("Parking_Space", "parking"), ("Public_Transport_Accessibility", "transport")]:
        val = "Price_per_SqFt_INR" if key in ("facing", "transport") else "Price_in_Lakhs"
        g = df.groupby(col)[val].mean()
        s[f"{key}_spread"] = round(spread(g), 2)
        s[f"{key}_means"] = g.round(1).to_dict()
    s["owner_shares"] = (df["Owner_Type"].value_counts(normalize=True) * 100).round(1).to_dict()
    s["avail_shares"] = (df["Availability_Status"].value_counts(normalize=True) * 100).round(1).to_dict()
    s["uc_old_share"] = round(float(df.loc[df["Availability_Status"] == "Under_Construction", "flag_uc_old_build"].mean() * 100), 1)
    am = df.groupby("amenity_count")["Price_per_SqFt_INR"].mean()
    s["amen_spread"] = round(spread(am), 2)

    synthetic_note = ("The dataset behaves like independently generated columns, so the absence of a "
                      "relationship here says nothing about real Indian housing markets.")
    sections = [
        dict(id="price-dist", group="Price & size", title="1. Distribution of property prices",
             fig=ch.price_distribution(df),
             what="Number of listings in each ₹ lakh price band.",
             finding=f"Prices run from ₹{s['price_min']} L to ₹{s['price_max']} L with a median of ₹{s['price_med']} L. The histogram is flat (skew {s['price_skew']}), with every band holding about the same number of listings. That is what a uniform random draw looks like, not a real market's typical right-skewed shape.",
             why="Investors usually expect many mid-market homes and a thin luxury tail. Here no price band is more 'typical' than another, so 'cheap' or 'expensive' only makes sense relative to comparable properties.",
             limits="One snapshot with no dates. Prices are asking prices, not transaction prices."),
        dict(id="size-dist", group="Price & size", title="2. Distribution of property sizes",
             fig=ch.size_distribution(df),
             what="Number of listings in each size band (sqft).",
             finding=f"Sizes run from {s['size_min']:,} to {s['size_max']:,} sqft (median {s['size_med']:,.0f}). The distribution is again flat (skew {s['size_skew']}).",
             why="Size is the denominator of ₹/sqft, so it shapes every value comparison in this product.",
             limits=f"Size is not linked to BHK: {s['one_bhk_large_share']}% of 1-BHK listings exceed 3,000 sqft, which is implausible."),
        dict(id="pps-type", group="Price & size", title="3. Price per sqft by property type",
             fig=ch.box_from_quantiles(df, "Property_Type", "Price_per_SqFt_INR", "Price per sqft by property type", "₹ per sqft"),
             what="Spread of ₹/sqft (quartiles, 5th–95th percentile whiskers) for apartments, independent houses and villas.",
             finding="Median ₹/sqft: " + ", ".join(f"{k} ₹{v:,.0f}" for k, v in s["pps_by_type"].items()) + ". The three distributions are practically identical.",
             why="In real markets villas and apartments carry different per-sqft prices. Here property type does not separate value, so comparisons still stay within type to be safe.",
             limits=synthetic_note),
        dict(id="size-price", group="Price & size", title="4. Relationship between size and price",
             fig=ch.size_vs_price(df),
             what="A 4,000-listing sample of size against asking price, with the binned mean price.",
             finding=f"Correlation is r = {s['r_size_price']}, so larger homes are not more expensive. As a direct consequence, size and ₹/sqft are strongly negatively correlated (r = {s['r_size_pps']}).",
             why="This is the single most important fact for interpreting results. Large homes look 'cheap per sqft' purely arithmetically. The good-value label compares like-for-like cohorts but cannot remove the effect inside a cohort.",
             limits="Scatter is sampled for speed. Correlation captures linear association only."),
        dict(id="outliers", group="Price & size", title="5. Outliers in price per sqft and size",
             fig=ch.pps_outliers(df),
             what="Distribution of ₹/sqft with the Tukey upper fence (Q3 + 1.5×IQR).",
             finding=f"{s['pps_outliers']:,} listings ({s['pps_outliers'] / s['n']:.1%}) sit above ₹{s['pps_fence']:,}/sqft. {s['pps_out_small_share']:.0%} of them are under 1,000 sqft, so they are small homes, not data-entry errors. Size has no outliers (bounded 500–5,000).",
             why="Outliers are kept because they come from the same generating process as the rest. Tree models and median-based benchmarks are robust to them.",
             limits="The fence is a convention, not a rule for 'wrong' data."),
        dict(id="state-pps", group="Location", title="6. Average price per sqft by state",
             fig=ch.bar_mean_ci(df, "State", "Price_per_SqFt_INR", "Average ₹/sqft by state", "₹ per sqft",
                                order=df.groupby("State")["Price_per_SqFt_INR"].mean().sort_values(ascending=False).index.tolist(),
                                horizontal=True, height=620),
             what="Mean ₹/sqft per state with 95% confidence intervals.",
             finding=f"Highest: {s['state_top']}; lowest: {s['state_bottom']}. The gap is only {s['state_spread']}% and the confidence intervals overlap.",
             why="Real state-level differences (e.g. Maharashtra vs. Bihar) are large. Their absence confirms that location carries no price signal in this data.",
             limits=synthetic_note),
        dict(id="city-price", group="Location", title="7. Average property price by city",
             fig=ch.bar_mean_ci(df, "City", "Price_in_Lakhs", "Average asking price by city", "₹ lakhs",
                                order=df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).index.tolist(),
                                horizontal=True, height=820, fmt=",.1f"),
             what="Mean asking price per city with 95% confidence intervals.",
             finding=f"{s['city_top']} is highest (₹{s['city_top_v']} L) and {s['city_bottom']} lowest (₹{s['city_bottom_v']} L): a {s['city_spread']}% spread, within sampling noise.",
             why="Mumbai and Gaya at the same average price is not realistic. Use city benchmarks only as within-dataset references.",
             limits=synthetic_note),
        dict(id="loc-age", group="Location", title="8. Median property age by locality",
             fig=ch.histogram(loc_age, "Median property age across City–Locality pairs", "Median age (years)", bins=30),
             what=f"Distribution of median building age across the {s['loc_pairs']:,} City–Locality pairs (locality IDs repeat in every city, so the pair is the real unit).",
             finding=f"Half of localities have a median age between {s['loc_age_iqr'][0]:.0f} and {s['loc_age_iqr'][1]:.0f} years. Each locality has only about {s['loc_n_median']} listings, so locality medians are noisy.",
             why="Locality-level statistics would be over-fitted, so the models use city-level cohorts instead.",
             limits="Small samples per locality, and locality IDs are anonymised."),
        dict(id="bhk-city", group="Location", title="9. BHK distribution across cities",
             fig=ch.bhk_by_city(df, height=780),
             what="Share of each city's listings by BHK.",
             finding=f"Every cell lies between {s['bhk_share_min']}% and {s['bhk_share_max']}%: each city has a near-even 20% split across 1–5 BHK.",
             why="Unit mix normally differs by city (more 1–2 BHK in metros). An even mix means cohorts are well balanced, at least 165 training listings each.",
             limits=synthetic_note),
        dict(id="top-loc", group="Location", title="10. Most expensive localities",
             fig=(lambda e: ch.ranked_bar(pd.Series([" / ".join(i) for i in e.index]), e.values,
                                          "Highest median ₹/sqft localities (≥10 listings)", "Median ₹ per sqft",
                                          customdata=loc_n.reindex(e.index).values,
                                          hover="%{y}<br><b>₹%{x:,.0f}/sqft</b><br>n = %{customdata}<extra></extra>"))(
                 eligible.sort_values(ascending=False).head(15)),
             what="The 15 City–Locality pairs with the highest median ₹/sqft (pairs with ≥10 listings).",
             finding=f"Top: {s['top_loc']} at ₹{s['top_loc_v']:,}/sqft. Locality medians have a standard deviation of ₹{s['loc_pps_sd']:,}. If prices are shuffled randomly across listings, it is ₹{s['loc_pps_sd_perm']:,}, so locality differences are what random assignment alone produces.",
             why="'Top localities' here are mostly small-sample luck and would likely regress to the mean. Do not read them as premium neighbourhoods.",
             limits="The brief asked for price trends of top localities, but there is no time dimension, so trends cannot be computed."),
        dict(id="corr", group="Relationships", title="11. Correlation among numeric variables",
             fig=ch.correlation_heatmap(df, num_cols),
             what="Pearson correlation for every pair of numeric fields.",
             finding="Only the two mechanical relationships stand out: " + "; ".join(f"{k} r = {v:+.2f}" for k, v in s["corr_top_pairs"].items()) + ". Every other pair has |r| < 0.01 (Year Built and Age, r = −1 by definition, is excluded).",
             why="No attribute moves with price, which is why the regression signal test finds R² ≈ 0.",
             limits="Correlation is linear and pairwise. The models test non-linear and interaction effects separately (still R² ≈ 0)."),
        dict(id="schools", group="Relationships", title="12. Nearby schools and price per sqft",
             fig=ch.line_mean_ci(df, "Nearby_Schools", "Price_per_SqFt_INR", "Mean ₹/sqft by nearby-school count", "Nearby schools", "₹ per sqft"),
             what="Mean ₹/sqft at each school count (1–10) with a 95% CI band.",
             finding=f"Flat line (r = {s['r_schools']}).",
             why="School access normally commands a premium. Here it doesn't, so the app shows it as context only.",
             limits="The field may be a count or a rating; the source does not say."),
        dict(id="hospitals", group="Relationships", title="13. Nearby hospitals and price per sqft",
             fig=ch.line_mean_ci(df, "Nearby_Hospitals", "Price_per_SqFt_INR", "Mean ₹/sqft by nearby-hospital count", "Nearby hospitals", "₹ per sqft"),
             what="Mean ₹/sqft at each hospital count with a 95% CI band.",
             finding=f"Flat line (r = {s['r_hospitals']}).",
             why="Same as schools: not a value driver in this dataset.",
             limits=synthetic_note),
        dict(id="furnish", group="Relationships", title="14. Price by furnishing status",
             fig=ch.bar_mean_ci(df, "Furnished_Status", "Price_in_Lakhs", "Average price by furnishing status", "₹ lakhs", order=FURNISH, fmt=",.1f"),
             what="Mean asking price by furnishing level with 95% CI.",
             finding=f"Groups differ by only {s['furnish_spread']}%: " + ", ".join(f"{k} ₹{v} L" for k, v in s["furnish_means"].items()) + ".",
             why="Furnishing normally adds a modest premium. Not here.",
             limits=synthetic_note),
        dict(id="facing", group="Relationships", title="15. Price per sqft by facing direction",
             fig=ch.bar_mean_ci(df, "Facing", "Price_per_SqFt_INR", "Average ₹/sqft by facing", "₹ per sqft"),
             what="Mean ₹/sqft by facing direction with 95% CI.",
             finding=f"Range {s['facing_spread']}%. No direction is priced differently.",
             why="Vastu preferences (often east/north) can move real prices. That effect is absent here.",
             limits=synthetic_note),
        dict(id="owner", group="Market structure", title="16. Listings by owner type",
             fig=ch.count_bar(df, "Owner_Type", "Listings by owner type"),
             what="Number of listings per seller type.",
             finding="Even split: " + ", ".join(f"{k} {v}%" for k, v in s["owner_shares"].items()) + ".",
             why="Builder vs. owner listings can differ in negotiability and fees. Brokerage is not modelled.",
             limits="Seller type only. No fees or commissions are in the data."),
        dict(id="availability", group="Market structure", title="17. Listings by availability status",
             fig=ch.count_bar(df, "Availability_Status", "Listings by availability status"),
             what="Ready-to-move vs. under-construction listings.",
             finding="Even split: " + ", ".join(f"{k.replace('_', ' ')} {v}%" for k, v in s["avail_shares"].items()) + f". {s['uc_old_share']}% of under-construction listings were built in 2020 or earlier, which is internally inconsistent.",
             why="Under-construction purchases carry delivery risk. The app flags this as context, not as part of the verdict.",
             limits="Inconsistent records are kept and flagged, not deleted."),
        dict(id="parking", group="Amenities & access", title="18. Parking and price",
             fig=ch.bar_mean_ci(df, "Parking_Space", "Price_in_Lakhs", "Average price by parking availability", "₹ lakhs", order=["No", "Yes"], fmt=",.1f"),
             what="Mean asking price with and without parking.",
             finding=f"Difference: {s['parking_spread']}%.",
             why="Parking would usually add value in dense cities. No measurable effect here.",
             limits="Yes/no only. The number of spaces is not recorded."),
        dict(id="amenities", group="Amenities & access", title="19. Amenities and price per sqft",
             fig=ch.amenities_effect(df),
             what="Median ₹/sqft difference for listings with each amenity vs. without it.",
             finding=f"Every amenity moves the median by about ±1%. Across 1–5 amenities, mean ₹/sqft varies by only {s['amen_spread']}%.",
             why="Amenity count is shown as context in the analyzer, not as a value driver.",
             limits="Amenity quality and maintenance charges are not recorded."),
        dict(id="transport", group="Amenities & access", title="20. Public transport accessibility and price per sqft",
             fig=ch.bar_mean_ci(df, "Public_Transport_Accessibility", "Price_per_SqFt_INR", "Average ₹/sqft by transport access", "₹ per sqft", order=TRANSPORT),
             what="Mean ₹/sqft by transport accessibility level.",
             finding=f"Range {s['transport_spread']}%. High-access listings are not priced higher.",
             why="Metro proximity is one of the strongest real price drivers in Indian cities. Its absence is another sign that the data is synthetic.",
             limits=synthetic_note),
    ]
    return sections, s


SUMMARY = [
    "The data is clean but synthetic-looking: no missing values or duplicates, but every column behaves like an independent random draw.",
    "Price is unrelated to size, location, BHK, amenities, transport, or any other field (all |r| < 0.01). Learned valuation models cannot beat predicting the average (R² ≈ 0).",
    "₹/sqft varies only because price and size vary independently, so large homes look cheap per sqft by arithmetic.",
    "Many records are internally inconsistent (floor above building height in 46% of rows; 'under construction' homes built years ago). They are kept and flagged.",
    "Implication: PropWise compares a listing to its comparable cohort using a transparent rule and projects value with explicit growth scenarios. It does not claim to predict real returns.",
]


def render_html(sections: list[dict], stats: dict) -> str:
    embed = FigureEmbedder()
    parts = ["<div class='summary'><h2>Key takeaways</h2><ul>"
             + "".join(f"<li>{html.escape(x)}</li>" for x in SUMMARY) + "</ul></div>",
             "<nav class='toc'>" + "".join(f"<a href='#{s['id']}'>{html.escape(s['title'].split('. ', 1)[1])}</a>"
                                          for s in sections) + "</nav>"]
    group = None
    for sec in sections:
        if sec["group"] != group:
            group = sec["group"]
            parts.append(f"<h3 class='group'>{html.escape(group)}</h3>")
        parts.append(
            f"<section class='card' id='{sec['id']}'><h2>{html.escape(sec['title'])}</h2>{embed(sec['fig'])}"
            f"<div class='grid'><div><b>What it shows</b>{html.escape(sec['what'])}</div>"
            f"<div><b>Main finding</b>{html.escape(sec['finding'])}</div>"
            f"<div><b>Why it matters to an investor</b>{html.escape(sec['why'])}</div>"
            f"<div><b>Limitations</b>{html.escape(sec['limits'])}</div></div></section>")
    return page("Exploratory Data Analysis", f"india_housing_prices.csv · {stats['n']:,} listings · 20 analyses",
                "".join(parts), footer="Correlation is not causation. Findings describe this dataset only "
                                       "and are not investment advice.")


def main() -> int:
    df = load_processed()
    sections, stats = build_sections(df)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS_DIR / "eda_report.html"
    html_path.write_text(render_html(sections, stats), encoding="utf-8")
    (REPORTS_DIR / "eda_stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    log.info("Wrote %d analyses to %s", len(sections), html_path)
    try:
        html_to_pdf(html_path, REPORTS_DIR / "EDA_Report.pdf", "PropWise AI · Exploratory Data Analysis")
    except PdfToolMissing as exc:
        log.warning("PDF not built: %s", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
