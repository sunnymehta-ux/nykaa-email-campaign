from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.email_campaign_pipeline import OUTPUTS_DIR, PROCESSED_DATA_PATH, run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
RECOMMENDATION_PATH = OUTPUTS_DIR / "campaign_recommendations.md"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"
TOP_COMBOS_PATH = OUTPUTS_DIR / "top_strategy_combos.csv"
MONTHLY_SUMMARY_PATH = OUTPUTS_DIR / "monthly_summary.csv"


st.set_page_config(
    page_title="Nykaa Email Campaign Dashboard",
    page_icon="N",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(233, 30, 99, 0.12), transparent 28%),
            linear-gradient(180deg, #fff8fb 0%, #ffffff 52%, #fff2f7 100%);
    }
    .hero-card {
        padding: 1.3rem 1.4rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #23121c 0%, #4a1832 100%);
        color: #ffffff;
        border: 1px solid rgba(233, 30, 99, 0.15);
        box-shadow: 0 18px 48px rgba(74, 24, 50, 0.16);
        margin-bottom: 1rem;
    }
    .hero-card h1 {
        margin: 0;
        font-size: 2.1rem;
    }
    .hero-card p {
        margin-bottom: 0;
        color: rgba(255, 255, 255, 0.82);
    }
    .insight-chip {
        padding: 0.75rem 0.9rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(35, 18, 28, 0.08);
        min-height: 104px;
    }
    .section-note {
        color: #5f4b57;
        font-size: 0.96rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_pct(value: float) -> str:
    return f"{value:.1%}"


def format_num(value: float) -> str:
    return f"{value:,.0f}"


def ensure_outputs() -> None:
    required_paths = [PROCESSED_DATA_PATH, METRICS_PATH, TOP_COMBOS_PATH, MONTHLY_SUMMARY_PATH, RECOMMENDATION_PATH]
    if not all(path.exists() for path in required_paths):
        run_pipeline()


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, str]:
    ensure_outputs()
    email_df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["Date"])
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    top_combos = pd.read_csv(TOP_COMBOS_PATH)
    monthly_summary = pd.read_csv(MONTHLY_SUMMARY_PATH)
    recommendation_text = RECOMMENDATION_PATH.read_text(encoding="utf-8")
    return email_df, metrics, top_combos, monthly_summary, recommendation_text


def build_line_chart(monthly_summary: pd.DataFrame) -> alt.Chart:
    chart_df = monthly_summary.copy()
    return (
        alt.Chart(chart_df)
        .mark_line(point=alt.OverlayMarkDef(size=70, filled=True, color="#e91e63"), strokeWidth=4, color="#ad1457")
        .encode(
            x=alt.X("Month:N", sort=list(chart_df["Month"])),
            y=alt.Y("Total_Revenue:Q", title="Total revenue"),
            tooltip=["Month", alt.Tooltip("Total_Revenue:Q", format=",.0f"), alt.Tooltip("Avg_ROI:Q", format=".2f")],
        )
        .properties(height=320)
    )


def build_bar_chart(df: pd.DataFrame, x_field: str, y_field: str, title: str, color: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color=color)
        .encode(
            x=alt.X(f"{x_field}:N", sort="-y", title=x_field.replace("_", " ")),
            y=alt.Y(f"{y_field}:Q", title=y_field.replace("_", " ")),
            tooltip=[x_field, alt.Tooltip(f"{y_field}:Q", format=".2f")],
        )
        .properties(height=320, title=title)
    )


def build_heatmap(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("Language:N"),
            y=alt.Y("Target_Audience:N", sort="-x"),
            color=alt.Color("Avg_ROI:Q", scale=alt.Scale(scheme="redpurple"), title="Avg ROI"),
            tooltip=[
                "Target_Audience",
                "Language",
                alt.Tooltip("Campaigns:Q", format=","),
                alt.Tooltip("Avg_ROI:Q", format=".2f"),
                alt.Tooltip("Avg_Revenue:Q", format=",.0f"),
            ],
        )
        .properties(height=300)
    )


email_df, metrics, top_combos, monthly_summary, recommendation_text = load_data()

st.markdown(
    """
    <div class="hero-card">
        <h1>Nykaa Email Campaign Command Center</h1>
        <p>
            Interactive summary of campaign performance, audience-language fit, ROI drivers,
            and recommended launch strategy built from the Nykaa campaign dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Dashboard Filters")
    selected_audiences = st.multiselect(
        "Target audience",
        sorted(email_df["Target_Audience"].unique()),
        default=sorted(email_df["Target_Audience"].unique()),
    )
    selected_languages = st.multiselect(
        "Language",
        sorted(email_df["Language"].unique()),
        default=sorted(email_df["Language"].unique()),
    )
    min_roi = st.slider("Minimum ROI", float(email_df["ROI"].min()), float(email_df["ROI"].max()), 0.0, 0.1)
    profitable_only = st.toggle("Only profitable campaigns", value=False)
    st.caption("Tip: narrow the audience and language filters to present a more focused business story.")


filtered_df = email_df[
    email_df["Target_Audience"].isin(selected_audiences)
    & email_df["Language"].isin(selected_languages)
    & (email_df["ROI"] >= min_roi)
].copy()

if profitable_only:
    filtered_df = filtered_df.loc[filtered_df["Profitable_Flag"] == 1].copy()

if filtered_df.empty:
    st.warning("No campaigns match the current filters. Loosen one of the filters to continue.")
    st.stop()


top_strategy = top_combos.iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Campaigns in View", f"{len(filtered_df):,}")
col2.metric("Average ROI", f"{filtered_df['ROI'].mean():.2f}")
col3.metric("Average Revenue", format_num(filtered_df["Revenue"].mean()))
col4.metric("Profitable Rate", format_pct(filtered_df["Profitable_Flag"].mean()))

chip1, chip2, chip3 = st.columns(3)
chip1.markdown(
    f"""
    <div class="insight-chip">
        <strong>Top strategy</strong><br>
        {top_strategy["Target_Audience"]} + {top_strategy["Language"]}<br><br>
        ROI {top_strategy["Avg_ROI"]:.2f} and revenue {top_strategy["Avg_Revenue"]:,.0f}
    </div>
    """,
    unsafe_allow_html=True,
)
chip2.markdown(
    f"""
    <div class="insight-chip">
        <strong>Model accuracy</strong><br>
        Classification accuracy {metrics["accuracy"]:.1%}<br><br>
        F1 score {metrics["f1"]:.1%}
    </div>
    """,
    unsafe_allow_html=True,
)
chip3.markdown(
    f"""
    <div class="insight-chip">
        <strong>Best launch month</strong><br>
        {monthly_summary.sort_values("Avg_ROI", ascending=False).iloc[0]["Month"]}<br><br>
        Based on highest average ROI
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, insights_tab, strategy_tab, assets_tab = st.tabs(
    ["Overview", "Segment Insights", "Strategy Lab", "Assets & Export"]
)

with overview_tab:
    st.subheader("Performance Overview")
    st.markdown('<p class="section-note">This view summarizes the current filtered slice of email campaigns.</p>', unsafe_allow_html=True)

    filtered_monthly = (
        filtered_df.groupby(["Month_Number", "Month"], as_index=False)
        .agg(Total_Revenue=("Revenue", "sum"), Avg_ROI=("ROI", "mean"))
        .sort_values("Month_Number")
    )
    audience_summary = (
        filtered_df.groupby("Target_Audience", as_index=False)
        .agg(Avg_ROI=("ROI", "mean"))
        .sort_values("Avg_ROI", ascending=False)
    )
    language_summary = (
        filtered_df.groupby("Language", as_index=False)
        .agg(Avg_ROI=("ROI", "mean"))
        .sort_values("Avg_ROI", ascending=False)
    )

    chart_col1, chart_col2 = st.columns([1.4, 1])
    chart_col1.altair_chart(build_line_chart(filtered_monthly), width="stretch")
    chart_col2.altair_chart(
        build_bar_chart(audience_summary, "Target_Audience", "Avg_ROI", "ROI by audience", "#e91e63"),
        width="stretch",
    )
    st.altair_chart(
        build_bar_chart(language_summary, "Language", "Avg_ROI", "ROI by language", "#880e4f"),
        width="stretch",
    )

with insights_tab:
    st.subheader("Segment Insights")
    combo_summary = (
        filtered_df.groupby(["Target_Audience", "Language"], as_index=False)
        .agg(
            Campaigns=("Campaign_ID", "count"),
            Avg_ROI=("ROI", "mean"),
            Avg_Revenue=("Revenue", "mean"),
        )
        .sort_values("Avg_ROI", ascending=False)
    )
    st.altair_chart(build_heatmap(combo_summary), width="stretch")

    left, right = st.columns(2)
    left.dataframe(
        combo_summary.head(10).style.format({"Avg_ROI": "{:.2f}", "Avg_Revenue": "{:,.0f}", "Campaigns": "{:,.0f}"}),
        width="stretch",
    )

    monthly_table = (
        filtered_df.groupby("Month", as_index=False)
        .agg(
            Campaigns=("Campaign_ID", "count"),
            Avg_ROI=("ROI", "mean"),
            Avg_CTR=("CTR", "mean"),
            Profitable_Rate=("Profitable_Flag", "mean"),
        )
        .sort_values("Avg_ROI", ascending=False)
    )
    right.dataframe(
        monthly_table.style.format(
            {"Avg_ROI": "{:.2f}", "Avg_CTR": "{:.2%}", "Profitable_Rate": "{:.1%}", "Campaigns": "{:,.0f}"}
        ),
        width="stretch",
    )

with strategy_tab:
    st.subheader("Strategy Lab")
    st.markdown('<p class="section-note">Choose a campaign setup to estimate how similar historical email campaigns performed.</p>', unsafe_allow_html=True)

    control1, control2, control3, control4 = st.columns(4)
    chosen_audience = control1.selectbox("Audience", sorted(email_df["Target_Audience"].unique()))
    chosen_language = control2.selectbox("Language", sorted(email_df["Language"].unique()))
    chosen_duration = control3.selectbox("Duration bucket", ["Short", "Medium", "Long"])
    chosen_channel_count = control4.selectbox("Channel count", sorted(email_df["Channel_Count"].unique()))

    strategy_df = email_df[
        (email_df["Target_Audience"] == chosen_audience)
        & (email_df["Language"] == chosen_language)
        & (email_df["Duration_Bucket"].astype(str) == chosen_duration)
        & (email_df["Channel_Count"] == chosen_channel_count)
    ].copy()

    if strategy_df.empty:
        st.info("No exact historical campaigns match this setup yet. Try another combination.")
    else:
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Matching Campaigns", f"{len(strategy_df):,}")
        metric2.metric("Expected ROI", f"{strategy_df['ROI'].mean():.2f}")
        metric3.metric("Expected Revenue", format_num(strategy_df["Revenue"].mean()))
        metric4.metric("Profitability", format_pct(strategy_df["Profitable_Flag"].mean()))

        st.dataframe(
            strategy_df[
                [
                    "Campaign_ID",
                    "Date",
                    "Target_Audience",
                    "Language",
                    "Duration_Bucket",
                    "Channel_Count",
                    "ROI",
                    "Revenue",
                    "Engagement_Score",
                ]
            ]
            .sort_values("ROI", ascending=False)
            .head(15)
            .style.format({"ROI": "{:.2f}", "Revenue": "{:,.0f}", "Engagement_Score": "{:.2f}"}),
            width="stretch",
        )

with assets_tab:
    st.subheader("Assets & Export")
    st.markdown("### Recommendation summary")
    st.markdown(recommendation_text)

    st.markdown("### Download project artifacts")
    export_col1, export_col2, export_col3 = st.columns(3)
    export_col1.download_button(
        "Download processed CSV",
        data=PROCESSED_DATA_PATH.read_bytes(),
        file_name="nykaa_email_campaigns_processed.csv",
        mime="text/csv",
    )
    export_col2.download_button(
        "Download recommendations",
        data=RECOMMENDATION_PATH.read_text(encoding="utf-8"),
        file_name="campaign_recommendations.md",
        mime="text/markdown",
    )
    export_col3.download_button(
        "Download top strategies",
        data=TOP_COMBOS_PATH.read_bytes(),
        file_name="top_strategy_combos.csv",
        mime="text/csv",
    )
