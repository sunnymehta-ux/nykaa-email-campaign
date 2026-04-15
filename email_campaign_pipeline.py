from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nykaa_campaign_data_clean.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nykaa_email_campaigns_processed.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator.divide(denominator).fillna(0.0)


def normalize_channel_combo(channel_value: str) -> str:
    channels = [item.strip() for item in str(channel_value).split(",") if item.strip()]
    return ", ".join(sorted(channels))


def prepare_email_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    email_df = raw_df.loc[raw_df["Campaign_Type"].str.lower() == "email"].copy()
    email_df["Date"] = pd.to_datetime(email_df["Date"])
    email_df["Month"] = email_df["Date"].dt.month_name()
    email_df["Month_Number"] = email_df["Date"].dt.month
    email_df["Quarter"] = "Q" + email_df["Date"].dt.quarter.astype(str)
    email_df["Weekday"] = email_df["Date"].dt.day_name()
    email_df["Channel_Combo"] = email_df["Channel_Used"].map(normalize_channel_combo)
    email_df["Duration_Bucket"] = pd.cut(
        email_df["Duration"],
        bins=[0, 10, 20, 31],
        labels=["Short", "Medium", "Long"],
        include_lowest=True,
    )
    email_df["CTR"] = safe_divide(email_df["Clicks"], email_df["Impressions"])
    email_df["Lead_Rate"] = safe_divide(email_df["Leads"], email_df["Clicks"])
    email_df["Conversion_Rate"] = safe_divide(email_df["Conversions"], email_df["Leads"])
    email_df["Impression_To_Conversion"] = safe_divide(email_df["Conversions"], email_df["Impressions"])
    email_df["Revenue_Per_Click"] = safe_divide(email_df["Revenue"], email_df["Clicks"])
    email_df["Revenue_Per_Conversion"] = safe_divide(email_df["Revenue"], email_df["Conversions"])
    email_df["Cost_Per_Conversion"] = safe_divide(email_df["Acquisition_Cost"], email_df["Conversions"])
    email_df["Profit"] = email_df["Revenue"] - email_df["Acquisition_Cost"]
    email_df["Profitable_Flag"] = (email_df["ROI"] > 0).astype(int)
    return email_df.sort_values("Date").reset_index(drop=True)


def summarize_segments(email_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    agg_map = {
        "Campaign_ID": "count",
        "Revenue": "mean",
        "ROI": "mean",
        "CTR": "mean",
        "Conversion_Rate": "mean",
        "Profit": "mean",
        "Profitable_Flag": "mean",
    }
    rename_map = {
        "Campaign_ID": "Campaigns",
        "Revenue": "Avg_Revenue",
        "ROI": "Avg_ROI",
        "CTR": "Avg_CTR",
        "Conversion_Rate": "Avg_Conversion_Rate",
        "Profit": "Avg_Profit",
        "Profitable_Flag": "Profitable_Rate",
    }

    audience_summary = (
        email_df.groupby("Target_Audience", as_index=False)
        .agg(agg_map)
        .rename(columns=rename_map)
        .sort_values(["Avg_ROI", "Avg_Revenue"], ascending=False)
    )

    language_summary = (
        email_df.groupby("Language", as_index=False)
        .agg(agg_map)
        .rename(columns=rename_map)
        .sort_values(["Avg_ROI", "Avg_Revenue"], ascending=False)
    )

    combo_summary = (
        email_df.groupby(["Target_Audience", "Language"], as_index=False)
        .agg(agg_map)
        .rename(columns=rename_map)
        .sort_values(["Avg_ROI", "Avg_Revenue"], ascending=False)
    )

    month_summary = (
        email_df.groupby(["Month_Number", "Month"], as_index=False)
        .agg(
            Campaigns=("Campaign_ID", "count"),
            Total_Revenue=("Revenue", "sum"),
            Avg_ROI=("ROI", "mean"),
            Avg_CTR=("CTR", "mean"),
            Profitable_Rate=("Profitable_Flag", "mean"),
        )
        .sort_values("Month_Number")
    )

    duration_summary = (
        email_df.groupby("Duration_Bucket", as_index=False)
        .agg(
            Campaigns=("Campaign_ID", "count"),
            Avg_Revenue=("Revenue", "mean"),
            Avg_ROI=("ROI", "mean"),
            Avg_Conversion_Rate=("Conversion_Rate", "mean"),
            Profitable_Rate=("Profitable_Flag", "mean"),
        )
        .sort_values("Avg_ROI", ascending=False)
    )

    channel_count_summary = (
        email_df.groupby("Channel_Count", as_index=False)
        .agg(
            Campaigns=("Campaign_ID", "count"),
            Avg_Revenue=("Revenue", "mean"),
            Avg_ROI=("ROI", "mean"),
            Avg_CTR=("CTR", "mean"),
            Profitable_Rate=("Profitable_Flag", "mean"),
        )
        .sort_values("Avg_ROI", ascending=False)
    )

    return {
        "audience_summary": audience_summary,
        "language_summary": language_summary,
        "audience_language_summary": combo_summary,
        "monthly_summary": month_summary,
        "duration_summary": duration_summary,
        "channel_count_summary": channel_count_summary,
    }


def build_design_matrix(features_df: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.get_dummies(features_df, drop_first=False, dtype=float)
    encoded.insert(0, "Intercept", 1.0)
    return encoded


def fit_ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    identity = np.eye(X.shape[1])
    identity[0, 0] = 0.0
    beta = np.linalg.pinv(X.T @ X + alpha * identity) @ X.T @ y
    return beta


def predict_linear(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return X @ beta


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    residual = actual - predicted
    mae = np.mean(np.abs(residual))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = np.sum((actual - actual.mean()) ** 2)
    r2 = 0.0 if denominator == 0 else 1 - np.sum(residual**2) / denominator
    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
    }


def classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    actual_bool = actual.astype(bool)
    predicted_bool = predicted.astype(bool)
    tp = np.logical_and(actual_bool, predicted_bool).sum()
    tn = np.logical_and(~actual_bool, ~predicted_bool).sum()
    fp = np.logical_and(~actual_bool, predicted_bool).sum()
    fn = np.logical_and(actual_bool, ~predicted_bool).sum()
    accuracy = (tp + tn) / max(len(actual), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def train_profitability_model(email_df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    feature_columns = [
        "Target_Audience",
        "Duration",
        "Language",
        "Customer_Segment",
        "Audience_Segment_Match",
        "Channel_Count",
        "Channel_Combo",
        "Acquisition_Cost",
        "Month",
        "Quarter",
        "Weekday",
    ]
    modeling_df = email_df[feature_columns + ["ROI", "Profitable_Flag"]].copy()
    design_matrix = build_design_matrix(modeling_df[feature_columns])
    split_index = int(len(modeling_df) * 0.8)

    X_train = design_matrix.iloc[:split_index].to_numpy(dtype=float)
    X_test = design_matrix.iloc[split_index:].to_numpy(dtype=float)
    y_train = modeling_df["ROI"].iloc[:split_index].to_numpy(dtype=float)
    y_test = modeling_df["ROI"].iloc[split_index:].to_numpy(dtype=float)
    profitable_test = modeling_df["Profitable_Flag"].iloc[split_index:].to_numpy(dtype=int)

    beta = fit_ridge_regression(X_train, y_train, alpha=3.0)
    predicted_roi = predict_linear(X_test, beta)
    predicted_profitable = (predicted_roi > 0).astype(int)

    metrics = regression_metrics(y_test, predicted_roi)
    metrics.update(classification_metrics(profitable_test, predicted_profitable))
    metrics["train_rows"] = int(split_index)
    metrics["test_rows"] = int(len(modeling_df) - split_index)

    coefficient_table = pd.DataFrame(
        {
            "Feature": design_matrix.columns,
            "Coefficient": beta,
        }
    ).sort_values("Coefficient", ascending=False)

    return metrics, coefficient_table


def add_business_score(summary_df: pd.DataFrame, min_campaigns: int = 200) -> pd.DataFrame:
    scored = summary_df.loc[summary_df["Campaigns"] >= min_campaigns].copy()
    if scored.empty:
        return scored
    score_fields = ["Avg_ROI", "Avg_Revenue", "Avg_Conversion_Rate", "Profitable_Rate"]
    for field in score_fields:
        field_min = scored[field].min()
        field_range = scored[field].max() - field_min
        if field_range == 0:
            scored[f"{field}_Scaled"] = 1.0
        else:
            scored[f"{field}_Scaled"] = (scored[field] - field_min) / field_range
    scored["Business_Score"] = (
        0.35 * scored["Avg_ROI_Scaled"]
        + 0.30 * scored["Avg_Revenue_Scaled"]
        + 0.20 * scored["Avg_Conversion_Rate_Scaled"]
        + 0.15 * scored["Profitable_Rate_Scaled"]
    )
    return scored.sort_values("Business_Score", ascending=False)


def score_summary_frame(summary_df: pd.DataFrame, min_campaigns: int = 200) -> pd.DataFrame:
    scored = summary_df.loc[summary_df["Campaigns"] >= min_campaigns].copy()
    if scored.empty:
        return scored
    score_fields = [field for field in ["Avg_ROI", "Avg_Revenue", "Avg_Conversion_Rate", "Avg_CTR", "Profitable_Rate"] if field in scored.columns]
    weights = {
        "Avg_ROI": 0.35,
        "Avg_Revenue": 0.30,
        "Avg_Conversion_Rate": 0.20,
        "Avg_CTR": 0.10,
        "Profitable_Rate": 0.15,
    }
    total_weight = sum(weights[field] for field in score_fields)
    for field in score_fields:
        field_min = scored[field].min()
        field_range = scored[field].max() - field_min
        if field_range == 0:
            scored[f"{field}_Scaled"] = 1.0
        else:
            scored[f"{field}_Scaled"] = (scored[field] - field_min) / field_range
    scored["Business_Score"] = 0.0
    for field in score_fields:
        scored["Business_Score"] += (weights[field] / total_weight) * scored[f"{field}_Scaled"]
    return scored.sort_values("Business_Score", ascending=False)


def plot_monthly_revenue(monthly_summary: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_summary["Month"], monthly_summary["Total_Revenue"], marker="o", linewidth=2, color="#e91e63")
    plt.title("Nykaa Email Campaign Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Revenue")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_bar(summary_df: pd.DataFrame, label_col: str, value_col: str, title: str, output_path: Path, color: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(summary_df[label_col], summary_df[value_col], color=color)
    plt.title(title)
    plt.xlabel(label_col.replace("_", " "))
    plt.ylabel(value_col.replace("_", " "))
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_heatmap(combo_summary: pd.DataFrame, output_path: Path) -> None:
    pivot = combo_summary.pivot(index="Target_Audience", columns="Language", values="Avg_ROI")
    plt.figure(figsize=(8, 5))
    plt.imshow(pivot.values, cmap="RdPu", aspect="auto")
    plt.colorbar(label="Average ROI")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Average ROI by Audience and Language")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_recommendation_markdown(
    email_df: pd.DataFrame,
    summaries: Dict[str, pd.DataFrame],
    metrics: Dict[str, float],
    coefficient_table: pd.DataFrame,
) -> str:
    combo_scored = add_business_score(summaries["audience_language_summary"])
    duration_scored = score_summary_frame(summaries["duration_summary"])
    channel_scored = score_summary_frame(summaries["channel_count_summary"])
    best_combo = combo_scored.iloc[0]
    best_duration = duration_scored.iloc[0]
    best_channel_count = channel_scored.iloc[0]
    best_month = summaries["monthly_summary"].sort_values("Avg_ROI", ascending=False).iloc[0]

    top_positive_drivers = coefficient_table[
        (~coefficient_table["Feature"].isin(["Intercept"])) & (coefficient_table["Coefficient"] > 0)
    ].head(8)
    top_negative_drivers = coefficient_table[
        (~coefficient_table["Feature"].isin(["Intercept"])) & (coefficient_table["Coefficient"] < 0)
    ].sort_values("Coefficient", ascending=True).head(8)

    report_lines = [
        "# Nykaa Email Marketing Campaign Recommendation",
        "",
        "## Executive Summary",
        f"- Total email campaigns analysed: {len(email_df):,}",
        f"- Average email ROI: {email_df['ROI'].mean():.2f}",
        f"- Average email revenue per campaign: {email_df['Revenue'].mean():,.0f}",
        f"- Share of profitable email campaigns: {email_df['Profitable_Flag'].mean():.1%}",
        "",
        "## Best Performing Campaign Blueprint",
        f"- Primary audience: {best_combo['Target_Audience']}",
        f"- Recommended language: {best_combo['Language']}",
        f"- Expected average ROI for this combo: {best_combo['Avg_ROI']:.2f}",
        f"- Expected average revenue for this combo: {best_combo['Avg_Revenue']:,.0f}",
        f"- Recommended duration bucket: {best_duration['Duration_Bucket']}",
        f"- Recommended channel count: {int(best_channel_count['Channel_Count'])} channels",
        f"- Best launch month based on ROI: {best_month['Month']}",
        "",
        "## Model Snapshot",
        f"- ROI prediction MAE: {metrics['mae']:.2f}",
        f"- ROI prediction RMSE: {metrics['rmse']:.2f}",
        f"- ROI prediction R2: {metrics['r2']:.2f}",
        f"- Profitability classification accuracy: {metrics['accuracy']:.1%}",
        f"- Profitability classification F1 score: {metrics['f1']:.1%}",
        "",
        "## Recommended Actions",
        "- Prioritise premium and working-women audience cohorts for high-value email journeys.",
        f"- Localise the flagship campaign in {best_combo['Language']} for the highest-scoring audience-language fit.",
        f"- Start with a {str(best_duration['Duration_Bucket']).lower()} email window because it offers the strongest blended business score in this dataset.",
        f"- Plan around {int(best_channel_count['Channel_Count'])} supporting channels to balance ROI, revenue, and profitable rate.",
        "- Treat audience-language fit and launch timing as planning levers before scaling acquisition cost.",
        "",
        "## Positive ROI Drivers",
    ]

    for _, row in top_positive_drivers.iterrows():
        report_lines.append(f"- {row['Feature']}: {row['Coefficient']:.3f}")

    report_lines.extend(
        [
            "",
            "## Negative ROI Drivers",
        ]
    )
    for _, row in top_negative_drivers.iterrows():
        report_lines.append(f"- {row['Feature']}: {row['Coefficient']:.3f}")

    return "\n".join(report_lines) + "\n"


def write_outputs(
    email_df: pd.DataFrame,
    summaries: Dict[str, pd.DataFrame],
    metrics: Dict[str, float],
    coefficient_table: pd.DataFrame,
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    email_df.to_csv(PROCESSED_DATA_PATH, index=False)
    for name, summary_df in summaries.items():
        summary_df.to_csv(OUTPUTS_DIR / f"{name}.csv", index=False)

    add_business_score(summaries["audience_language_summary"]).to_csv(
        OUTPUTS_DIR / "top_strategy_combos.csv",
        index=False,
    )
    coefficient_table.to_csv(OUTPUTS_DIR / "model_coefficients.csv", index=False)
    with (OUTPUTS_DIR / "model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    plot_monthly_revenue(summaries["monthly_summary"], CHARTS_DIR / "monthly_revenue_trend.png")
    plot_bar(
        summaries["audience_summary"],
        "Target_Audience",
        "Avg_ROI",
        "Average ROI by Audience",
        CHARTS_DIR / "roi_by_audience.png",
        "#f06292",
    )
    plot_bar(
        summaries["language_summary"],
        "Language",
        "Avg_ROI",
        "Average ROI by Language",
        CHARTS_DIR / "roi_by_language.png",
        "#c2185b",
    )
    plot_heatmap(summaries["audience_language_summary"], CHARTS_DIR / "audience_language_roi_heatmap.png")

    recommendation_markdown = create_recommendation_markdown(email_df, summaries, metrics, coefficient_table)
    (OUTPUTS_DIR / "campaign_recommendations.md").write_text(recommendation_markdown, encoding="utf-8")


def run_pipeline() -> Dict[str, float]:
    raw_df = pd.read_csv(RAW_DATA_PATH)
    email_df = prepare_email_dataset(raw_df)
    summaries = summarize_segments(email_df)
    metrics, coefficient_table = train_profitability_model(email_df)
    write_outputs(email_df, summaries, metrics, coefficient_table)
    return metrics


def main() -> None:
    metrics = run_pipeline()
    print("Nykaa email campaign project build completed.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
