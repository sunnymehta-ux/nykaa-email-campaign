# Nykaa Email Marketing Campaign Project

This project turns the provided Nykaa campaign dataset into an end-to-end email marketing case study. It filters the data to email campaigns, engineers performance KPIs, builds a lightweight profitability model, and exports charts plus business recommendations.

## What this project does

- Loads the raw campaign CSV from `data/raw`
- Filters only `Email` campaigns
- Creates marketing KPIs such as CTR, lead rate, conversion rate, cost per conversion, and profit
- Ranks the best-performing audience, language, duration, and timing combinations
- Trains a lightweight ridge-style ROI prediction model using only `pandas` and `numpy`
- Generates output CSV files, charts, model metrics, and a final recommendation report
- Includes an interactive Streamlit dashboard for exploration and presentation
- Includes a Power BI-ready package with fact tables, dimensions, theme, and DAX measures

## Project structure

```text
nykaa-email-campaign/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── charts/
│   ├── audience_summary.csv
│   ├── audience_language_summary.csv
│   ├── campaign_recommendations.md
│   ├── channel_count_summary.csv
│   ├── duration_summary.csv
│   ├── language_summary.csv
│   ├── model_coefficients.csv
│   ├── model_metrics.json
│   ├── monthly_summary.csv
│   └── top_strategy_combos.csv
├── .streamlit/
│   └── config.toml
├── src/
│   └── email_campaign_pipeline.py
├── power_bi/
│   ├── dax_measures.md
│   ├── README.md
│   └── theme.json
├── build_power_bi_assets.py
├── dashboard.py
├── PUBLISHING_GUIDE.md
├── requirements.txt
└── run_pipeline.py
```

## How to run

From the project folder:

```powershell
python run_pipeline.py
```

To launch the dashboard:

```powershell
streamlit run dashboard.py
```

To rebuild the Power BI package:

```powershell
python build_power_bi_assets.py
```

To generate the Desktop-openable Power BI Project:

```powershell
python build_power_bi_project.py
```

## Main outputs

- `data/processed/nykaa_email_campaigns_processed.csv`: Email-only prepared dataset
- `outputs/campaign_recommendations.md`: Presentation-ready recommendation summary
- `outputs/model_metrics.json`: ROI and profitability model metrics
- `outputs/charts/*.png`: Visuals for the final presentation
- `dashboard.py`: Streamlit app for interactive storytelling and demo
- `data/power_bi/*.csv`: Power BI-ready fact and dimension tables
- `power_bi/`: Power BI setup guide, theme, and DAX measures
- `power_bi_project/`: generated PBIP project files for Power BI Desktop
- `PUBLISHING_GUIDE.md`: Steps to publish on GitHub and showcase on LinkedIn

## Suggested presentation storyline

1. Email campaign performance overview
2. Best audience and language segments for Nykaa
3. Seasonality and launch timing insights
4. Model-backed profitability drivers
5. Final recommended campaign blueprint

## Notes

- The model intentionally avoids `scikit-learn` so the project remains runnable with the base Python tools available in this environment.
- Streamlit is listed in `requirements.txt` so the app can be installed and shared easily.
