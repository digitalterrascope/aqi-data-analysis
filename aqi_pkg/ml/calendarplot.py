import polars as pl
import plotly.graph_objects as go
import numpy as np
from aqi_pkg.filters import Filter, DataLoader


def generate_calendar_plot(
    location_id: str,
    metric: str,
    agg: str = "mean",
    location: str = "xyz",
    display_now: bool = False,
    colorblind: bool = False,
) -> go.Figure:
    
    filter = Filter(locationId=location_id)
    dataLoader = DataLoader(filter)
    df = dataLoader.get_df()

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {df.columns}")

    filtered = (
        df.filter(pl.col("locationId") == location_id)
        .filter(pl.col("last_updated").is_not_null())
        .filter(pl.col(metric).is_not_null())
        .with_columns([
            pl.col("last_updated").dt.hour().alias("hour"),
            pl.col("last_updated").dt.year().alias("year"),
            pl.col("last_updated").dt.week().alias("week"),
        ])
        .with_columns(
            (pl.col("year").cast(pl.Utf8) + "-W"
             + pl.col("week").cast(pl.Utf8).str.zfill(2)).alias("week_label")
        )
    )

    if filtered.is_empty():
        raise ValueError(f"No data found for locationId='{location_id}'")

    # ── 2. Aggregate ──────────────────────────────────────────────────────────
    agg_expr = {
        "mean":   pl.col(metric).mean(),
        "median": pl.col(metric).median(),
        "max":    pl.col(metric).max(),
        "min":    pl.col(metric).min(),
    }.get(agg)
    if agg_expr is None:
        raise ValueError(f"agg must be one of mean/median/max/min, got '{agg}'")

    pivoted = (
        filtered
        .group_by(["week_label", "year", "week", "hour"])
        .agg(agg_expr.alias("value"))
        .sort(["year", "week", "hour"])
    )

    # ── 3. Build dense week × hour matrix ────────────────────────────────────
    all_weeks = (
        pivoted
        .select(["week_label", "year", "week"])
        .unique()
        .sort(["year", "week"])
        ["week_label"]
        .to_list()
    )
    all_hours = list(range(24))

    # index lookup
    week_idx = {w: i for i, w in enumerate(all_weeks)}

    matrix = np.full((len(all_weeks), 24), np.nan)
    for row in pivoted.iter_rows(named=True):
        r = week_idx[row["week_label"]]
        c = int(row["hour"])
        matrix[r][c] = row["value"]

    # ── 4. Build Plotly figure ────────────────────────────────────────────────
    hour_labels = [f"{h:02d}:00" for h in all_hours]

    # Friendly metric label
    metric_units = {
        "AQI_IN": "AQI (IN)", "AQI_US": "AQI (US)",
        "CO_PPB": "CO (ppb)", "H_PERCENT": "Humidity (%)",
        "NO2_PPB": "NO₂ (ppb)", "O3_PPB": "O₃ (ppb)",
        "PM10_UGM3": "PM10 (µg/m³)", "PM2_5_UGM3": "PM2.5 (µg/m³)",
        "SO2_PPB": "SO₂ (ppb)", "T_C": "Temp (°C)",
        "PM1_UGM3": "PM1 (µg/m³)", "TVOC_PPM": "TVOC (ppm)",
        "Noise_DB": "Noise (dB)",
    }
    metric_label = metric_units.get(metric, metric)

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=hour_labels,
            y=all_weeks,
            colorscale="RdYlGn_r" if not colorblind else "Viridis",
            colorbar=dict(title=metric_label, thickness=14, len=0.8),
            hoverongaps=False,
            hovertemplate=(
                "<b>Week:</b> %{y}<br>"
                "<b>Hour:</b> %{x}<br>"
                f"<b>{metric_label}:</b> %{{z:.2f}}<extra></extra>"
            ),
            xgap=1,
            ygap=1,
        )
    )

    location_name = (
        filtered.select(["city", "state", "country"])
        .unique()
        .head(1)
        .with_columns(
            pl.concat_str(
                [pl.col("city"), pl.col("state"), pl.col("country")],
                separator=", "
            ).alias("label")
        )["label"][0]
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{metric_label}</b> — {location}, location_name<br>"
                f"<sup>Hour of day vs. Week  ·  {agg.capitalize()} per bucket</sup>"
            ),
            x=0.02, xanchor="left",
        ),
        xaxis=dict(
            title="Hour of Day",
            tickmode="array",
            tickvals=hour_labels[::2],
            ticktext=hour_labels[::2],
            side="bottom",
        ),
        yaxis=dict(
            title="Week",
            autorange="reversed",   # most-recent week at top
        ),
        height=max(350, len(all_weeks) * 28 + 150),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#e0e0e0"),
        margin=dict(l=90, r=60, t=90, b=60),
    )

    if display_now:
        fig.show()
        return fig
    else:
        fig_html = fig.to_html(full_html=False, include_plotlyjs="True")
        return fig_html