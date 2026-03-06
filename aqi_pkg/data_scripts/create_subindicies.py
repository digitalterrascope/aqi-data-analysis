from sqlalchemy import text
from sqlalchemy.dialects.mysql import insert as mysql_insert

from tqdm import tqdm
from datetime import datetime, timedelta
from collections import deque, Counter, defaultdict

import polars as pl
pl.Config.set_tbl_rows(100)


def convert_units(df, conversion_map):
    expressions = []

    for old_name, conversions in conversion_map.items():
        if old_name not in df.columns:
            continue

        # Ensure conversions is a list
        if isinstance(conversions, tuple):
            conversions = [conversions]

        for new_name, factor in conversions:
            expressions.append(
                (pl.col(old_name).cast(pl.Float64) * factor).alias(new_name)
            )

    return df.with_columns(expressions)


def compute_rolling(df, pollutant, hours):
    # Skip if pollutant not in df.columns
    if pollutant not in df.columns:
        print(f"Pollutant {pollutant} not found in DataFrame columns. Skipping rolling average computation.")
        return pl.DataFrame()  # Return empty DataFrame

    df = (
        df.with_columns(
            pl.col("last_updated").cast(pl.Datetime)
        )
        .sort(["locationId", "last_updated"])
    )

    result = (
        df.rolling(
            index_column="last_updated",
            period=f"{hours}h",
            by="locationId",
            closed="both"
        )
        .agg(
            pl.col(pollutant).mean().alias(f"{pollutant}_{hours}h_Subindex")
        )
    )

    return result


def attach_scrape_ids(df, rolled):
    return df.join(
        rolled,
        on=["locationId", "last_updated"],
        how="left"
    )


def bulk_insert(session, df, pollutant, hours, MetricAverages, chunk=100_000):

    engine = session.get_bind()

    subindex_col_name = f"{pollutant}_{hours}h_subindex"

    rows_iter = (
        df.select(["scrape_id", subindex_col_name])
          .filter(pl.col(subindex_col_name).is_not_null())
          .iter_rows()
    )

    batch = []

    with engine.begin() as conn:
        for scrape_id, avg in rows_iter:
            batch.append({
                "scrape_id": scrape_id,
                "metric_name": pollutant,
                "hours": hours,
                "average_value": avg,
            })

            if len(batch) >= chunk:
                stmt = mysql_insert(MetricAverages).values(batch)
                stmt = stmt.prefix_with("IGNORE")
                conn.execute(stmt)
                batch.clear()

        if batch:
            stmt = mysql_insert(MetricAverages).values(batch)
            stmt = stmt.prefix_with("IGNORE")
            conn.execute(stmt)


def export_csv_UnitConversions(df, pollutant, path="unit_conversions.csv"):
    out = (
        df.select(["scrape_id", pollutant])
          .with_columns([
              pl.lit(pollutant).alias("metric_name"),
              pl.col(pollutant).round(2).alias("value"),
          ])
          .select([
              "scrape_id",
              "metric_name",
              "value"
          ])
    )

    out.write_csv(path)
    return path

def export_csv_MetricAverages(df, pollutant, hours, path="metric_avg.csv"):
    subindex_col_name = f"{pollutant}_{hours}h_subindex"
    out = (
        df.select(["scrape_id", subindex_col_name])
          .with_columns([
              pl.lit(pollutant).alias("metric_name"),
              pl.lit(hours).alias("hours"),
              pl.col(subindex_col_name).round(2).alias("average_value"),
          ])
          .select([
              "scrape_id",
              "metric_name",
              "hours",
              "average_value"
          ])
    )

    out.write_csv(path)
    return path