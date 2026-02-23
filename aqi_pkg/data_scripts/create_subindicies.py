from sqlalchemy import text
from sqlalchemy.dialects.mysql import insert as mysql_insert

from tqdm import tqdm
from datetime import datetime, timedelta
from collections import deque, Counter, defaultdict

import polars as pl
pl.Config.set_tbl_rows(100)



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
            pl.col(pollutant).mean().alias("rolling_avg")
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

    rows_iter = (
        df.select(["scrape_id", "rolling_avg"])
          .filter(pl.col("rolling_avg").is_not_null())
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

def export_csv(df, pollutant, hours, path="metric_avg.csv"):
    out = (
        df.select(["scrape_id", "rolling_avg"])
          .filter(pl.col("rolling_avg").is_not_null())
          .with_columns([
              pl.lit(pollutant).alias("metric_name"),
              pl.lit(hours).alias("hours"),
              pl.col("rolling_avg").alias("average_value"),
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