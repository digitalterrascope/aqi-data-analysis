from sqlalchemy import text
from sqlalchemy.dialects.mysql import insert as mysql_insert

from tqdm import tqdm
from datetime import datetime, timedelta
from collections import deque, Counter, defaultdict

import polars as pl
pl.Config.set_tbl_rows(100)



def remove_duplicates_by_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove duplicates, keep the entry with latest scrape_id
    
    :param df: Polars DF
    :type df: pl.DataFrame
    :return: DF with duplicaes removed
    :rtype: DataFrame
    """
    return (
        df.sort("scrape_id")
          .unique(subset=["locationId", "last_updated"], keep="last")
    )



def load_data(session, pollutants, start='2025-02-09', end='2026-02-15'):
    if isinstance(pollutants, dict):
        pollutants = list(pollutants.keys())
    elif isinstance(pollutants, str):
        pollutants = [pollutants]

    cols = ", ".join(pollutants)

    query = f"""
        SELECT scrape_id, city, locationId, last_updated, {cols}
        FROM AqiInScrape
        WHERE last_updated BETWEEN '{start}' AND '{end}'
    """

    return pl.read_database(query=query, connection=session.bind.execution_options(stream_results=True), 
                            iter_batches=True,
                            batch_size=100_000
                            )




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


def bulk_insert(session, df, pollutant, hours, MetricAverages, chunk=50000):

    data = df.select(["scrape_id", "rolling_avg"]) \
             .filter(pl.col("rolling_avg").is_not_null()) \
             .to_dicts()

    for i in range(0, len(data), chunk):
        rows = [
            {
                "scrape_id": r["scrape_id"],
                "metric_name": pollutant,
                "hours": hours,
                "average_value": r["rolling_avg"],
            }
            for r in data[i:i+chunk]
        ]

        stmt = mysql_insert(MetricAverages).values(rows)
        stmt = stmt.prefix_with("IGNORE")

        session.execute(stmt)

    session.commit()
