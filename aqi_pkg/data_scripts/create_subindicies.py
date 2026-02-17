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



def check_duplicates(data):
    timestamp_counts = Counter(d.last_updated for d in data)
    duplicates = {ts: c for ts, c in timestamp_counts.items() if c > 1}
    print(f"Total duplicate timestamps: {len(duplicates)}")


def calculate_subindices(session, data, pollutant: str, window_hours: int, MetricAverages):
    """
    calculate_subindices
    
    :param data: SQLAlchemy query result of Entry objects for each locationId without duplicates
    :param pollutant: string, name of the pollutant column to calculate subindex for (e.g., "PM2_5_UGM3")
    :param window_hours: int, number of hours for rolling average window

    Creates entry in MetricAverages table for each scrape_id with the calculated subindex for the specified pollutant and rolling average window.
    """

    # Create polars df
    df = pl.DataFrame([{
        "scrape_id": d.scrape_id,
        "last_updated": d.last_updated,
        pollutant: getattr(d, pollutant)
    } for d in data])

    
    # Ensure datetime column is in correct type
    df = df.with_columns(
        pl.col("last_updated").cast(pl.Datetime)
    )

    # Sort by last_updated
    df = df.sort("last_updated")
    
    # Calculate rolling average
    rolled = (
        df.rolling(
            index_column="last_updated",
            period=f"{window_hours}h",
            closed="both"
        )
        .agg(
            pl.col(pollutant).mean().alias(f"{pollutant}_rolling_avg")
        )
    )

    # join back to original rows
    df = df.join(rolled, on="last_updated", how="left")

    print(df.head())

    # Insert into MetricAverages table
    for row in df.iter_rows():
        session.add(MetricAverages(
            scrape_id=row[0],
            metric_name=pollutant,
            hours=window_hours,
            average_value=row[3]  # Assuming the rolling average is the 4th column
        ))

    session.commit()


def load_data(session, pollutant="PM2_5_UGM3"):
    query = f"""
        SELECT scrape_id, locationId, last_updated, {pollutant}
        FROM AqiInScrape
        WHERE city = 'Chandigarh' AND last_updated BETWEEN '2025-11-01' AND '2026-02-15'
        """

    return pl.read_database(query, session.bind)


def compute_rolling(df, pollutant="PM2_5_UGM3", hours=24):
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


def bulk_insert(session, df, pollutant, hours, MetricAverages):
    rows = [
        {
            "scrape_id": r[0],
            "metric_name": pollutant,
            "hours": hours,
            "average_value": r[1],
        }
        for r in df.select(["scrape_id", "rolling_avg"]).iter_rows()
    ]

    # Ignore when duplicate primary key entry is being inserted
    stmt = mysql_insert(MetricAverages).values(rows)
    stmt = stmt.prefix_with("IGNORE") 

    session.execute(stmt)
    session.commit()
