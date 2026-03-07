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
            pl.col(pollutant).mean().alias(f"{pollutant}_{hours}h_subindex")
        )
    )

    return result


def attach_scrape_ids(df, rolled):
    return df.join(
        rolled,
        on=["locationId", "last_updated"],
        how="left"
    )


"""
TO CALC AQIs RUN:
 1. convert_units_and_calculate_subindicies
 2. calculate_aqi_metrics
"""

def convert_units_and_calculate_subindicies(df: pl.DataFrame) -> pl.DataFrame:
    subindexToCalculate = {
        "NO2": 24,
        "O3": 8,
        "SO2": 24,
        "CO": 8,
        "PM2_5": 24,
        "PM10": 24,
    }

    from aqi_pkg.tags import UNIT_CONVERSION_MAP as units_conversion_map
    df = convert_units(df, units_conversion_map)

    units = ["PPB", "UGM3"]
    for pollutant, hours in subindexToCalculate.items():
            for unit in units:
                pollutant_name = f"{pollutant}_{unit}"
                if pollutant == "CO": # Handle CO Units
                    pollutant_name = pollutant_name.replace("_PPB", "_PPM").replace("_UGM3", "_MGM3")
                if "PM" in pollutant and "PP" in unit:
                    continue # skips the rest of THIS iteration

                rolled = compute_rolling(df, pollutant_name, hours)
                df = attach_scrape_ids(df, rolled)


    return df


def calculate_aqi_metrics(df: pl.DataFrame, rho: float = 2.2) -> pl.DataFrame:
    from aqi_pkg.aqi_standards.in_cbcp import calculate_pollutant_aqi_cpcb
    from aqi_pkg.aqi_standards.in_safar import calculate_pollutant_aqi_safar

    def calculate_aqi(pollutant, value):
        # Helper function to have try catch here
        try:
            return calculate_pollutant_aqi_cpcb(pollutant, value)
        except Exception as e:
            print(f"CPCB WOMP: {e}")
            try:
                return calculate_pollutant_aqi_safar(pollutant, value)
            except Exception as e:
                print(f"SAFAR WOMP: {e}")
                return None

    col_names = ['NO2_PPB_24h_subindex', 
                'NO2_UGM3_24h_subindex', 
                'O3_PPB_8h_subindex', 
                'O3_UGM3_8h_subindex', 
                'SO2_PPB_24h_subindex', 
                'SO2_UGM3_24h_subindex', 
                'CO_PPM_8h_subindex', 
                'CO_MGM3_8h_subindex', 
                'PM2_5_UGM3_24h_subindex', 
                'PM10_UGM3_24h_subindex']

    exprs = []

    for col in col_names:
        pollutant = "_".join(col.split("_")[:-2])  # removes '24h_subindex' or '8h_subindex'
        new_col_name = f"AQI_{pollutant}"
        expr = (
            pl.col(col)
            .map_elements(
                lambda x: calculate_aqi(pollutant, x)
                if x is not None else None,
                return_dtype=pl.Int64
            )
            .alias(new_col_name)
        )
        
        exprs.append(expr)

    df = df.with_columns(exprs)

    # Aggregate AQI_CPCB, AQI_SAFAR
    aqi_cols = [c for c in df.columns if "AQI_" in c and "IN" not in c and "US" not in c]
    aqi_cols.sort()
 
    cpcb_cols = [
        c for c in aqi_cols
        if any(k in c for k in ["NO2_UGM3", "O3_UGM3", "SO2_UGM3", "CO_MGM3", "PM2_5", "PM10"])
    ]

    safar_cols = [
        c for c in aqi_cols
        if any(k in c for k in ["CO_PPM", "NO2_PPB", "O3_PPB", "PM2_5", "PM10"])
    ]

    df = df.with_columns([
        pl.max_horizontal([pl.col(c) for c in cpcb_cols]).alias("AQI_CPCB"),
        pl.max_horizontal([pl.col(c) for c in safar_cols]).alias("AQI_SAFAR"),
    ])

    df = df.with_columns(
        (
            sum(pl.col(col).pow(rho) for col in cpcb_cols)
            .pow(1 / rho)
        ).alias("AQI_rho")
    )

    return df

