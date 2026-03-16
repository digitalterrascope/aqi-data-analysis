import numpy as np
import polars as pl

RHO = 2.2

def compute_rho_AQI(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute RHO-AQI for a given dataset
    Operates on CPCB cols
    """

    aqi_cols = [c for c in df.columns if "AQI_" in c]

    cpcb_cols = [
        c for c in aqi_cols
        if any(k in c for k in ["NO2_UGM3", "O3_UGM3", "SO2_UGM3", "CO_MGM3", "PM2_5", "PM10"])
    ]


    return df.with_columns(
        (
            sum(pl.col(col).pow(RHO) for col in cpcb_cols)
            .pow(1 / RHO)
        ).alias("AQI_rho")
    )