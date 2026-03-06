"""
Utility functions from chandigarh-rho-aqi.ipynb copied for use in correlation-at-locations.ipynb
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("colorblind")
plt.style.use("tableau-colorblind10")

from aqi_pkg.data_scripts.create_subindicies import *
import aqi_pkg as ap

CPCB_COLS = ['AQI_CO_MGM3', 'AQI_NO2_UGM3', 'AQI_O3_UGM3', 'AQI_PM10_UGM3', 'AQI_PM2_5_UGM3', 'AQI_SO2_UGM3']
SAFAR_COLS = ['AQI_CO_PPM', 'AQI_NO2_PPB', 'AQI_O3_PPB', 'AQI_PM10_UGM3', 'AQI_PM2_5_UGM3']
AQI_COLS = ['AQI_CO_MGM3', 'AQI_CO_PPM', 'AQI_NO2_PPB', 'AQI_NO2_UGM3', 'AQI_O3_PPB', 'AQI_O3_UGM3', 'AQI_PM10_UGM3', 'AQI_PM2_5_UGM3', 'AQI_SO2_PPB', 'AQI_SO2_UGM3', 'AQI_IN', 'AQI_US', 'AQI_CPCB', 'AQI_SAFAR', 'AQI_rho']

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
                'PM10_UGM3_24h_subindex'
                ]

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
    print(aqi_cols)

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

def filter_by_location(df: pl.DataFrame, location_name:str) -> pl.DataFrame:
    interest_coords = (
        df
        .filter(pl.col("location").str.to_lowercase() == location_name.lower())
        .select(["lat", "lon"])
        .unique()
        .rows()[0]
    )
    
    interest_locationid = df.filter(
        (pl.col("lat") == interest_coords[0]) &
        (pl.col("lon") == interest_coords[1])
    ).select("locationId").unique().item()
    
    return df.filter(pl.col("locationId") == interest_locationid)

def plot_cols(df: pl.DataFrame, cols: list[str]):
    pdf = df.to_pandas()

    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2*n), sharex=True)

    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.plot(pdf["last_updated"], pdf[col])
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid("on")

    axes[-1].set_xlabel("last_updated")

    fig.autofmt_xdate()
    fig.tight_layout()

    return fig

def plot_aqi_correlation_heatmap(df: pl.DataFrame, interest_location: str, rho: float = 2.2):
    aqi_metrics = ["AQI_IN", "AQI_US", "AQI_SAFAR", "AQI_CPCB", "AQI_rho"]

    results = []
    for metric in aqi_metrics:
        pollutants = ['CO_PPB', 'CO_MGM3', 'CO_PPM',
                    'NO2_UGM3', 'NO2_PPB', 
                    'O3_UGM3', 'O3_PPB', 
                    'SO2_UGM3', 'SO2_PPB', 
                    'PM2_5_UGM3', 'PM10_UGM3']
        pollutants.extend(list(set(CPCB_COLS + SAFAR_COLS))) # for AQI

        for p in pollutants:
            corr = df.select(pl.corr(metric, p)).item()

            results.append({
                "AQI_metric": metric,
                "pollutant": p,
                "correlation": corr
            })

    corr_df = pl.DataFrame(results)

    # Pivot to matrix form 
    corr_matrix = corr_df.pivot(
        values="correlation",
        index="pollutant",
        on="AQI_metric"
    )
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        corr_matrix.to_pandas().set_index("pollutant").sort_index(),
        annot=True,
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title(f"Correlation of Pollutants and AQIs (rho = {rho}) at {interest_location}")

    fig.tight_layout()

    return fig, corr_matrix

def add_georef_location(df: pl.DataFrame, CPCB_NODES_COORDS: dict, georefs_filepath: str = "tricity_georefs.xlsx") -> pl.DataFrame:
    # CPCB Locations
    cpcb_df = pl.DataFrame([
        {"lat": lat, "lon": lon, "location_cpcb": name}
        for name, (lat, lon) in CPCB_NODES_COORDS.items()
    ])

    df = df.join(
        cpcb_df,
        on=["lat", "lon"],
        how="left"
    )

    georef_df = pl.read_excel(georefs_filepath)
    georef_df = georef_df.with_columns(
        pl.when(pl.col("LocationNote").is_not_null())
        .then(pl.col("LocationNote"))
        .when(pl.col("suburb").is_not_null())
        .then(pl.concat_str([pl.col("suburb"), pl.col("city")], separator=" "))
        .otherwise(
            pl.concat_str(
                [pl.col("name"), pl.col("street"), pl.col("city")],
                separator=", "
            )
        )
        .alias("location_georef")
    ).select([
        pl.col("original_lat").alias("lat"),
        pl.col("original_lon").alias("lon"),
        "location_georef"
    ])

    df = df.join(
        georef_df,
        on=["lat", "lon"],
        how="left"
    )

    # Final location priority
    df = df.with_columns(
        pl.coalesce(
            ["location_cpcb", "location_georef"]
        ).alias("location")
    ).drop(["location_cpcb", "location_georef"])

    return df
