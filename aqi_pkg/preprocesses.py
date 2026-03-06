import polars as pl
import aqi_pkg as ap

def add_georef_location(df: pl.DataFrame, georefs_filepath: str = "tricity_georefs.xlsx") -> pl.DataFrame:
    # CPCB Locations
    CPCB_NODES_COORDS = ap.tags.CPCB_NODES_COORDS
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
