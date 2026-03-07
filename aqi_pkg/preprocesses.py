import polars as pl
import aqi_pkg as ap
from typing import Tuple, List


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


def locationId_from_coord(coords: List[Tuple[float, float]] | Tuple[float, float]) \
    -> dict[Tuple[float, float], str]:

    if isinstance(coords, tuple):
        coords = [coords]

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    filter = ap.filters.Filter(lat=lats, lon=lons)
    dataLoader = ap.filters.DataLoader(filter)
    df = dataLoader.get_df()

    df = df.select(["locationId", "lat", "lon"]).unique()

    return {
        (row["lat"], row["lon"]): row["locationId"]
        for row in df.to_dicts()
    }


def all_georef_location_to_csv(
    output_path: str = "georef_locations.csv",
    georefs_filepath: str = "tricity_georefs.xlsx"
) -> None:
    filter = ap.filters.Filter(city=ap.tags.TRICITY_CITIES_LIST)
    dataLoader = ap.filters.DataLoader(filter)
    df = dataLoader.get_df()

    df = add_georef_location(df, georefs_filepath)
    (
        df
        .select(["locationId", "location"])
        .unique(subset=["locationId"], keep="last")
        .write_csv(output_path)
    )


def get_location_from_locationId(locationId: str) -> str:
    df = pl.read_csv("/home/studentiotlab/aqi-data-analysis/georef_locations.csv")
    location = df.filter(pl.col("locationId") == locationId).select("location").item()
    return location