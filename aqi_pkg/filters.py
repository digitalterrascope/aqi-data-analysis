from dataclasses import dataclass
from datetime import datetime, time
from typing import List, Tuple
from aqi_pkg.db import ENGINE_URL
from aqi_pkg.tables import Entry
from sqlalchemy import select, and_, or_
from sqlalchemy.dialects import mysql
from sqlalchemy import func
import polars as pl

@dataclass
class Filter:
    scrape_id: List[int] | int = None
    scrape_id_range: List[Tuple[int]] | Tuple[int] = None
    locationId: List[str] | str = None
    lat: List[float] | float = None
    lon: List[float] | float = None
    city: List[str] | str = None
    state: List[str] | str = None
    country: List[str] | str = None
    last_updated_range: List[Tuple[datetime]] | Tuple[datetime] = None
    time_between: List[Tuple[datetime]] | Tuple[datetime] = None
    metrics: List[str] | str = None

    def __str__(self):
        string = ""
        if self.scrape_id is not None:
            string += f"scrape_id={self.scrape_id}, "
        if self.scrape_id_range is not None:
            string += f"scrape_id_range={self.scrape_id_range}, "
        if self.locationId is not None:
            string += f"locationId={self.locationId}, "
        if self.lat is not None:
            string += f"lat={self.lat}, "
        if self.lon is not None:
            string += f"lon={self.lon}, "
        if self.city is not None:
            string += f"city={self.city}, "
        if self.state is not None:
            string += f"state={self.state}, "
        if self.country is not None:
            string += f"country={self.country}, "
        if self.last_updated_range is not None:
            string += f"last_updated_range={self.last_updated_range}, "
        if self.time_between is not None:
            string += f"time_between={self.time_between}, "
        if self.metrics is not None:
            string += f"metrics={self.metrics}, "
        return string[:-2] if string else "unfiltered"


class DataLoader:
    def __init__(self, filters: Filter):
        self.filters = filters
        self.query = None

    def get_query(self):
        f = self.filters
        conditions = []

        # scrape_id
        if f.scrape_id is not None:
            if isinstance(f.scrape_id, list):
                conditions.append(Entry.scrape_id.in_(f.scrape_id))
            else:
                conditions.append(Entry.scrape_id == f.scrape_id)

        # scrape_id range
        if f.scrape_id_range is not None:
            ranges = f.scrape_id_range
            if not isinstance(ranges, list):
                ranges = [ranges]

            range_conditions = [
                Entry.scrape_id.between(start, end)
                for start, end in ranges
            ]
            conditions.append(or_(*range_conditions))

        # string fields helper
        def handle_string_field(field, value):
            if isinstance(value, list):
                return field.in_(value)
            return field == value

        if f.locationId is not None:
            conditions.append(handle_string_field(Entry.locationId, f.locationId))

        

        if f.city is not None:
            conditions.append(handle_string_field(Entry.city, f.city))

        if f.state is not None:
            conditions.append(handle_string_field(Entry.state, f.state))

        if f.country is not None:
            conditions.append(handle_string_field(Entry.country, f.country))

        # lat filter
        if f.lat is not None:
            if isinstance(f.lat, list):
                conditions.append(Entry.lat.in_(f.lat))
            else:
                conditions.append(Entry.lat == f.lat)

        # lon filter
        if f.lon is not None:
            if isinstance(f.lon, list):
                conditions.append(Entry.lon.in_(f.lon))
            else:
                conditions.append(Entry.lon == f.lon)

        # last_updated range
        last_updated_conditions = []
        if f.last_updated_range is not None:
            ranges = f.last_updated_range
            if not isinstance(ranges, list):
                ranges = [ranges]

            for start, end in ranges:
                last_updated_conditions.append(Entry.last_updated.between(start, end))

            conditions.append(or_(*last_updated_conditions))

        # time_between (time of day filter)
        if f.time_between is not None:
            ranges = f.time_between
            if not isinstance(ranges, list):
                ranges = [ranges]

            time_conditions = []

            for start, end in ranges:
                t = func.time(Entry.last_updated)

                if start <= end:
                    # normal case (same day)
                    time_conditions.append(t.between(start, end))
                else:
                    # crosses midnight
                    time_conditions.append(or_(t >= start, t <= end))

            conditions.append(or_(*time_conditions))

        stmt = select(Entry)

        if conditions:
            stmt = stmt.where(and_(*conditions))

        # Compile to MariaDB SQL string with literal values
        compiled = stmt.compile(
            dialect=mysql.dialect(),
            compile_kwargs={"literal_binds": True}
        )

        self.query = str(compiled)
        return self.query
    

    @staticmethod
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

    @staticmethod
    def hourly_aggregates(df: pl.DataFrame) -> pl.DataFrame:
        """
        Groupby locationId and average out coloumns by per hour basis, so there's only one entry per hour
        """

        df = df.sort("last_updated")

        numeric_cols = [
            c for c, dtype in zip(df.columns, df.dtypes)
            if dtype.is_numeric() and c != "scrape_id"
        ]

        agg_exprs = [pl.col(c).mean().alias(c) for c in numeric_cols]

        return (
            df.group_by_dynamic(
                index_column="last_updated",
                every="1h",
                by="locationId",
                closed="left",
            )
            .agg(agg_exprs)
            .sort(["locationId", "last_updated"])
        )

    def get_df(self, cores: int = 9, remove_duplicates: bool = True, hourly_data_only: bool = False):
        if self.query is None:
            self.get_query()

        uri = ENGINE_URL.replace("+pymysql", "")
        uri = uri.replace("localhost", "127.0.0.1")
        df =  pl.read_database_uri(query=self.query, uri=uri, engine="connectorx", partition_on="scrape_id", partition_num=cores)
        if remove_duplicates:
            df = self.remove_duplicates_by_timestamp(df)
        if hourly_data_only:
            df = self.hourly_aggregates(df)
        return df


if __name__ == "__main__":
    from aqi_pkg.db import get_session
    start_time = datetime.now()
    session = get_session()
    try:
        filter = Filter(
            city="Chandigarh",
            last_updated_range=[(datetime(2025, 2, 9), datetime(2026, 2, 15))],
            time_between=(time(22, 0), time(4, 0))
        )
        
        dataLoader = DataLoader(
            filters=filter
        )

        df = dataLoader.get_df()

        print(df.select("last_updated").describe())

    finally:
        session.close()
        print("Session closed.")
    end_time = datetime.now()
    print(f"Execution time: {end_time - start_time}")