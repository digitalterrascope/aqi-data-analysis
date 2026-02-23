from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple
from aqi_pkg.db import ENGINE_URL
from aqi_pkg.tables import Entry
from sqlalchemy import select, and_, or_
from sqlalchemy.dialects import mysql
import polars as pl

@dataclass
class Filter:
    scrape_id: List[int] | int = None
    scrape_id_range: List[Tuple[int]] | Tuple[int] = None
    locationId: List[str] | str = None
    city: List[str] | str = None
    state: List[str] | str = None
    country: List[str] | str = None
    last_updated_range: List[Tuple[datetime]] | Tuple[datetime] = None
    metrics: List[str] | str = None


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

        # last_updated range
        last_updated_conditions = []
        if f.last_updated_range is not None:
            ranges = f.last_updated_range
            if not isinstance(ranges, list):
                ranges = [ranges]

            for start, end in ranges:
                last_updated_conditions.append(Entry.last_updated.between(start, end))

            conditions.append(or_(*last_updated_conditions))

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
    

    def get_df(self, cores: int = 9):
        if self.query is None:
            raise ValueError("Query not generated. Call get_query() first.")

        uri = ENGINE_URL.replace("+pymysql", "")
        uri = uri.replace("localhost", "127.0.0.1")
        return pl.read_database_uri(query=self.query, uri=uri, engine="connectorx", partition_on="scrape_id", partition_num=cores)


if __name__ == "__main__":
    from aqi_pkg.db import get_session
    start_time = datetime.now()
    session = get_session()
    try:
        filter = Filter(
            city="Chandigarh",
            last_updated_range=[(datetime(2025, 2, 9), datetime(2026, 2, 15))],
        )
        
        dataLoader = DataLoader(
            filter=filter    
        )
    finally:
        session.close()
        print("Session closed.")
    end_time = datetime.now()
    print(f"Execution time: {end_time - start_time}")