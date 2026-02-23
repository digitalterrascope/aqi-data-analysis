from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

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