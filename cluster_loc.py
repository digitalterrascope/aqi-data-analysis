from datetime import datetime, timedelta
import datetime as dt
from collections import deque, defaultdict, Counter
import matplotlib.pyplot as plt

import aqi_pkg as ap
from aqi_pkg.db import get_session
from aqi_pkg.filters import *
from aqi_pkg.ml.clustering import *

measurements = ["AQI_IN", "AQI_US", "CO_PPB", "NO2_PPB", "O3_PPB", "SO2_PPB", "PM1_UGM3", "PM2_5_UGM3", "PM10_UGM3", "H_PERCENT", "T_C", "TVOC_PPM", "Noise_DB"]

def my_hash_function(string: str) -> int:
    sum = 0
    for char in string:
        sum += ord(char)
    return abs(sum) % 2**32  # np needs 32 bit unsigned integer

def main(session):
    CPCB_NODES_COORDS = ap.tags.CPCB_NODES_COORDS
    COORDS_LOCID = ap.preprocesses.locationId_from_coord(CPCB_NODES_COORDS.values())

    for location_name, coords in CPCB_NODES_COORDS.items():
        locationId = COORDS_LOCID[coords]
        filter = Filter(locationId=locationId, last_updated_range=(datetime(2026, 2, 1), datetime(2026, 2, 28)))
        metrics, metrics_fig = cluster_filter(filter)
        metrics_fig.suptitle(f"{location_name} KMeans Cluster Metrics")
        metrics_fig.savefig(f"{location_name} Cluster Metrics")

        np.random.seed(my_hash_function(location_name))
        k = np.random.randint(3, 7)
        plot_clusters(filter, k=k, location=location_name, display_now=True)


if __name__ == "__main__":
    start_time = datetime.now()
    session = get_session()
    plt.style.use("tableau-colorblind10")
    try:
        main(session)
    finally:
        session.close()
        print("Session closed.")
    end_time = datetime.now()
    print(f"Execution time: {end_time - start_time}")
