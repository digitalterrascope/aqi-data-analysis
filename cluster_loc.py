from datetime import datetime, timedelta
import datetime as dt
from collections import deque, defaultdict, Counter
import matplotlib.pyplot as plt

import aqi_pkg as ap
from aqi_pkg.db import get_session
from aqi_pkg.filters import *
from aqi_pkg.ml.clustering import *

measurements = ["AQI_IN", "AQI_US", "CO_PPB", "NO2_PPB", "O3_PPB", "SO2_PPB", "PM1_UGM3", "PM2_5_UGM3", "PM10_UGM3", "H_PERCENT", "T_C", "TVOC_PPM", "Noise_DB"]

def main(session):
    location = "Sector 22"
    locationId = "13741"

    DIURNAL_TIME_MAP = ap.tags.DIURNAL_TIME_MAP

    for timename, time in DIURNAL_TIME_MAP.items():
        filter = Filter(
            locationId=locationId,
            time_between=time
            )
        metrics, metrics_fig = cluster_filter(filter)
        metrics_fig.savefig(f"{location} {timename} Cluster Metrics")

        plot_clusters(filter, k=7)


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
