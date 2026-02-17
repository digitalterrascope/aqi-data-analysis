from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
import matplotlib.pyplot as plt

from aqi_pkg.db import get_session
from aqi_pkg.filters import *
from aqi_pkg.plots import *
from aqi_pkg.aqi_standards.in_safar import calculate_pollutant_aqi_safar
from aqi_pkg.data_scripts.create_subindicies import *
from aqi_pkg.models import Entry, IsDuplicate, MetricAverages

measurements = ["AQI_IN", "AQI_US", "CO_PPB", "NO2_PPB", "O3_PPB", "SO2_PPB", "PM1_UGM3", "PM2_5_UGM3", "PM10_UGM3", "H_PERCENT", "T_C", "TVOC_PPM", "Noise_DB"]


def main(session):

    pollutants = {
        "PM2_5_UGM3": 24,
        "PM10_UGM3": 24,
        "NO2_UGM3": 1,
        "OZONE_UGM3": 8,
    }

    t1 = datetime.now()
    print("Loading data...")
    df = load_data(session)
    print(f"Data loaded in {datetime.now() - t1}\n")
    
    t = datetime.now()
    print("Removing duplicates...")
    df = remove_duplicates_by_timestamp(df)
    print(f"Duplicates removed in {datetime.now() - t}\n")

    t = datetime.now()
    print("Calculating subindices...")
    rolled = compute_rolling(df, "PM2_5_UGM3", 24)
    print(f"Subindices calculated in {datetime.now() - t}\n")

    t = datetime.now()
    print("Attaching scrape_ids...")
    final_df = attach_scrape_ids(df, rolled)
    print(f"Scrape_ids attached in {datetime.now() - t}\n")

    t = datetime.now()
    print("Inserting into database...")
    bulk_insert(session, final_df, "PM2_5_UGM3", 24, MetricAverages)
    print(f"Inserted into database in {datetime.now() - t}\n")

    print("Total execution time:", datetime.now() - t1)




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