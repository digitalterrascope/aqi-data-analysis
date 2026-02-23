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
        "NO2_PPB": 1,
        "O3_PPB": 8,
        "SO2_PPB": 1,
        "CO_PPB": 1,
    }

    t1 = datetime.now()
    print("Loading data...")
    df = load_data(session, pollutants)
    # print(f"Data loaded in {datetime.now() - t1}\n")
    
    # print("Removing duplicates...")
    # t = datetime.now()
    # df = remove_duplicates_by_timestamp(df)
    # print(f"Duplicates removed in {datetime.now() - t}\n")

    # for pollutant, hours in pollutants.items():
    #     print(f"\nProcessing {pollutant} ({hours}h window)")

    #     t = datetime.now()
    #     rolled = compute_rolling(df, pollutant, hours)
    #     print(f"Rolling average computed in {datetime.now() - t}")

    #     t = datetime.now()
    #     final_df = attach_scrape_ids(df, rolled)
    #     print(f"Scrape IDs attached in {datetime.now() - t}")

    #     t = datetime.now()
    #     bulk_insert(session, final_df, pollutant, hours, MetricAverages)
    #     print(f"Inserted into DB in {datetime.now() - t}")

    # print("Total execution time:", datetime.now() - t1)




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