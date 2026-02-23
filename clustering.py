from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
import matplotlib.pyplot as plt

from aqi_pkg.db import get_session
from aqi_pkg.filters import *

measurements = ["AQI_IN", "AQI_US", "CO_PPB", "NO2_PPB", "O3_PPB", "SO2_PPB", "PM1_UGM3", "PM2_5_UGM3", "PM10_UGM3", "H_PERCENT", "T_C", "TVOC_PPM", "Noise_DB"]

def main(session):
    filter = Filter()
    
    dataLoader = DataLoader(filters=filter)

    dataLoader.get_query()
    df = dataLoader.get_df()



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
