from datetime import datetime
from aqi_pkg.db import get_session
from aqi_pkg.filters import *
from aqi_pkg.plots import *
from aqi_pkg.data_scripts.collate_aqi import *

measurements = ["AQI_IN", "AQI_US", "CO_PPB", "NO2_PPB", "O3_PPB", "SO2_PPB", "PM1_UGM3", "PM2_5_UGM3", "PM10_UGM3", "H_PERCENT", "T_C", "TVOC_PPM", "Noise_DB"]

def main(session):
    populate_isduplicate(session)


if __name__ == "__main__":
    session = get_session()
    try:
        main(session)
    finally:
        session.close()
        print("Session closed.")