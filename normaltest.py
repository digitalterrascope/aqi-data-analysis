from datetime import datetime
from aqi_pkg.db import get_session
from aqi_pkg.filters import *
from aqi_pkg.plots import *
from scipy.stats import normaltest

measurements = ["AQI_IN", "AQI_US", "CO_PPB", "NO2_PPB", "O3_PPB", "SO2_PPB", "PM1_UGM3", "PM2_5_UGM3", "PM10_UGM3", "H_PERCENT", "T_C", "TVOC_PPM", "Noise_DB"]


def main(session):
    for measurement in measurements:
        data = get_measurements(session, measurement, city="Chandigarh", start=datetime(2025, 12, 1), end=datetime(2025, 12, 31))
        data = [x for x in data if x is not None]

        stat, p = normaltest(data)
        print(f"{measurement}: stat={stat:.4f}, p={p:.4e}")


if __name__ == "__main__":
    session = get_session()
    try:
        main(session)
    finally:
        session.close()
        print("Session closed.")