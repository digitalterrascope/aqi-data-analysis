from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
import matplotlib.pyplot as plt

from aqi_pkg.db import get_session
from aqi_pkg.filters import *
from aqi_pkg.ml.ensemble import run_ensemble_experiment

from sklearn.linear_model import LinearRegression


measurements = ["AQI_IN", "AQI_US", "CO_PPB", "NO2_PPB", "O3_PPB", "SO2_PPB", "PM1_UGM3", "PM2_5_UGM3", "PM10_UGM3", "H_PERCENT", "T_C", "TVOC_PPM", "Noise_DB"]


def main(session):
    filter = Filter()

    dataLoader = DataLoader(
        filters=filter
    )

    time = datetime.now()
    print("Loading data...")

    df = dataLoader.get_df(
        remove_duplicates=True,
        hourly_data_only=True
    )

    print(f"Data loaded in {datetime.now() - time}")

    for n in range(1, 10):
        run_ensemble_experiment(
            df=df,
            target_col="PM2_5_UGM3",
            feature_cols=["PM2_5_UGM3"],
            num_hours_lookback=n,
            model=LinearRegression(),
            min_location_size=2500,
            test_size=0.2,
            shuffle=True,
            random_state=9320,
            output_dir="figs/ensemble"
        )





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
