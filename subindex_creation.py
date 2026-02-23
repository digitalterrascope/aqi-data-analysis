from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
import matplotlib.pyplot as plt

from aqi_pkg.db import get_session
from aqi_pkg.tables import Entry, MetricAverages, IsDuplicate
from aqi_pkg.filters import *
from aqi_pkg.data_scripts.create_subindicies import *

import threading

def insert_csv_to_db(csv_path):
    from sqlalchemy.exc import SQLAlchemyError
    session = get_session() 

    try:
        df = pl.read_csv(csv_path)
        # Adjust columns to match MetricAverages table
        records = df.to_dicts()
        session.bulk_insert_mappings(MetricAverages, records)
        session.commit()
        print(f"Inserted {len(records)} records from {csv_path} into database.")
    except SQLAlchemyError as e:
        session.rollback()
        print(f"SQLAlchemy error inserting {csv_path}: {e}")
    except Exception as e:
        print(f"Error inserting {csv_path}: {e}")
    finally:
        session.close()

measurements = ["AQI_IN", "AQI_US", "CO_PPB", "NO2_PPB", "O3_PPB", "SO2_PPB", "PM1_UGM3", "PM2_5_UGM3", "PM10_UGM3", "H_PERCENT", "T_C", "TVOC_PPM", "Noise_DB"]


def main(session):
    filter = Filter()
    
    
    dataLoader = DataLoader(
        filters=filter    
    )

    time = datetime.now()
    print("Loading data...")
    df = dataLoader.get_df()
    print(f"Data loaded in {datetime.now() - time}")


    subindexToCalculate = {
        "NO2_PPB": 24,
        "O3_PPB": 8,
        "SO2_PPB": 24,
    }


    threads = []
    for pollutant, hours in subindexToCalculate.items():
        time = datetime.now()
        print(f"Calculating {pollutant} subindex for {hours} hours...")
        rolled = compute_rolling(df, pollutant, hours)
        df_with_ids = attach_scrape_ids(df, rolled)
        print(df_with_ids.head())
        print(f"{pollutant} subindex calculated in {datetime.now() - time}")
        time = datetime.now()
        csv_filename = f'{pollutant}-{hours}h-SubIndex.csv'
        export_csv(df_with_ids, pollutant, hours, csv_filename)
        print(f"{pollutant} subindex inserted in database in {datetime.now() - time}")

        # Start thread and collect
        thread = threading.Thread(target=insert_csv_to_db, args=(csv_filename,))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish before closing session
    for thread in threads:
        thread.join()



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
