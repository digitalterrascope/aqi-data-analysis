from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

from sqlalchemy.dialects.mysql import insert

from aqi_pkg.db import get_session
from aqi_pkg.tables import Entry, MetricAverages, IsDuplicate, UnitConversions
from aqi_pkg.filters import *
from aqi_pkg.data_scripts.create_subindicies import *

import threading

BATCH_SIZE = 10_000

def bulk_insert_ignore(session, model, records):
    if not records:
        return

    stmt = insert(model)

    update_dict = {
        col.name: stmt.inserted[col.name]
        for col in model.__table__.columns
        if not col.primary_key  # don't update the Primary Key itself
    }

    upsert_stmt = stmt.on_duplicate_key_update(update_dict)

    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(records), BATCH_SIZE),
                  total=total_batches,
                  desc="Inserting",
                  unit="batch"):
        batch = records[i:i+BATCH_SIZE]
        session.execute(upsert_stmt, batch)
        session.commit()


def insert_csv_to_UnitConversion(csv_path):
    from sqlalchemy.exc import SQLAlchemyError
    session = get_session() 
    try:
        df = pl.read_csv(csv_path)
        # Adjust columns to match UnitConversions table
        records = df.to_dicts()
        bulk_insert_ignore(session, UnitConversions, records)
        print(f"Inserted {len(records)} records from {csv_path} into database.")
    except SQLAlchemyError as e:
        session.rollback()
        print(f"SQLAlchemy error inserting {csv_path}: {e}")
    except Exception as e:
        print(f"Error inserting {csv_path}: {e}")
    finally:        
        session.close()


def insert_csv_to_MetricAverages(csv_path):
    from sqlalchemy.exc import SQLAlchemyError
    session = get_session() 

    try:
        df = pl.read_csv(csv_path)
        # Adjust columns to match MetricAverages table
        records = df.to_dicts()
        bulk_insert_ignore(session, MetricAverages, records)
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
    print(df.shape)

    subindexToCalculate = {
        # "NO2_PPB": 24,
        "O3_PPB": 8,
        "SO2_PPB": 24,
        "CO_PPB": 8,
    }

    units_conversion_map = {
        # "NO2_PPB": ("NO2_UGM3", 1.88),
        "O3_PPB": ("O3_UGM3", 1.96),
        "SO2_PPB": ("SO2_UGM3", 2.62),
        "CO_PPB": ("CO_UGM3", 1.96),
    }


    threads = []
    for pollutant, hours in subindexToCalculate.items():
        time = datetime.now()
        print(f"Calculating {pollutant} subindex for {hours} hours...")
        
        df = convert_units(df, units_conversion_map)

        pollutant = pollutant.replace("_PPB", "_UGM3")

        rolled = compute_rolling(df, pollutant, hours)
        df_with_ids = attach_scrape_ids(df, rolled)
        
        print(df_with_ids.head())
        print(f"{pollutant} subindex calculated in {datetime.now() - time}")

        time = datetime.now()
        csv_filename_MetricAverages = f'csv/{pollutant}-{hours}h-SubIndex.csv'
        csv_filename_UnitConversion = f'csv/{pollutant}_conversion.csv'

        export_csv_UnitConversions(df_with_ids, pollutant, csv_filename_UnitConversion)
        export_csv_MetricAverages(df_with_ids, pollutant, hours, csv_filename_MetricAverages)


        # Start thread and collect
        thread_subindex = threading.Thread(target=insert_csv_to_MetricAverages, args=(csv_filename_MetricAverages,))
        thread_subindex.start()
        threads.append(thread_subindex)

        # Start thread for unit conversions
        thread_conversion = threading.Thread(target=insert_csv_to_UnitConversion, args=(csv_filename_UnitConversion,))
        thread_conversion.start()
        threads.append(thread_conversion)
        
        print(f"{pollutant} insert thread strated in {datetime.now() - time}")

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
