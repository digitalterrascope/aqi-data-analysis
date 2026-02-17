from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
import matplotlib.pyplot as plt

from aqi_pkg.db import get_session
from aqi_pkg.filters import *
from aqi_pkg.plots import *
from aqi_pkg.aqi_standards.in_safar import calculate_pollutant_aqi_safar

measurements = ["AQI_IN", "AQI_US", "CO_PPB", "NO2_PPB", "O3_PPB", "SO2_PPB", "PM1_UGM3", "PM2_5_UGM3", "PM10_UGM3", "H_PERCENT", "T_C", "TVOC_PPM", "Noise_DB"]


def deduplicate_by_timestamp(data):
    grouped = defaultdict(list)

    # group rows by timestamp
    for d in data:
        grouped[d.last_updated].append(d)

    deduped = []

    for ts in sorted(grouped.keys()):
        rows = grouped[ts]

        base = rows[0]
        deduped.append(base)

    return deduped

def check_duplicates(data):
    timestamp_counts = Counter(d.last_updated for d in data)
    duplicates = {ts: c for ts, c in timestamp_counts.items() if c > 1}
    print(f"Total duplicate timestamps: {len(duplicates)}")

def main(session):
    # city = 'Chandiagrh', locationId = '12428' 
    filters = {
        "city": "Chandigarh",
        "locationId": "12428",
        "start": datetime(2025, 11, 1),
        "end": datetime(2026, 2, 14)
    }

    data = session.query(Entry).filter(Entry.city == filters["city"], Entry.locationId == filters["locationId"], Entry.last_updated >= filters["start"], Entry.last_updated <= filters["end"]).all()
    data = deduplicate_by_timestamp(data)
    check_duplicates(data)


    # Calculate rolling average for PM2.5
    AVG_WINDOW = 24 # Hour(s)
    COLS= ["PM2_5_UGM3"]

    # Datapoints are recorded in a non fixed window, so we will calculate the rolling average based on the timestamp of the entries
    data.sort(key=lambda x: (x.last_updated, x.scrape_id))  # Ensure data is sorted by timestamp
    subindicies = {}


    t1 = datetime.now()
    for COL in COLS:
        rolling_averages = []
        for i in range(len(data)):
            window_start = data[i].last_updated - timedelta(hours=AVG_WINDOW)
            window_data = [getattr(d, COL) for d in data if window_start <= d.last_updated <= data[i].last_updated and getattr(d, COL) is not None]
            if window_data:
                rolling_averages.append((data[i].last_updated, sum(window_data) / len(window_data)))
            else:
                rolling_averages.append((data[i].last_updated, None))
        subindicies[COL] = rolling_averages
    
    t2 = datetime.now()
    print(f"Rolling average calculated in {t2 - t1}")

    for COL in COLS:
        rolling_averages = []

        window = deque()   # holds (timestamp, value)
        window_sum = 0.0
        window_count = 0

        start_idx = 0

        for point in data:
            current_time = point.last_updated
            window_start_time = current_time - timedelta(hours=AVG_WINDOW)

            value = getattr(point, COL)

            # add new value
            if value is not None:
                window.append((current_time, value))
                window_sum += value
                window_count += 1

            # remove old values outside window
            while window and window[0][0] < window_start_time:
                _, old_val = window.popleft()
                window_sum -= old_val
                window_count -= 1

            # compute average
            if window_count > 0:
                avg = window_sum / window_count
            else:
                avg = None

            rolling_averages.append((current_time, avg))

        subindicies[COL + "_FAST"] = rolling_averages

    t3 = datetime.now()
    print(f"Fast rolling average calculated in {t3 - t2}")


    # Plot rolling averages for PM2.5, AQI(PM_2.5) vs AQI_US
    timestamps_25, pm25_rolling = zip(*subindicies["PM2_5_UGM3"])
    aqi_pm25 = [calculate_pollutant_aqi_safar("PM2_5_UGM3", pm) for pm in pm25_rolling]
    timestamps_AQI_IN = [d.last_updated for d in data if getattr(d, "AQI_IN") is not None]
    aqi_in = [d.AQI_IN for d in data]

    timestamps_25, pm25_rolling_fast = zip(*subindicies["PM2_5_UGM3_FAST"])
    aqi_pm25_fast = [calculate_pollutant_aqi_safar("PM2_5_UGM3", pm) for pm in pm25_rolling_fast]

    # Find MAE and RMSE between pm25_rolling_fast and pm25_rolling using timestamps as reference using numpy 
    import numpy as np
    # Align the two series based on timestamps
    aligned_fast = []
    aligned_slow = []
    for t, val in zip(timestamps_25, pm25_rolling):
        if val is not None:
            # Find the corresponding value in the fast series
            idx = np.searchsorted(timestamps_25, t)
            if idx < len(timestamps_25) and timestamps_25[idx] == t:
                aligned_slow.append(val)
                aligned_fast.append(pm25_rolling_fast[idx])
    aligned_slow = np.array(aligned_slow)
    aligned_fast = np.array(aligned_fast)
    mae = np.mean(np.abs(aligned_slow - aligned_fast))
    rmse = np.sqrt(np.mean((aligned_slow - aligned_fast) ** 2))
    print(f"MAE between slow and fast rolling average: {mae}")
    print(f"RMSE between slow and fast rolling average: {rmse}")
    
    


    plt.figure(figsize=(12, 6))

    plt.plot(timestamps_25, pm25_rolling, label="PM2.5 Rolling Average")
    plt.plot(timestamps_25, aqi_pm25, label="AQI_IN(PM2.5)")

    plt.plot(timestamps_25, pm25_rolling_fast, label="PM2.5 Fast Rolling Average") 
    plt.plot(timestamps_25, aqi_pm25_fast, label="AQI_IN(PM2.5)")

    plt.plot(timestamps_AQI_IN, aqi_in, label="AQI_IN", color='orange')

    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.title(f"PM vs AQI_IN for {filters['city']} ({filters['locationId']})")
    plt.legend()
    plt.savefig("fast_vs_slow_rolling_avg.png")




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