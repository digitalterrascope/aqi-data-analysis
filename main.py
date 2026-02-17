from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from aqi_pkg.db import get_session
from aqi_pkg.filters import *
from aqi_pkg.plots import *
from aqi_pkg.aqi_standards.in_safar import calculate_pollutant_aqi_safar

measurements = ["AQI_IN", "AQI_US", "CO_PPB", "NO2_PPB", "O3_PPB", "SO2_PPB", "PM1_UGM3", "PM2_5_UGM3", "PM10_UGM3", "H_PERCENT", "T_C", "TVOC_PPM", "Noise_DB"]

def main(session):
    # city = 'Chandiagrh', locationId = '12428' 
    filters = {
        "city": "Chandigarh",
        "locationId": "12428",
        "start": datetime(2026, 2, 1),
        "end": datetime(2026, 2, 14)
    }

    data = session.query(Entry).filter(Entry.city == filters["city"], Entry.locationId == filters["locationId"], Entry.last_updated >= filters["start"], Entry.last_updated <= filters["end"]).all()

    # Calculate rolling average for PM2.5
    AVG_WINDOW = 24 # Hour(s)
    COLS= ["PM2_5_UGM3", "PM10_UGM3"]

    # Datapoints are recorded in a non fixed window, so we will calculate the rolling average based on the timestamp of the entries
    data.sort(key=lambda x: x.last_updated)  # Ensure data is sorted by timestamp
    subindicies = {}

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


    # Plot rolling averages for PM2.5, AQI(PM_2.5) vs AQI_US
    timestamps_25, pm25_rolling = zip(*subindicies["PM2_5_UGM3"])
    timesstamps_10, pm10_rolling = zip(*subindicies["PM10_UGM3"])
    timestamps_AQI_IN = [d.last_updated for d in data if getattr(d, "AQI_IN") is not None]
    aqi_in = [d.AQI_IN for d in data]

    aqi_pm25 = [calculate_pollutant_aqi_safar("PM2_5_UGM3", pm) for pm in pm25_rolling]
    aqi_pm10 = [calculate_pollutant_aqi_safar("PM10_UGM3", pm) for pm in pm10_rolling]

    plt.figure(figsize=(12, 6))

    # plt.plot(timestamps_25, pm25_rolling, label="PM2.5 Rolling Average")
    # plt.plot(timestamps_25, aqi_pm25, label="AQI_IN(PM2.5)")

    plt.plot(timesstamps_10, pm10_rolling, label="PM10 Rolling Average")
    plt.plot(timesstamps_10, aqi_pm10, label="AQI_IN(PM10)",)

    plt.plot(timestamps_AQI_IN, aqi_in, label="AQI_IN", color='orange')

    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.title(f"PM vs AQI_IN for {filters['city']} ({filters['locationId']})")
    plt.legend()
    plt.savefig("pm10_aqi_trend.png")




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