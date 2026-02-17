from datetime import datetime
from aqi_pkg.db import get_session
from aqi_pkg.filters import *
from aqi_pkg.plots import *
from aqi_pkg.ml.k_means import *
import folium
from folium.plugins import HeatMap
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import base64
# from aqi_pkg.ml.k_means import k_means_clustering, plot_inertia

measurements = ["AQI_IN", "AQI_US", "CO_PPB", "NO2_PPB", "O3_PPB", "SO2_PPB", "PM1_UGM3", "PM2_5_UGM3", "PM10_UGM3", "H_PERCENT", "T_C", "TVOC_PPM", "Noise_DB"]

def main(session):
    k_means_spatial_clustering(session, 8)

if __name__ == "__main__":
    session = get_session()
    plt.style.use("tableau-colorblind10")
    try:
        main(session)
    finally:
        session.close()
        print("Session closed.")