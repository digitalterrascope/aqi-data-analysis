""" 
Module for managing AQI data analysis tags and constants.

This module serves as a centralized location for frequently used shortcut variables
and constants.
"""

# List of cities for all nodes in tricity
TRICITY_CITIES_LIST = ['Chandigarh', 'Mohali', 'Zirakpur', 'Panchkula', 'Sahibzada Ajit Singh Nagar']

# CPCB Nodes lat lon as per our dataset
CPCB_NODES_COORDS = {
    "Sector 22": (30.7356, 76.7757),
    "Sector 53": (30.7199, 76.7386),
    "Sector 25": (30.7515, 76.7629),
    "Sector 6 Panchkula": (30.7058, 76.8532),
    
}

# Use the following map with the aqi_pkg.data_scripts.create_subindicies.convert_units function
UNIT_CONVERSION_MAP = {
    "NO2_PPB": ("NO2_UGM3", 1.88),
    "O3_PPB": ("O3_UGM3", 1.96),
    "SO2_PPB": ("SO2_UGM3", 2.62),
    "CO_PPB": [("CO_MGM3", 1.15 / 1000), ("CO_PPM", 1/1000)],
}