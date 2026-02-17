# Breakpoint concentrations for all pollutants as per SAFAR standards
# SAFAR provides breakpoints for gases in ppm and ppb units which are the same as our data
# Ref: https://safar.tropmet.res.in/AQI-47-12-Details


BREAKPOINTS = {
    "AQI_SAFAR": [
        (0, 50),
        (50, 100),
        (100, 200),
        (200, 300),
        (300, 400),
        (400, 500)
    ],
    "CO_PPM": [
        (0, 0.9),
        (1.0, 1.7),
        (1.8, 8.7),
        (8.8, 14.8),
        (14.9, 29.7),  
        (29.8, 40)
    ],
    "NO2_PPB": [
        (0, 21),
        (22, 43),
        (44, 96),
        (97, 149),
        (150, 213),
        (214, 750)
    ],
    "O3_PPB": [
        (0, 25),
        (26, 51),
        (52, 86),
        (87, 106),
        (107, 381),
        (382, 450)
    ],
    "PM10_UGM3": [
        (0, 50),
        (50, 100),
        (100, 250),
        (250, 350),
        (350, 430),
        (430, 700)
    ],
    "PM2_5_UGM3": [
        (0, 30),
        (30, 60),
        (60, 90),
        (90, 120),
        (120, 250),
        (250, 380)
    ],
}

def calculate_pollutant_aqi_safar(pollutant, concentration):
    if pollutant not in BREAKPOINTS:
        raise ValueError(f"Unsupported pollutant: {pollutant}")
    
    breakpoints = BREAKPOINTS[pollutant]
    for i, (C_low, C_high) in enumerate(breakpoints):
        if C_low <= concentration <= C_high:
            AQI_low, AQI_high = BREAKPOINTS["AQI_SAFAR"][i]
            aqi = ((AQI_high - AQI_low) / (C_high - C_low)) * (concentration - C_low) + AQI_low
            return round(aqi)
    
    return None  # Return None if concentration is out of defined range