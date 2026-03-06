# Breakpoint concentrations for all pollutants as per CPCB standards
# Ref: https://cpcb.nic.in/displaypdf.php?id=bmF0aW9uYWwtYWlyLXF1YWxpdHktaW5kZXgvQWJvdXRfQVFJLnBkZg==
import numpy as np

BREAKPOINTS = {
    "AQI_CPCB": [
        (0, 50),
        (51, 100),
        (101, 200),
        (201, 300),
        (301, 400),
        (401, 500)
    ],
    "CO_MGM3": [
        (0, 1.0),
        (1.1, 2.0),
        (2.1, 8.7),
        (10, 14.8),
        (17, 29.7),  
        (34, np.inf)
    ],
    "NO2_UGM3": [
        (0, 40),
        (41, 80),
        (81, 180),
        (181, 280),
        (281, 400),
        (401, np.inf)
    ],
    "O3_UGM3": [
        (0, 50),
        (51, 100),
        (101, 168),
        (169, 208),
        (209, 748),
        (749, np.inf)
    ],
    "SO2_UGM3": [
        (0, 40),
        (41, 80),
        (81, 380),
        (381,800),
        (801, 1600),
        (1601, np.inf)
    ],
    "PM10_UGM3": [
        (0, 50),
        (51, 100),
        (101, 250),
        (251, 350),
        (351, 430),
        (431, np.inf)
    ],
    "PM2_5_UGM3": [
        (0, 30),
        (31, 60),
        (61, 90),
        (91, 120),
        (121, 250),
        (251, np.inf)
    ],
}


def calculate_pollutant_aqi_cpcb(pollutant, concentration):
    if pollutant not in BREAKPOINTS:
        raise ValueError(f"Unsupported pollutant: {pollutant}")
    
    if concentration < 0:
        return None  # invalid concentration
    
    pollutant_breakpoints = BREAKPOINTS[pollutant]
    aqi_breakpoints = BREAKPOINTS["AQI_CPCB"]

    for i, (C_low, C_high) in enumerate(pollutant_breakpoints):
        
        if C_low <= concentration <= C_high:
            AQI_low, AQI_high = aqi_breakpoints[i]

            # Handle last breakpoint (infinite upper bound)
            if np.isinf(C_high):
                return AQI_high

            # Linear interpolation
            aqi = (
                (AQI_high - AQI_low) / (C_high - C_low)
            ) * (concentration - C_low) + AQI_low

            return round(aqi)

    return None

if __name__ == "__main__":
    print(BREAKPOINTS["CO_MGM3"])
    print(BREAKPOINTS["CO_UGM3"])
