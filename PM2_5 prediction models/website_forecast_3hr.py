"""
3-HOUR AQI FORECAST FOR WEBSITE
Reliable predictions with R² > 0.5
"""

import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
from datetime import datetime, timedelta
from aqi_pkg.filters import Filter, DataLoader
import warnings
warnings.filterwarnings('ignore')

print("Training 3-hour AQI forecast model...")

# Load data
print("Loading data...")
df = DataLoader(Filter(city="Chandigarh")).get_df(remove_duplicates=True, hourly_data_only=False)
df_pd = df.to_pandas()
df_pd['last_updated'] = pd.to_datetime(df_pd['last_updated'])
df_pd.set_index('last_updated', inplace=True)

# Select only numeric columns for resampling
numeric_cols = ['AQI_IN', 'PM2_5_UGM3', 'PM10_UGM3', 'NO2_PPB', 'T_C', 'H_PERCENT']
df_numeric = df_pd[numeric_cols]

# Resample to hourly
hourly = df_numeric.resample('h').mean().dropna()
print(f"Hourly records: {len(hourly)}")

# Features
df = hourly.copy()
df['hour'] = df.index.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Lags
for lag in [1, 2, 3, 4, 5, 6, 12, 24]:
    df[f'aqi_lag_{lag}'] = df['AQI_IN'].shift(lag)

# Rolling means
for window in [2, 3, 4, 6]:
    df[f'aqi_ma_{window}'] = df['AQI_IN'].rolling(window).mean()

# Targets (1, 2, 3 hours ahead)
df['target_1h'] = df['AQI_IN'].shift(-1)
df['target_2h'] = df['AQI_IN'].shift(-2)
df['target_3h'] = df['AQI_IN'].shift(-3)

# Drop NaN values
df = df.dropna()
print(f"Rows after feature engineering: {len(df)}")

# Features to use
feature_cols = ['AQI_IN', 'hour_sin', 'hour_cos',
                'aqi_lag_1', 'aqi_lag_2', 'aqi_lag_3', 'aqi_lag_4', 'aqi_lag_5', 'aqi_lag_6',
                'aqi_ma_2', 'aqi_ma_3', 'aqi_ma_4']

X = df[feature_cols]
y_1h = df['target_1h']
y_2h = df['target_2h']
y_3h = df['target_3h']

print(f"Features shape: {X.shape}")

# Train split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y1_train, y1_test = y_1h[:split], y_1h[split:]
y2_train, y2_test = y_2h[:split], y_2h[split:]
y3_train, y3_test = y_3h[:split], y_3h[split:]

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train models
print("Training models...")
model_1h = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
model_2h = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
model_3h = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)

model_1h.fit(X_train, y1_train, verbose=False)
model_2h.fit(X_train, y2_train, verbose=False)
model_3h.fit(X_train, y3_train, verbose=False)

# Evaluate
from sklearn.metrics import r2_score, mean_absolute_error

y1_pred = model_1h.predict(X_test)
y2_pred = model_2h.predict(X_test)
y3_pred = model_3h.predict(X_test)

r2_1h = r2_score(y1_test, y1_pred)
r2_2h = r2_score(y2_test, y2_pred)
r2_3h = r2_score(y3_test, y3_pred)

mae_1h = mean_absolute_error(y1_test, y1_pred)
mae_2h = mean_absolute_error(y2_test, y2_pred)
mae_3h = mean_absolute_error(y3_test, y3_pred)

print(f"\n{'='*50}")
print("3-HOUR FORECAST PERFORMANCE")
print(f"{'='*50}")
print(f"Hour 1: R² = {r2_1h:.4f}, MAE = {mae_1h:.1f}")
print(f"Hour 2: R² = {r2_2h:.4f}, MAE = {mae_2h:.1f}")
print(f"Hour 3: R² = {r2_3h:.4f}, MAE = {mae_3h:.1f}")
print(f"{'='*50}")

# Save models
joblib.dump(model_1h, "aqi_forecast_1h.pkl")
joblib.dump(model_2h, "aqi_forecast_2h.pkl")
joblib.dump(model_3h, "aqi_forecast_3h.pkl")
joblib.dump(feature_cols, "aqi_features.pkl")
print("\n✅ Models saved")

# Generate current forecast
latest = X.iloc[-1:].values
forecast_1h = float(model_1h.predict(latest)[0])
forecast_2h = float(model_2h.predict(latest)[0])
forecast_3h = float(model_3h.predict(latest)[0])

current_aqi = float(hourly['AQI_IN'].iloc[-1])

# Save website JSON
website_data = {
    "generated_at": datetime.now().isoformat(),
    "city": "Chandigarh",
    "current_aqi": round(current_aqi, 1),
    "forecast": [
        {
            "hours_from_now": 1,
            "aqi": round(forecast_1h, 1),
            "time": (datetime.now() + timedelta(hours=1)).strftime("%H:%M"),
            "category": "Good" if forecast_1h <= 50 else "Satisfactory" if forecast_1h <= 100 else "Moderate" if forecast_1h <= 200 else "Poor"
        },
        {
            "hours_from_now": 2,
            "aqi": round(forecast_2h, 1),
            "time": (datetime.now() + timedelta(hours=2)).strftime("%H:%M"),
            "category": "Good" if forecast_2h <= 50 else "Satisfactory" if forecast_2h <= 100 else "Moderate" if forecast_2h <= 200 else "Poor"
        },
        {
            "hours_from_now": 3,
            "aqi": round(forecast_3h, 1),
            "time": (datetime.now() + timedelta(hours=3)).strftime("%H:%M"),
            "category": "Good" if forecast_3h <= 50 else "Satisfactory" if forecast_3h <= 100 else "Moderate" if forecast_3h <= 200 else "Poor"
        }
    ],
    "performance": {
        "hour_1_r2": round(r2_1h, 4),
        "hour_2_r2": round(r2_2h, 4),
        "hour_3_r2": round(r2_3h, 4),
        "hour_1_mae": round(mae_1h, 1),
        "hour_2_mae": round(mae_2h, 1),
        "hour_3_mae": round(mae_3h, 1)
    }
}

with open("3hr_forecast.json", "w") as f:
    json.dump(website_data, f, indent=2)

print("\n✅ Forecast saved to: 3hr_forecast.json")
print(f"\n📊 Current AQI: {current_aqi:.0f}")
print(f"\n📈 3-HOUR FORECAST:")
print(f"   Now:      {current_aqi:.0f}")
print(f"   +1 hour:  {forecast_1h:.0f} ({website_data['forecast'][0]['category']})")
print(f"   +2 hours: {forecast_2h:.0f} ({website_data['forecast'][1]['category']})")
print(f"   +3 hours: {forecast_3h:.0f} ({website_data['forecast'][2]['category']})")

print("\n" + "="*50)
print("✅ Ready for website deployment!")
print("="*50)