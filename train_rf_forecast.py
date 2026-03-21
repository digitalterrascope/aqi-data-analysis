import polars as pl
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from aqi_pkg.filters import Filter, DataLoader

# ==============================
# CONFIG
# ==============================
CITY = "Chandigarh"
LOCATION_ID = "PLLODA000245"
FORECAST_HORIZON_HOURS = 1  # Predict 1 hour ahead
MODEL_PATH = "rf_model_forecast.pkl"
MAX_GAP_MINUTES = 120

print(f"""
{'='*60}
PM2.5 FORECASTING MODEL
{'='*60}
Forecast Horizon: {FORECAST_HORIZON_HOURS} hour(s) ahead
Data: {CITY} - {LOCATION_ID}
{'='*60}
""")

# ==============================
# 1. LOAD DATA
# ==============================
print("\n1. Loading data...")

df = DataLoader(
    Filter(city=CITY)
).get_df(
    remove_duplicates=True,
    hourly_data_only=False
)

df = df.sort("last_updated")

if LOCATION_ID is not None:
    df = df.filter(pl.col("locationId") == LOCATION_ID)

print(f"   Total records: {df.shape[0]}")
print(f"   Date range: {df['last_updated'].min()} to {df['last_updated'].max()}")

# ==============================
# 2. IDENTIFY CONTINUOUS BLOCKS
# ==============================
print("\n2. Identifying continuous blocks...")

df = df.with_columns([
    pl.col("last_updated").diff().dt.total_minutes().alias("gap_minutes")
])

df = df.with_columns([
    (pl.col("gap_minutes") > MAX_GAP_MINUTES).cast(pl.Int32).cum_sum().alias("block_id")
])

block_stats = df.group_by("block_id").agg([
    pl.len().alias("record_count"),
    pl.col("last_updated").min().alias("start_time"),
    pl.col("last_updated").max().alias("end_time"),
]).sort("block_id")

block_stats = block_stats.with_columns([
    ((pl.col("end_time") - pl.col("start_time")).dt.total_hours()).cast(pl.Int64).alias("duration_hours")
])

# Keep blocks with enough data
usable_blocks = block_stats.filter(pl.col("record_count") > 100)
print(f"   Usable blocks: {len(usable_blocks)}")

# ==============================
# 3. PROCESS EACH BLOCK
# ==============================
print("\n3. Creating features (using ONLY past data)...")

all_processed_blocks = []

for block_id in usable_blocks["block_id"].to_list():
    print(f"\n   Processing Block {block_id}...")
    block_df = df.filter(pl.col("block_id") == block_id)
    block_df = block_df.sort("last_updated")

    # Convert to pandas
    block_pd = block_df.to_pandas().sort_values('last_updated').reset_index(drop=True)

    # Get median interval
    time_diffs = block_pd['last_updated'].diff().dt.total_seconds().dropna()
    median_interval_seconds = time_diffs.median() if len(time_diffs) > 0 else 300
    median_interval_minutes = median_interval_seconds / 60

    print(f"      Records: {len(block_pd)}")
    print(f"      Median interval: {median_interval_minutes:.1f} minutes")

    # Calculate horizon steps
    horizon_seconds = FORECAST_HORIZON_HOURS * 3600
    horizon_steps = int(np.ceil(horizon_seconds / median_interval_seconds))
    print(f"      Horizon steps: {horizon_steps}")

    # ===== CREATE SHIFTED TARGET =====
    block_pd['target'] = block_pd['PM2_5_UGM3'].shift(-horizon_steps)

    # ===== ONLY USE PAST DATA FOR FEATURES =====

    # 1. PM2.5 lags
    for i in [1, 2, 3]:
        block_pd[f'lag_{i}'] = block_pd['PM2_5_UGM3'].shift(i)

    # 2. PM10 lag (most important)
    if 'PM10_UGM3' in block_pd.columns:
        block_pd['pm10_lag'] = block_pd['PM10_UGM3'].shift(1)

    # 3. Simple rolling average (30 min)
    window_steps = max(1, int(30 / median_interval_minutes))
    rolling_mean = block_pd['PM2_5_UGM3'].rolling(window=window_steps, min_periods=1).mean()
    block_pd['rolling_mean_30min'] = rolling_mean.shift(1)

    # 4. Time features
    block_pd['hour'] = block_pd['last_updated'].dt.hour
    block_pd['minute'] = block_pd['last_updated'].dt.minute
    block_pd['day_of_week'] = block_pd['last_updated'].dt.weekday
    block_pd['month'] = block_pd['last_updated'].dt.month

    # ===== DROP ROWS WITH NULLS =====
    initial_rows = len(block_pd)
    block_pd = block_pd.dropna(subset=['target', 'lag_1'])  # Only require target and lag_1

    print(f"      After dropping nulls: {len(block_pd)} rows (lost {initial_rows - len(block_pd)})")

    if len(block_pd) > 50:
        # Fill remaining nulls in optional features with forward fill, then 0
        for col in ['lag_2', 'lag_3', 'pm10_lag', 'rolling_mean_30min']:
            if col in block_pd.columns:
                # Forward fill (propagate last valid value)
                block_pd[col] = block_pd[col].ffill()
                # Fill any remaining nulls with 0
                block_pd[col] = block_pd[col].fillna(0)

        all_processed_blocks.append(block_pd)
        print(f"      ✅ Usable records: {len(block_pd)}")
    else:
        print(f"      ⚠️ Not enough data after dropping nulls")

# ==============================
# 4. COMBINE ALL BLOCKS
# ==============================
print("\n4. Combining blocks...")

if not all_processed_blocks:
    print("   ❌ No usable data with 1-hour forecast!")
    print("   Trying with 30-minute forecast instead...")

    # Try with 30-minute forecast
    FORECAST_HORIZON_HOURS = 0.5
    MODEL_PATH = "rf_model_forecast_30min.pkl"
    all_processed_blocks = []

    for block_id in usable_blocks["block_id"].to_list():
        block_df = df.filter(pl.col("block_id") == block_id)
        block_df = block_df.sort("last_updated")
        block_pd = block_df.to_pandas().sort_values('last_updated').reset_index(drop=True)

        time_diffs = block_pd['last_updated'].diff().dt.total_seconds().dropna()
        median_interval_seconds = time_diffs.median() if len(time_diffs) > 0 else 300
        median_interval_minutes = median_interval_seconds / 60

        horizon_seconds = 0.5 * 3600  # 30 minutes
        horizon_steps = int(np.ceil(horizon_seconds / median_interval_seconds))

        block_pd['target'] = block_pd['PM2_5_UGM3'].shift(-horizon_steps)

        # Same minimal features
        for i in [1, 2, 3]:
            block_pd[f'lag_{i}'] = block_pd['PM2_5_UGM3'].shift(i)

        if 'PM10_UGM3' in block_pd.columns:
            block_pd['pm10_lag'] = block_pd['PM10_UGM3'].shift(1)

        window_steps = max(1, int(30 / median_interval_minutes))
        rolling_mean = block_pd['PM2_5_UGM3'].rolling(window=window_steps, min_periods=1).mean()
        block_pd['rolling_mean_30min'] = rolling_mean.shift(1)

        block_pd['hour'] = block_pd['last_updated'].dt.hour
        block_pd['minute'] = block_pd['last_updated'].dt.minute
        block_pd['day_of_week'] = block_pd['last_updated'].dt.weekday
        block_pd['month'] = block_pd['last_updated'].dt.month

        block_pd = block_pd.dropna(subset=['target', 'lag_1'])

        if len(block_pd) > 50:
            # Fill nulls in optional features
            for col in ['lag_2', 'lag_3', 'pm10_lag', 'rolling_mean_30min']:
                if col in block_pd.columns:
                    block_pd[col] = block_pd[col].ffill()
                    block_pd[col] = block_pd[col].fillna(0)

            all_processed_blocks.append(block_pd)
            print(f"   Block {block_id}: {len(block_pd)} usable records")

    if not all_processed_blocks:
        print("   Still no data. Exiting.")
        exit(1)

combined_df = pd.concat(all_processed_blocks, ignore_index=True)
print(f"   Total usable records: {len(combined_df)}")

# ==============================
# 5. SELECT NUMERIC FEATURES ONLY
# ==============================
print("\n5. Selecting numeric features only...")

# Define which columns we want (numeric features only)
desired_features = [
    'lag_1', 'lag_2', 'lag_3', 'pm10_lag', 'rolling_mean_30min',
    'hour', 'minute', 'day_of_week', 'month'
]

# Only keep features that exist in the dataframe
feature_cols = [col for col in desired_features if col in combined_df.columns]

# Also add any other numeric columns that might be useful
# But exclude string columns like locationId, city, etc.
numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
# Filter to only the columns we want plus any other numeric that might be useful
feature_cols = [col for col in feature_cols if col in numeric_cols]

print(f"   Features: {feature_cols}")

X = combined_df[feature_cols]
y = combined_df['target']

# ==============================
# 6. TRAIN-TEST SPLIT
# ==============================
print("\n6. Creating train/test split...")

# Sort by timestamp
combined_df = combined_df.sort_values('last_updated')
X = combined_df[feature_cols]
y = combined_df['target']

split_idx = int(0.8 * len(combined_df))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"   Train size: {len(X_train)}")
print(f"   Test size: {len(X_test)}")

# ==============================
# 7. TRAIN MODEL
# ==============================
print("\n7. Training Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 8. EVALUATE
# ==============================
print("\n8. Evaluating...")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
r2 = r2_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"FORECASTING MODEL RESULTS")
print(f"{'='*60}")
print(f"Horizon: {FORECAST_HORIZON_HOURS} hour(s) ahead")
print(f"MAE: {mae:.2f} µg/m³")
print(f"RMSE: {rmse:.2f} µg/m³")
print(f"R² Score: {r2:.3f}")
print(f"{'='*60}")

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\nFeature Importance:")
print(feature_importance)

# ==============================
# 9. SAVE MODEL
# ==============================
print("\n9. Saving model...")

joblib.dump(model, MODEL_PATH)
print(f"   Model saved to {MODEL_PATH}")

# Save metadata
import json
metadata = {
    "model_type": "forecasting",
    "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
    "features": feature_cols,
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2_score": float(r2)
}

with open("forecast_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Forecasting model ready!")