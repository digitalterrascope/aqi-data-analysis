"""
XGBoost Models for PM2.5 Prediction - Separate Model for Each Location
Trains an individual XGBoost model for every sensor location in Chandigarh
"""

import polars as pl
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import pandas as pd
import os
import json
import warnings
warnings.filterwarnings('ignore')

from aqi_pkg.filters import Filter, DataLoader

# ==============================
# CONFIG
# ==============================
CITY = "Chandigarh"
FORECAST_HORIZON_HOURS = 1  # Predict 1 hour ahead
OUTPUT_DIR = "xgboost_models_by_location"
MAX_GAP_MINUTES = 120

# Optimized hyperparameters from tuning
XGB_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 800,
    'min_child_weight': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'reg_alpha': 0,
    'reg_lambda': 1.0,
    'gamma': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
    'early_stopping_rounds': 50
}

print(f"""
{'='*60}
XGBOOST MODELS - SEPARATE MODEL PER LOCATION
{'='*60}
City: {CITY}
Forecast Horizon: {FORECAST_HORIZON_HOURS} hour(s) ahead
Training: Individual model for each sensor location
Output Directory: {OUTPUT_DIR}
{'='*60}
""")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 1. LOAD ALL DATA
# ==============================
print("\n1. Loading data for all locations...")

df = DataLoader(
    Filter(city=CITY)
).get_df(
    remove_duplicates=True,
    hourly_data_only=False
)

df = df.sort("last_updated")
print(f"   Total records: {df.shape[0]}")
print(f"   Date range: {df['last_updated'].min()} to {df['last_updated'].max()}")

# Get unique locations
locations = df['locationId'].unique().to_list()
print(f"   Number of locations: {len(locations)}")

# ==============================
# 2. REMOVE LEAKAGE COLUMNS (KEEP LOCATIONID)
# ==============================
print("\n2. Removing leakage columns...")

remove_columns = [
    'AQI_IN', 'AQI_US',           # Derived from target - LEAKAGE!
    'scrape_id',                   # Metadata
    'lat', 'lon',                  # Static location
    'city', 'state', 'country',    # Static strings
]

df = df.drop(remove_columns)
print(f"   Removed {len(remove_columns)} leakage columns")

# ==============================
# 3. TRAIN SEPARATE MODEL FOR EACH LOCATION
# ==============================
print("\n3. Training separate model for each location...")

all_results = []

for idx, location in enumerate(locations):
    print(f"\n{'='*50}")
    print(f"Location {idx+1}/{len(locations)}: {location}")
    print(f"{'='*50}")

    # Filter data for this location
    loc_df = df.filter(pl.col("locationId") == location)
    loc_df = loc_df.sort("last_updated")

    print(f"   Total records: {len(loc_df)}")

    if len(loc_df) < 100:
        print(f"   ⚠️ Skipping - insufficient data (<100 records)")
        continue

    # ===== IDENTIFY CONTINUOUS BLOCKS =====
    loc_df = loc_df.with_columns([
        pl.col("last_updated").diff().dt.total_minutes().alias("gap_minutes")
    ])

    loc_df = loc_df.with_columns([
        (pl.col("gap_minutes") > MAX_GAP_MINUTES).cast(pl.Int32).cum_sum().alias("block_id")
    ])

    # Process each block
    block_stats = loc_df.group_by("block_id").agg([
        pl.len().alias("record_count")
    ]).sort("block_id")

    usable_blocks = block_stats.filter(pl.col("record_count") > 50)

    all_blocks = []

    for block_id in usable_blocks["block_id"].to_list():
        block_df = loc_df.filter(pl.col("block_id") == block_id)
        block_df = block_df.sort("last_updated")

        block_pd = block_df.to_pandas().sort_values('last_updated').reset_index(drop=True)

        # Get median interval
        time_diffs = block_pd['last_updated'].diff().dt.total_seconds().dropna()

        if len(time_diffs) == 0 or time_diffs.median() == 0:
            median_interval_seconds = 600
        else:
            median_interval_seconds = time_diffs.median()

        median_interval_minutes = median_interval_seconds / 60

        # Horizon steps
        horizon_seconds = FORECAST_HORIZON_HOURS * 3600
        horizon_steps = max(1, int(np.ceil(horizon_seconds / median_interval_seconds)))

        # Target
        block_pd['target'] = block_pd['PM2_5_UGM3'].shift(-horizon_steps)

        # PM2.5 lags
        for steps in [1, 2, 3, 4, 6, 8, 12]:
            block_pd[f'pm25_lag_{steps}'] = block_pd['PM2_5_UGM3'].shift(steps)

        # Rolling statistics
        windows_minutes = [30, 60, 120]
        for window in windows_minutes:
            window_steps = max(2, int(window / median_interval_minutes))
            rolling_mean = block_pd['PM2_5_UGM3'].rolling(window=window_steps, min_periods=1).mean()
            block_pd[f'rolling_mean_{window}min'] = rolling_mean.shift(1)

        # Other sensor lags
        sensors = ['PM10_UGM3', 'NO2_PPB', 'T_C', 'H_PERCENT']
        for sensor in sensors:
            if sensor in block_pd.columns:
                lag_steps_1h = max(1, int(60 / median_interval_minutes))
                block_pd[f'{sensor}_lag_1h'] = block_pd[sensor].shift(lag_steps_1h)

        # Time features
        block_pd['hour'] = block_pd['last_updated'].dt.hour
        block_pd['day_of_week'] = block_pd['last_updated'].dt.weekday
        block_pd['month'] = block_pd['last_updated'].dt.month

        # Cyclical encoding
        block_pd['hour_sin'] = np.sin(2 * np.pi * block_pd['hour'] / 24)
        block_pd['hour_cos'] = np.cos(2 * np.pi * block_pd['hour'] / 24)

        # Weekend flag
        block_pd['is_weekend'] = (block_pd['day_of_week'] >= 5).astype(int)

        # Clean data
        block_pd = block_pd.dropna(subset=['target'])

        exclude_cols = ['last_updated', 'target', 'block_id', 'gap_minutes', 'PM2_5_UGM3', 'locationId']
        feature_cols = [col for col in block_pd.columns if col not in exclude_cols]

        for col in feature_cols:
            if col in block_pd.columns:
                block_pd[col] = block_pd[col].ffill().fillna(0)

        block_pd = block_pd.dropna()

        if len(block_pd) > 50:
            all_blocks.append(block_pd)

    if not all_blocks:
        print(f"   ❌ No usable blocks after processing")
        continue

    # Combine blocks for this location
    location_df = pd.concat(all_blocks, ignore_index=True)
    print(f"   Total usable records: {len(location_df)}")

    # Prepare features
    exclude_cols = ['last_updated', 'target', 'block_id', 'gap_minutes', 'PM2_5_UGM3', 'locationId']
    feature_cols = [col for col in location_df.columns if col not in exclude_cols]
    feature_cols = location_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = location_df[feature_cols]
    y = location_df['target']

    # Time-based split
    location_df_sorted = location_df.sort_values('last_updated')
    X = location_df_sorted[feature_cols]
    y = location_df_sorted['target']

    split_idx = int(0.8 * len(location_df_sorted))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"   Train size: {len(X_train)}")
    print(f"   Test size: {len(X_test)}")

    # Train model
    print(f"   Training XGBoost model...")
    final_params = {k: v for k, v in XGB_PARAMS.items() if k != 'early_stopping_rounds'}

    model = xgb.XGBRegressor(**final_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"   ✅ R² Score: {r2:.4f}")
    print(f"   MAE: {mae:.2f} µg/m³")
    print(f"   RMSE: {rmse:.2f} µg/m³")

    # Save model
    safe_name = location.replace(':', '_').replace('/', '_')
    model_path = os.path.join(OUTPUT_DIR, f"xgboost_{safe_name}.pkl")
    joblib.dump(model, model_path)

    # Save metadata
    metadata = {
        "location_id": location,
        "model_type": "xgboost_per_location",
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "features": feature_cols,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "total_records": int(len(location_df)),
        "r2_score": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "model_parameters": {k: float(v) if isinstance(v, (int, float)) else v
                             for k, v in final_params.items()}
    }

    metadata_path = os.path.join(OUTPUT_DIR, f"xgboost_{safe_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_path = os.path.join(OUTPUT_DIR, f"xgboost_{safe_name}_feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)

    all_results.append({
        "location": location,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_records": len(location_df)
    })

# ==============================
# 4. SUMMARY
# ==============================
print(f"\n{'='*60}")
print("SUMMARY - MODELS BY LOCATION")
print(f"{'='*60}")

if all_results:
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('r2', ascending=False)

    print("\nLocation Performance Summary:")
    print(results_df.to_string(index=False))

    # Save summary
    results_df.to_csv(os.path.join(OUTPUT_DIR, "all_models_summary.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"Total models trained: {len(results_df)}")
    print(f"Average R²: {results_df['r2'].mean():.4f}")
    print(f"Median R²: {results_df['r2'].median():.4f}")
    print(f"Best Location: {results_df.iloc[0]['location']} (R²={results_df.iloc[0]['r2']:.4f})")
    print(f"Worst Location: {results_df.iloc[-1]['location']} (R²={results_df.iloc[-1]['r2']:.4f})")
    print(f"\nModels saved in: {OUTPUT_DIR}/")
    print(f"{'='*60}")
else:
    print("❌ No models were successfully trained!")

print("\n✅ Training complete!")
studentiotlab@cass-node-03:~/aqi-data-analysis>