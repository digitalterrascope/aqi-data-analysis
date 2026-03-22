import polars as pl
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from aqi_pkg.filters import Filter, DataLoader

# ==============================
# CONFIG
# ==============================
CITY = "Chandigarh"
LOCATION_ID = "PLLODA000245"
MODEL_PATH = "rf_model_block_based_fixed.pkl"
MAX_GAP_MINUTES = 120  # Gaps > 2 hours are considered new blocks

# ==============================
# 1. LOAD ALL DATA
# ==============================
print("Loading all data...")

df = DataLoader(
    Filter(city=CITY)
).get_df(
    remove_duplicates=True,
    hourly_data_only=False
)

df = df.sort("last_updated")

if LOCATION_ID is not None:
    df = df.filter(pl.col("locationId") == LOCATION_ID)

print(f"Total records: {df.shape[0]}")
print(f"Date range: {df['last_updated'].min()} to {df['last_updated'].max()}")

# ==============================
# 2. IDENTIFY CONTINUOUS BLOCKS
# ==============================
print("\nIdentifying continuous data blocks...")

# Calculate gaps between records
df = df.with_columns([
    pl.col("last_updated").diff().dt.total_minutes().alias("gap_minutes")
])

# Create block_id: new block when gap > MAX_GAP_MINUTES
df = df.with_columns([
    (pl.col("gap_minutes") > MAX_GAP_MINUTES).cast(pl.Int32).cum_sum().alias("block_id")
])

# Analyze blocks
block_stats = df.group_by("block_id").agg([
    pl.len().alias("record_count"),
    pl.col("last_updated").min().alias("start_time"),
    pl.col("last_updated").max().alias("end_time"),
]).sort("block_id")

# Calculate duration in hours
block_stats = block_stats.with_columns([
    ((pl.col("end_time") - pl.col("start_time")).dt.total_hours()).cast(pl.Int64).alias("duration_hours")
])

print(f"Number of continuous blocks: {len(block_stats)}")

print("\nBlocks with >10 records (usable for training):")
usable_blocks = block_stats.filter(pl.col("record_count") > 10)
print(usable_blocks)

# ==============================
# 3. PROCESS EACH BLOCK SEPARATELY
# ==============================
print("\nProcessing each block separately...")

all_processed_blocks = []

for block_id in usable_blocks["block_id"].to_list():
    print(f"\nProcessing Block {block_id}...")
    block_df = df.filter(pl.col("block_id") == block_id)

    # Calculate duration correctly
    start_time = block_df["last_updated"].min()
    end_time = block_df["last_updated"].max()
    duration_hours = (end_time - start_time).total_seconds() / 3600

    print(f"  Records: {block_df.shape[0]}")
    print(f"  Duration: {duration_hours:.1f} hours")
    print(f"  Date range: {start_time} to {end_time}")

    # Sort within block
    block_df = block_df.sort("last_updated")

    # Create time features
    block_df = block_df.with_columns([
        pl.col("last_updated").dt.hour().alias("hour"),
        pl.col("last_updated").dt.minute().alias("minute"),
        pl.col("last_updated").dt.weekday().alias("day_of_week"),
        pl.col("last_updated").dt.month().alias("month"),
        (pl.col("last_updated").dt.hour() * 60 + pl.col("last_updated").dt.minute()).alias("minutes_since_midnight"),
    ])

    # Create lags WITHIN this block only (using shift)
    block_df = block_df.with_columns([
        pl.col("PM2_5_UGM3").shift(1).alias("lag_1"),
        pl.col("PM2_5_UGM3").shift(2).alias("lag_2"),
        pl.col("PM2_5_UGM3").shift(3).alias("lag_3"),
        pl.col("PM2_5_UGM3").shift(4).alias("lag_4"),
        pl.col("PM2_5_UGM3").shift(5).alias("lag_5"),
        pl.col("PM2_5_UGM3").shift(6).alias("lag_6"),
    ])

    # ========== IMPROVED: Create actual 24-hour lag ==========
    # Instead of arbitrary step counts, create a 24-hour lag by matching time of day
    print(f"  Creating actual 24-hour lag features...")

    # Convert to pandas for time-based operations
    block_pd = block_df.to_pandas().sort_values('last_updated').reset_index(drop=True)

    # Create 24-hour lag (same time yesterday)
    block_pd['lag_24h'] = None
    block_pd['lag_48h'] = None
    block_pd['rolling_mean_24h'] = None

    for idx, row in block_pd.iterrows():
        current_time = row['last_updated']

        # Find reading from exactly 24 hours ago (within 1 hour window)
        target_time_24h = current_time - timedelta(hours=24)
        target_time_48h = current_time - timedelta(hours=48)

        # Look for readings within 1 hour of the target time
        time_window = timedelta(hours=1)

        # 24-hour lag
        mask_24h = (block_pd['last_updated'] >= target_time_24h - time_window) &                    (block_pd['last_updated'] <= target_time_24h + time_window)
        readings_24h = block_pd.loc[mask_24h, 'PM2_5_UGM3']
        if len(readings_24h) > 0:
            # Take the closest reading in time
            time_diffs = np.abs(block_pd.loc[mask_24h, 'last_updated'] - target_time_24h)
            closest_idx = time_diffs.idxmin()
            block_pd.at[idx, 'lag_24h'] = block_pd.loc[closest_idx, 'PM2_5_UGM3']

        # 48-hour lag (same time day before yesterday)
        mask_48h = (block_pd['last_updated'] >= target_time_48h - time_window) &                    (block_pd['last_updated'] <= target_time_48h + time_window)
        readings_48h = block_pd.loc[mask_48h, 'PM2_5_UGM3']
        if len(readings_48h) > 0:
            time_diffs = np.abs(block_pd.loc[mask_48h, 'last_updated'] - target_time_48h)
            closest_idx = time_diffs.idxmin()
            block_pd.at[idx, 'lag_48h'] = block_pd.loc[closest_idx, 'PM2_5_UGM3']

        # Rolling average of past 24 hours (excluding current)
        window_start_24h = current_time - timedelta(hours=24)
        mask_24h_avg = (block_pd['last_updated'] >= window_start_24h) &                        (block_pd['last_updated'] < current_time)
        past_24h_readings = block_pd.loc[mask_24h_avg, 'PM2_5_UGM3']
        if len(past_24h_readings) > 0:
            block_pd.at[idx, 'rolling_mean_24h'] = past_24h_readings.mean()

    # ========== EXISTING ROLLING STATISTICS ==========
    # Calculate rolling means based on actual time windows
    for window_minutes in [15, 30, 60, 120, 360, 720, 1440]:
        rolling_means = []
        rolling_stds = []

        for idx, row in block_pd.iterrows():
            current_time = row['last_updated']
            window_start = current_time - timedelta(minutes=window_minutes)

            # Find readings in the window (excluding current)
            mask = (block_pd['last_updated'] >= window_start) &                    (block_pd['last_updated'] < current_time)
            past_readings = block_pd.loc[mask, 'PM2_5_UGM3']

            if len(past_readings) > 0:
                rolling_means.append(past_readings.mean())
                rolling_stds.append(past_readings.std() if len(past_readings) > 1 else 0)
            else:
                rolling_means.append(np.nan)
                rolling_stds.append(np.nan)

        block_pd[f'rolling_mean_{window_minutes}min'] = rolling_means
        block_pd[f'rolling_std_{window_minutes}min'] = rolling_stds

    # Convert back to polars
    block_df = pl.from_pandas(block_pd)

    # Drop rows with nulls in critical features
    block_df = block_df.drop_nulls(subset=["lag_1", "lag_2", "lag_3", "rolling_mean_15min"])

    if block_df.shape[0] > 0:
        all_processed_blocks.append(block_df)
        print(f"  Usable records after feature engineering: {block_df.shape[0]}")
        print(f"  Records with 24-hour lag available: {block_df['lag_24h'].null_count()}")
    else:
        print(f"  ⚠️ No usable records after feature engineering")

# ==============================
# 4. COMBINE ALL BLOCKS
# ==============================
print("\nCombining all blocks...")

if not all_processed_blocks:
    print("❌ No usable data blocks found!")
    exit(1)

combined_df = pl.concat(all_processed_blocks)
print(f"Combined data shape: {combined_df.shape}")

# ==============================
# 5. SELECT FEATURES
# ==============================
print("\nSelecting features...")

# Original features
lag_features = ["lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6"]
rolling_features = [col for col in combined_df.columns if col.startswith("rolling_mean_") or col.startswith("rolling_std_")]
time_features = ["hour", "minute", "day_of_week", "month", "minutes_since_midnight"]
sensor_features = ["PM10_UGM3", "NO2_PPB", "T_C", "H_PERCENT"]

# NEW: 24-hour lag features (actual same-time-of-day comparisons)
daily_features = ["lag_24h", "lag_48h", "rolling_mean_24h"]

# Combine all available features
all_features = []
for feature_list in [lag_features, rolling_features, time_features, sensor_features, daily_features]:
    all_features.extend([f for f in feature_list if f in combined_df.columns])

print(f"Total features: {len(all_features)}")
print(f"Features: {all_features[:15]}...")  # Show first 15

target = "PM2_5_UGM3"

# ==============================
# 6. PREPARE FOR TRAINING
# ==============================
print("\nPreparing for training...")

# Convert to pandas
pdf = combined_df.to_pandas()

X = pdf[all_features]
y = pdf[target]

# Remove any remaining NaN
valid_mask = ~(X.isna().any(axis=1) | y.isna())
X = X[valid_mask]
y = y[valid_mask]

print(f"Final training data: X={X.shape}, y={y.shape}")

if X.shape[0] == 0:
    print("❌ No valid training data!")
    exit(1)

# ==============================
# 7. TRAIN-TEST SPLIT
# ==============================
# Use random split since blocks are independent
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# ==============================
# 8. TRAIN MODEL
# ==============================
print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
    verbose=1  # Show progress
)

model.fit(X_train, y_train)

# ==============================
# 9. EVALUATE
# ==============================
print("\nEvaluating...")

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(np.mean((y_test - preds) ** 2))

print(f"\n{'='*50}")
print(f"Model Performance:")
print(f"{'='*50}")
print(f"MAE: {mae:.2f} µg/m³")
print(f"RMSE: {rmse:.2f} µg/m³")
print(f"R² Score: {model.score(X_test, y_test):.3f}")

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=all_features).sort_values(ascending=False)
print(f"\nTop 15 Most Important Features:")
print(feature_importance.head(15))

# ==============================
# 10. SAVE MODEL
# ==============================
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# Save block info for reference
block_info = usable_blocks.select(["block_id", "record_count", "duration_hours"]).to_pandas()
block_info.to_csv("block_info.csv", index=False)
print("Block information saved to block_info.csv")

# Save feature importance
feature_importance.to_csv("feature_importance_fixed.csv")
print("Feature importance saved to feature_importance_fixed.csv")

# Save metadata
import json
metadata = {
    "total_records": int(df.shape[0]),
    "usable_records": int(X.shape[0]),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "features": all_features,
    "num_blocks": len(usable_blocks),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2_score": float(model.score(X_test, y_test)),
    "daily_features_included": ["lag_24h", "lag_48h", "rolling_mean_24h"]
}

with open("model_metadata_fixed.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Model metadata saved to model_metadata_fixed.json")

print(f"\n{'='*50}")
print("Training complete with actual 24-hour lag features!")
print(f"{'='*50}")