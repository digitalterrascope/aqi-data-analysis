"""
True Forecasting Model: Predict PM2.5 H hours into the future
Using ONLY data available at prediction time.
Runs for all Chandigarh nodes and creates comparison table.
USING OPTIMAL HYPERPARAMETERS FROM OPTIMIZATION:
- n_estimators: 200
- max_depth: 15
- min_samples_split: 8
- min_samples_leaf: 2
- max_features: 0.5
"""

import polars as pl
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
import os
import json
warnings.filterwarnings('ignore')

from aqi_pkg.filters import Filter, DataLoader

# ==============================
# CONFIG
# ==============================
CITY = "Chandigarh"
FORECAST_HORIZON_HOURS = 1  # Predict 1 hour ahead
MAX_GAP_MINUTES = 120
MODELS_DIR = "chandigarh_forecast_models"
RESULTS_FILE = "chandigarh_forecast_results.csv"

# All Chandigarh location IDs from the table
LOCATION_IDS = [
    "12428", "13741", "13876", "48F6EE568468", "8856A6EC6FDC",
    "8856A6ED1F6C", "8856A6EEFFB0", "8856A6EF0AEC", "8856A6EF0B4C",
    "8856A6EF21AC", "8856A6FD4054", "8CBFEA374B8C", "8CBFEA374EA4",
    "9C139E7D1574", "ACA7048685AC", "ACA7049ED040", "E4B06332EC44",
    "PLLODA000245", "PLLODA000591", "PLLODA000600", "PLLODA000621",
    "PLLODA000685", "VIR4221"
]

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

def process_location_forecast(location_id):
    """Process a single location for forecasting and return results"""
    print(f"\n{'='*60}")
    print(f"Processing Location: {location_id}")
    print(f"{'='*60}")
    
    try:
        # ==============================
        # 1. LOAD DATA FOR THIS LOCATION
        # ==============================
        print("Loading data...")
        
        df = DataLoader(
            Filter(city=CITY)
        ).get_df(
            remove_duplicates=True,
            hourly_data_only=False
        )
        
        df = df.sort("last_updated")
        df = df.filter(pl.col("locationId") == location_id)
        
        if df.shape[0] == 0:
            print(f"⚠️ No data found for {location_id}")
            return None
        
        print(f"Total records: {df.shape[0]}")
        print(f"Date range: {df['last_updated'].min()} to {df['last_updated'].max()}")
        
        # Check if PM2.5 data exists
        if "PM2_5_UGM3" not in df.columns or df["PM2_5_UGM3"].null_count() == df.shape[0]:
            print(f"⚠️ No PM2.5 data for {location_id}")
            return None
        
        # ==============================
        # 2. IDENTIFY CONTINUOUS BLOCKS
        # ==============================
        print("Identifying continuous blocks...")
        
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
        
        # Keep blocks with enough data (at least 100 records for forecasting)
        usable_blocks = block_stats.filter(pl.col("record_count") > 100)
        
        if len(usable_blocks) == 0:
            print(f"⚠️ No usable blocks with >100 records for {location_id}")
            return None
        
        print(f"Usable blocks: {len(usable_blocks)}")
        
        # ==============================
        # 3. PROCESS EACH BLOCK
        # ==============================
        print("Creating features (using ONLY past data)...")
        all_processed_blocks = []
        
        for block_id in usable_blocks["block_id"].to_list():
            print(f"   Processing Block {block_id}...")
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
            
            # 2. PM10 lag (if available)
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
            block_pd = block_pd.dropna(subset=['target', 'lag_1'])
            
            print(f"      After dropping nulls: {len(block_pd)} rows (lost {initial_rows - len(block_pd)})")
            
            if len(block_pd) > 50:
                # Fill remaining nulls in optional features with forward fill, then 0
                for col in ['lag_2', 'lag_3', 'pm10_lag', 'rolling_mean_30min']:
                    if col in block_pd.columns:
                        block_pd[col] = block_pd[col].ffill()
                        block_pd[col] = block_pd[col].fillna(0)
                
                all_processed_blocks.append(block_pd)
                print(f"      ✅ Usable records: {len(block_pd)}")
            else:
                print(f"      ⚠️ Not enough data after dropping nulls")
        
        if not all_processed_blocks:
            print(f"⚠️ No processed blocks for {location_id}")
            return None
        
        # ==============================
        # 4. COMBINE ALL BLOCKS
        # ==============================
        print("Combining blocks...")
        combined_df = pd.concat(all_processed_blocks, ignore_index=True)
        print(f"Total usable records: {len(combined_df)}")
        
        # ==============================
        # 5. SELECT NUMERIC FEATURES ONLY
        # ==============================
        print("Selecting numeric features...")
        
        # Define which columns we want (numeric features only)
        desired_features = [
            'lag_1', 'lag_2', 'lag_3', 'pm10_lag', 'rolling_mean_30min',
            'hour', 'minute', 'day_of_week', 'month'
        ]
        
        # Only keep features that exist in the dataframe
        feature_cols = [col for col in desired_features if col in combined_df.columns]
        
        # Also add any other numeric columns that might be useful
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col in numeric_cols]
        
        print(f"Features: {feature_cols}")
        
        X = combined_df[feature_cols]
        y = combined_df['target']
        
        # ==============================
        # 6. TRAIN-TEST SPLIT (time-based)
        # ==============================
        print("Creating train/test split...")
        
        # Sort by timestamp
        combined_df = combined_df.sort_values('last_updated')
        X = combined_df[feature_cols]
        y = combined_df['target']
        
        split_idx = int(0.8 * len(combined_df))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Train size: {len(X_train)}")
        print(f"Test size: {len(X_test)}")
        
        # ==============================
        # 7. TRAIN MODEL WITH OPTIMAL HYPERPARAMETERS
        # ==============================
        print("Training Random Forest with optimal parameters...")
        print("   n_estimators: 200")
        print("   max_depth: 15")
        print("   min_samples_split: 8")
        print("   min_samples_leaf: 2")
        print("   max_features: 0.5")
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=8,
            min_samples_leaf=2,
            max_features=0.5,
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # ==============================
        # 8. EVALUATE
        # ==============================
        print("Evaluating...")
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        r2 = r2_score(y_test, y_pred)
        
        # Get top 3 features
        feature_importance = pd.Series(model.feature_importances_, index=feature_cols)
        top_3_features = feature_importance.nlargest(3).index.tolist()
        top_3_importance = feature_importance.nlargest(3).values.tolist()
        
        # ==============================
        # 9. SAVE MODEL AND METADATA
        # ==============================
        model_path = os.path.join(MODELS_DIR, f"{location_id}_forecast_model.pkl")
        joblib.dump(model, model_path)
        
        metadata = {
            "location_id": location_id,
            "model_type": "forecasting",
            "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
            "total_records": int(df.shape[0]),
            "usable_records": int(len(combined_df)),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "num_blocks": len(usable_blocks),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2_score": float(r2),
            "features": feature_cols,
            "top_3_features": top_3_features,
            "top_3_importance": top_3_importance,
            "hyperparameters": {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 8,
                "min_samples_leaf": 2,
                "max_features": 0.5
            }
        }
        
        metadata_path = os.path.join(MODELS_DIR, f"{location_id}_forecast_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved for {location_id}")
        print(f"   MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
        print(f"   Top features: {top_3_features[0]}, {top_3_features[1]}, {top_3_features[2]}")
        
        return {
            "location_id": location_id,
            "total_records": int(df.shape[0]),
            "usable_records": int(len(combined_df)),
            "num_blocks": len(usable_blocks),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "top_1_feature": top_3_features[0],
            "top_1_importance": top_3_importance[0],
            "top_2_feature": top_3_features[1] if len(top_3_features) > 1 else "N/A",
            "top_2_importance": top_3_importance[1] if len(top_3_importance) > 1 else 0,
            "top_3_feature": top_3_features[2] if len(top_3_features) > 2 else "N/A",
            "top_3_importance": top_3_importance[2] if len(top_3_importance) > 2 else 0
        }
        
    except Exception as e:
        print(f"❌ Error processing {location_id}: {str(e)}")
        return None

# ==============================
# MAIN EXECUTION
# ==============================
print("="*80)
print("PM2.5 FORECASTING MODEL - ALL CHANDIGARH NODES")
print("="*80)
print(f"Forecast Horizon: {FORECAST_HORIZON_HOURS} hour(s) ahead")
print(f"OPTIMAL HYPERPARAMETERS: n_estimators=200, max_depth=15, min_samples_split=8, min_samples_leaf=2, max_features=0.5")
print(f"Total locations to process: {len(LOCATION_IDS)}")
print(f"Models will be saved to: {MODELS_DIR}/")
print("="*80)

results = []
for idx, location_id in enumerate(LOCATION_IDS, 1):
    print(f"\nProgress: {idx}/{len(LOCATION_IDS)}")
    result = process_location_forecast(location_id)
    if result:
        results.append(result)

# ==============================
# CREATE SUMMARY TABLE
# ==============================
print("\n" + "="*80)
print("SUMMARY TABLE - PM2.5 FORECASTING MODEL (1-HOUR AHEAD)")
print("Chandigarh - All Nodes Performance")
print("="*80)

# Create DataFrame for results
if results:
    results_df = pd.DataFrame(results)
    
    # Sort by R² score descending
    results_df = results_df.sort_values('r2', ascending=False)
    
    # Print formatted table
    print("\n{:<15} {:>10} {:>10} {:>10} {:>10} {:>20} {:>20} {:>20}".format(
        "Location ID", "MAE", "RMSE", "R²", "Records", "Top 1 Feature", "Top 2 Feature", "Top 3 Feature"
    ))
    print("-" * 130)
    
    for _, row in results_df.iterrows():
        print("{:<15} {:>10.2f} {:>10.2f} {:>10.3f} {:>10,} {:>20} {:>20} {:>20}".format(
            row['location_id'],
            row['mae'],
            row['rmse'],
            row['r2'],
            row['usable_records'],
            f"{row['top_1_feature']} ({row['top_1_importance']:.3f})",
            f"{row['top_2_feature']} ({row['top_2_importance']:.3f})" if row['top_2_feature'] != "N/A" else "N/A",
            f"{row['top_3_feature']} ({row['top_3_importance']:.3f})" if row['top_3_feature'] != "N/A" else "N/A"
        ))
    
    # Save to CSV
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\n✅ Results saved to: {RESULTS_FILE}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS - FORECASTING MODELS")
    print("="*80)
    print(f"Total nodes successfully processed: {len(results_df)}")
    print(f"Average MAE: {results_df['mae'].mean():.2f} µg/m³")
    print(f"Average RMSE: {results_df['rmse'].mean():.2f} µg/m³")
    print(f"Average R²: {results_df['r2'].mean():.3f}")
    print(f"Best node (highest R²): {results_df.iloc[0]['location_id']} (R²={results_df.iloc[0]['r2']:.3f})")
    print(f"Worst node (lowest R²): {results_df.iloc[-1]['location_id']} (R²={results_df.iloc[-1]['r2']:.3f})")
    
    # Feature importance summary
    print("\n" + "="*80)
    print("MOST COMMON TOP FEATURES ACROSS NODES (FORECASTING)")
    print("="*80)
    top_features_count = {}
    for _, row in results_df.iterrows():
        for i in range(1, 4):
            feat = row[f'top_{i}_feature']
            if feat != "N/A":
                top_features_count[feat] = top_features_count.get(feat, 0) + 1
    
    for feat, count in sorted(top_features_count.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results_df)) * 100
        print(f"{feat:<30} {count:>3}/{len(results_df)} nodes ({percentage:.1f}%)")
    
    # Comparison with real-time models (if results exist)
    print("\n" + "="*80)
    print("COMPARISON NOTE")
    print("="*80)
    print("Forecasting models (1-hour ahead) typically have:")
    print("  • Higher MAE (worse accuracy) than real-time estimation models")
    print("  • Lower R² scores due to the inherent difficulty of prediction")
    print("  • lag_1 (most recent reading) as the dominant feature")
    print("\nThis is expected because predicting the future is harder than estimating current values!")
    
else:
    print("❌ No nodes were successfully processed!")

print("\n" + "="*80)
print("FORECASTING MODEL TRAINING COMPLETE!")
print("="*80)
