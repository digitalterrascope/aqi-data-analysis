import polars as pl
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from aqi_pkg.filters import Filter, DataLoader

CITY = "Chandigarh"
FORECAST_HORIZON_HOURS = 1
MODEL_PATH = "xgboost_single_model.pkl"
OUTPUT_DIR = "xgboost_single_model_results"
MAX_GAP_MINUTES = 120

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
{'='*80}
XGBOOST SINGLE MODEL - PM2.5 FORECASTING
{'='*80}
City: {CITY}
Forecast Horizon: {FORECAST_HORIZON_HOURS} hour(s) ahead
{'='*80}
""")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n1. Loading data...")
df = DataLoader(Filter(city=CITY)).get_df(remove_duplicates=True, hourly_data_only=False)
df = df.sort("last_updated")
print(f"   Total records: {df.shape[0]}")
print(f"   Date range: {df['last_updated'].min()} to {df['last_updated'].max()}")

locations = df['locationId'].unique().to_list()
print(f"   Number of locations: {len(locations)}")

print("\n2. Removing leakage columns...")
remove_columns = ['AQI_IN', 'AQI_US', 'scrape_id', 'lat', 'lon', 'city', 'state', 'country']
df = df.drop(remove_columns)
print(f"   Removed {len(remove_columns)} leakage columns")

print("\n3. Creating location encoding...")
location_to_id = {loc: idx for idx, loc in enumerate(locations)}
id_to_location = {v: k for k, v in location_to_id.items()}
print(f"   Encoded {len(locations)} locations")

print("\n4. Processing all locations...")
all_processed_blocks = []

for idx, location in enumerate(locations):
    print(f"   Processing {idx+1}/{len(locations)}: {location[:12]}...")
    
    loc_df = df.filter(pl.col("locationId") == location)
    loc_df = loc_df.sort("last_updated")
    
    if len(loc_df) < 100:
        print(f"      Skipping - insufficient data")
        continue
    
    loc_df = loc_df.with_columns([
        pl.col("last_updated").diff().dt.total_minutes().alias("gap_minutes")
    ])
    
    loc_df = loc_df.with_columns([
        (pl.col("gap_minutes") > MAX_GAP_MINUTES).cast(pl.Int32).cum_sum().alias("block_id")
    ])
    
    block_stats = loc_df.group_by("block_id").agg([pl.len().alias("record_count")]).sort("block_id")
    usable_blocks = block_stats.filter(pl.col("record_count") > 50)
    
    for block_id in usable_blocks["block_id"].to_list():
        block_df = loc_df.filter(pl.col("block_id") == block_id)
        block_df = block_df.sort("last_updated")
        
        block_pd = block_df.to_pandas().sort_values('last_updated').reset_index(drop=True)
        
        time_diffs = block_pd['last_updated'].diff().dt.total_seconds().dropna()
        median_interval_seconds = time_diffs.median() if len(time_diffs) > 0 and time_diffs.median() != 0 else 600
        median_interval_minutes = median_interval_seconds / 60
        horizon_steps = max(1, int(np.ceil(3600 / median_interval_seconds)))
        
        block_pd['target'] = block_pd['PM2_5_UGM3'].shift(-horizon_steps)
        
        for steps in [1, 2, 3, 4, 6, 8, 12]:
            block_pd[f'pm25_lag_{steps}'] = block_pd['PM2_5_UGM3'].shift(steps)
        
        for window in [30, 60, 120]:
            window_steps = max(2, int(window / median_interval_minutes))
            rolling_mean = block_pd['PM2_5_UGM3'].rolling(window=window_steps, min_periods=1).mean()
            block_pd[f'rolling_mean_{window}min'] = rolling_mean.shift(1)
        
        for sensor in ['PM10_UGM3', 'NO2_PPB', 'T_C', 'H_PERCENT']:
            if sensor in block_pd.columns:
                lag_steps = max(1, int(60 / median_interval_minutes))
                block_pd[f'{sensor}_lag_1h'] = block_pd[sensor].shift(lag_steps)
        
        block_pd['hour'] = block_pd['last_updated'].dt.hour
        block_pd['hour_sin'] = np.sin(2 * np.pi * block_pd['hour'] / 24)
        block_pd['hour_cos'] = np.cos(2 * np.pi * block_pd['hour'] / 24)
        block_pd['is_weekend'] = (block_pd['last_updated'].dt.weekday >= 5).astype(int)
        block_pd['location_id'] = location_to_id[location]
        
        block_pd = block_pd.dropna(subset=['target'])
        exclude_cols = ['last_updated', 'target', 'block_id', 'gap_minutes', 'PM2_5_UGM3', 'locationId']
        feature_cols = [col for col in block_pd.columns if col not in exclude_cols]
        
        for col in feature_cols:
            block_pd[col] = block_pd[col].ffill().fillna(0)
        
        block_pd = block_pd.dropna()
        
        if len(block_pd) > 50:
            all_processed_blocks.append(block_pd)

print("\n5. Combining data...")
combined_df = pd.concat(all_processed_blocks, ignore_index=True)
print(f"   Total usable records: {len(combined_df)}")

print("\n6. Preparing features...")
exclude_cols = ['last_updated', 'target', 'block_id', 'gap_minutes', 'PM2_5_UGM3', 'locationId']
feature_cols = [col for col in combined_df.columns if col not in exclude_cols]
feature_cols = combined_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"   Total features: {len(feature_cols)}")

X = combined_df[feature_cols]
y = combined_df['target']
location_ids = combined_df['location_id']

print("\n7. Creating train/test split...")
combined_df_sorted = combined_df.sort_values('last_updated')
X = combined_df_sorted[feature_cols]
y = combined_df_sorted['target']
location_ids = combined_df_sorted['location_id']

split_idx = int(0.8 * len(combined_df_sorted))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
loc_test = location_ids[split_idx:]

print(f"   Train size: {len(X_train)}")
print(f"   Test size: {len(X_test)}")

print("\n8. Training XGBoost model...")
final_params = {k: v for k, v in XGB_PARAMS.items() if k != 'early_stopping_rounds'}
model = xgb.XGBRegressor(**final_params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("   Training complete!")

print("\n9. Evaluating model...")
y_pred = model.predict(X_test)

global_r2 = r2_score(y_test, y_pred)
global_mae = mean_absolute_error(y_test, y_pred)
global_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
global_mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

test_df = pd.DataFrame({'location_id': loc_test, 'actual': y_test, 'predicted': y_pred})

per_location = []
for loc_id in test_df['location_id'].unique():
    loc_data = test_df[test_df['location_id'] == loc_id]
    if len(loc_data) > 20:
        per_location.append({
            'location': id_to_location[loc_id],
            'test_samples': len(loc_data),
            'r2': r2_score(loc_data['actual'], loc_data['predicted']),
            'mae': mean_absolute_error(loc_data['actual'], loc_data['predicted']),
            'rmse': np.sqrt(mean_squared_error(loc_data['actual'], loc_data['predicted'])),
            'mape': np.mean(np.abs((loc_data['actual'] - loc_data['predicted']) / loc_data['actual'])) * 100
        })

per_location_df = pd.DataFrame(per_location).sort_values('r2', ascending=False)

print(f"\n{'='*80}")
print(f"GLOBAL PERFORMANCE")
print(f"{'='*80}")
print(f"R² Score: {global_r2:.4f}")
print(f"MAE: {global_mae:.2f} µg/m³")
print(f"RMSE: {global_rmse:.2f} µg/m³")
print(f"MAPE: {global_mape:.1f}%")
print(f"{'='*80}")

print(f"\n{'='*80}")
print(f"ALL LOCATIONS - PERFORMANCE SUMMARY")
print(f"{'='*80}")
print(per_location_df.to_string(index=False))
print(f"{'='*80}")

print(f"\n{'='*80}")
print(f"STATISTICS")
print(f"{'='*80}")
print(f"Total locations evaluated: {len(per_location_df)}")
print(f"Average R²: {per_location_df['r2'].mean():.4f}")
print(f"Median R²: {per_location_df['r2'].median():.4f}")
print(f"Best R²: {per_location_df['r2'].max():.4f} ({per_location_df.iloc[0]['location'][:15]})")
print(f"Worst R²: {per_location_df['r2'].min():.4f} ({per_location_df.iloc[-1]['location'][:15]})")
print(f"Locations with R² > 0.7: {len(per_location_df[per_location_df['r2'] > 0.7])}")
print(f"Locations with R² > 0.5: {len(per_location_df[per_location_df['r2'] > 0.5])}")
print(f"{'='*80}")

individual_results_path = "xgboost_models_by_location/all_models_summary.csv"
if os.path.exists(individual_results_path):
    individual_df = pd.read_csv(individual_results_path)
    comparison_df = per_location_df.merge(individual_df[['location', 'r2']], on='location', suffixes=('_single', '_individual'))
    comparison_df['difference'] = comparison_df['r2_single'] - comparison_df['r2_individual']
    better_single = len(comparison_df[comparison_df['difference'] > 0])
    print(f"\n{'='*80}")
    print(f"SINGLE MODEL VS INDIVIDUAL MODELS")
    print(f"{'='*80}")
    print(f"Single Model better: {better_single}/{len(comparison_df)} locations ({better_single/len(comparison_df)*100:.1f}%)")
    print(f"Average improvement: {comparison_df['difference'].mean():.4f}")
    print(f"\nDETAILED COMPARISON:")
    comparison_df_sorted = comparison_df.sort_values('difference', ascending=False)
    print(comparison_df_sorted[['location', 'r2_single', 'r2_individual', 'difference']].to_string(index=False))
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

print("\n10. Analyzing feature importance...")
importance_df = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
print(f"\nTop 10 Features:")
print(importance_df.head(10).to_string(index=False))

print("\n11. Saving model and results...")
joblib.dump(model, MODEL_PATH)
print(f"   Model saved to {MODEL_PATH}")

metadata = {
    "model_type": "xgboost_single_model",
    "city": CITY,
    "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
    "total_locations": len(locations),
    "total_records": int(len(combined_df)),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "global_metrics": {"r2": float(global_r2), "mae": float(global_mae), "rmse": float(global_rmse), "mape": float(global_mape)}
}

with open(os.path.join(OUTPUT_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

per_location_df.to_csv(os.path.join(OUTPUT_DIR, "per_location_results.csv"), index=False)
importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

print("\n12. Creating visualizations...")
plt.style.use('seaborn-v0_8-darkgrid')

fig, ax = plt.subplots(figsize=(10, 8))
top10 = importance_df.head(10)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))
bars = ax.barh(range(len(top10)), top10['importance'].values, color=colors[::-1])
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10['feature'].values)
ax.set_xlabel('Importance')
ax.set_title('Top 10 Features - XGBoost Single Model')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(16, 10))
locations_short = [loc[:12] for loc in per_location_df['location'].values]
colors = ['green' if r2 > 0.7 else 'orange' if r2 > 0.5 else 'red' for r2 in per_location_df['r2'].values]
bars = ax.bar(range(len(per_location_df)), per_location_df['r2'].values, color=colors)
ax.set_xticks(range(len(per_location_df)))
ax.set_xticklabels(locations_short, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_xlabel('Location ID', fontsize=12)
ax.set_title('Per-Location Performance - XGBoost Single Model', fontsize=14)
ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (R²=0.7)')
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (R²=0.5)')
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'per_location_r2.png'), dpi=150, bbox_inches='tight')
plt.close()

if os.path.exists(individual_results_path):
    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(len(comparison_df))
    width = 0.35
    locations_short = [loc[:12] for loc in comparison_df['location'].values]
    ax.bar(x - width/2, comparison_df['r2_single'], width, label='Single Model', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, comparison_df['r2_individual'], width, label='Individual Models', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(locations_short, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_xlabel('Location ID', fontsize=12)
    ax.set_title('Model Comparison: Single Model vs Individual Models', fontsize=14)
    ax.legend()
    ax.set_ylim(-0.5, 1)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

fig, ax = plt.subplots(figsize=(12, 5))
sample_size = min(500, len(y_test))
ax.plot(y_test.values[:sample_size], label='Actual', alpha=0.7, linewidth=1)
ax.plot(y_pred[:sample_size], label='Predicted', alpha=0.7, linewidth=1)
ax.set_xlabel('Time Steps')
ax.set_ylabel('PM2.5 (µg/m³)')
ax.set_title(f'Actual vs Predicted - 1 Hour Forecast')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 5))
residuals = y_test - y_pred
ax.scatter(y_pred, residuals, alpha=0.3, s=10)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted PM2.5 (µg/m³)')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot - XGBoost Single Model')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'residuals.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*80}")
print(f"✅ COMPLETE!")
print(f"{'='*80}")
print(f"Model: {MODEL_PATH}")
print(f"Results: {OUTPUT_DIR}/")
print(f"Global R²: {global_r2:.4f}")
print(f"All location results saved to: {OUTPUT_DIR}/per_location_results.csv")
print(f"{'='*80}")
