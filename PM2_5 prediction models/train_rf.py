import polars as pl
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from aqi_pkg.filters import Filter, DataLoader

# ==============================
# CONFIG (CHANGE THIS)
# ==============================
CITY = "Chandigarh"
LOCATION_ID = None   # set to a specific locationId if you want
MODEL_PATH = "rf_model.pkl"

# ==============================
# 1. LOAD DATA
# ==============================
print("Loading data...")

df = DataLoader(
    Filter(city=CITY)
).get_df(
    remove_duplicates=True,
    hourly_data_only=True
)

df = df.sort("last_updated")

# If a specific sensor is provided
if LOCATION_ID is not None:
    df = df.filter(pl.col("locationId") == LOCATION_ID)

print(f"Data shape: {df.shape}")

# ==============================
# 2. FEATURE ENGINEERING
# ==============================
print("Creating features...")

df = df.with_columns([
    # Lag features (VERY IMPORTANT)
    pl.col("PM2_5_UGM3").shift(1).alias("lag1"),
    pl.col("PM2_5_UGM3").shift(2).alias("lag2"),
    pl.col("PM2_5_UGM3").shift(24).alias("lag24"),

    # Time features
    pl.col("last_updated").dt.hour().alias("hour"),
    pl.col("last_updated").dt.weekday().alias("day_of_week"),
])

# Drop rows with nulls from shifting
df = df.drop_nulls()

print(f"After feature engineering: {df.shape}")

# ==============================
# 3. SELECT FEATURES
# ==============================
features = [
    "lag1", "lag2", "lag24",
    "hour", "day_of_week",
    "PM10_UGM3",
    "NO2_PPB",
    "T_C",
    "H_PERCENT"
]

target = "PM2_5_UGM3"

# Convert to pandas (sklearn requirement)
pdf = df.to_pandas()

X = pdf[features]
y = pdf[target]

# ==============================
# 4. TRAIN-TEST SPLIT (TIME BASED)
# ==============================
split = int(0.8 * len(pdf))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ==============================
# 5. TRAIN MODEL
# ==============================
print("Training Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    n_jobs=-1,   # uses all CPU cores on HPC
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 6. EVALUATE
# ==============================
print("Evaluating...")

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"MAE: {mae:.2f}")

# ==============================
# 7. SAVE MODEL
# ==============================
joblib.dump(model, MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")