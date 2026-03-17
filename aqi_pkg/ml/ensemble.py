import polars as pl
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


def df_to_dataset(df, target_col, feature_cols, num_hours_lookback):
    df = df.sort(["locationId", "last_updated"])

    exprs = []

    # Create lag features
    for col in feature_cols:
        for i in range(num_hours_lookback):
            exprs.append(
                pl.col(col)
                .shift(i)
                .over("locationId")
                .alias(f"{col}_{i}h")
            )

    exprs.append(
        pl.col(target_col)
        .shift(-1)
        .over("locationId")
        .alias("target")
    )

    df_out = df.with_columns(exprs)

    lag_cols = [f"{col}_{i}h" for col in feature_cols for i in range(num_hours_lookback)]
    df_out = df_out.select(lag_cols + ["target", "locationId", "last_updated"])

    df_out = df_out.drop_nulls()

    return df_out


def split_per_location(df, test_size=0.2, random_state=42, shuffle=True):
    feature_cols = df.select(
        pl.exclude(["target", "locationId", "last_updated"])
    ).columns

    splits = {}

    for location_id, group in df.group_by("locationId"):
        X = group.select(feature_cols).to_numpy()
        y = group["target"].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle
        )

        splits[location_id] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    return splits

def train_per_location(splits, model, fit_fn=None):
    models = {}

    for location_id, data in splits.items():
        X_train = data["X_train"]
        y_train = data["y_train"]

        # clone model so each location gets a fresh instance bc we are passing in an object
        m = clone(model)

        if fit_fn is not None:
            fit_fn(m, X_train, y_train)
        else:
            m.fit(X_train, y_train)

        models[location_id] = m

    return models


def evaluate_models(models, splits, filepath):
    performance_rows = []

    for location_id, data in splits.items():
        model = models.get(location_id)
        if model is None:
            continue

        X_test = data["X_test"]
        y_true = data["y_test"]

        if len(X_test) == 0:
            continue

        y_pred = model.predict(X_test)

        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        performance_rows.append({
            "locationId": location_id,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
        })

    performance_df = pl.DataFrame(performance_rows).sort("r2", descending=True)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(performance_df["r2"].to_list(), bins=30, edgecolor="black")
    axes[0].set_title("Frequency of R² Scores")
    axes[0].set_xlabel("R²")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(performance_df["mae"].to_list(), bins=30, edgecolor="black")
    axes[1].set_title("Frequency of MAE Scores")
    axes[1].set_xlabel("MAE")
    axes[1].set_ylabel("Frequency")

    axes[2].hist(performance_df["rmse"].to_list(), bins=30, edgecolor="black")
    axes[2].set_title("Frequency of RMSE Scores")
    axes[2].set_xlabel("RMSE")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return performance_df


def run_ensemble_experiment(
    df: pl.DataFrame,
    target_col: str,
    feature_cols: list[str],
    num_hours_lookback: int,
    model,
    min_location_size: int = 2500,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42,
    output_dir: str = "figs/ensemble",
):
    print(f"Initial data shape: {df.shape}")
    
    df = df.filter(
        pl.len().over("locationId") >= min_location_size
    )

    print(f"Filtered data shape: {df.shape}")

    # ---------- Dataset ----------
    dataset = df_to_dataset(
        df,
        target_col=target_col,
        feature_cols=feature_cols,
        num_hours_lookback=num_hours_lookback
    )

    print(f"Dataset shape: {dataset.shape}")

    # ---------- Split ----------
    splits = split_per_location(
        dataset,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )

    # ---------- Train ----------
    print("Training models...")
    models = train_per_location(splits, model)

    # ---------- Filename (cleaned) ----------
    model_name = type(model).__name__
    feature_str = "_".join(feature_cols)

    filename = (
        f"{model_name}"
        f"_target_{target_col}"
        f"_features_{feature_str}"
        f"_lookback_{num_hours_lookback}.png"
    )

    filepath = os.path.join(output_dir, filename)

    # ---------- Evaluate ----------
    print("Evaluating models...")
    performance_df = evaluate_models(models, splits, filepath)

    print(f"Saved results to: {filepath}")

    return {
        "models": models,
        "splits": splits,
        "performance": performance_df,
        "filepath": filepath
    }