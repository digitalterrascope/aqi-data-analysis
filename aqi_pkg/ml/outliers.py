from sklearn.cluster import DBSCAN
from aqi_pkg.filters import *
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import select
import pandas as pd
import numpy as np
from scipy.stats import norm

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

from tqdm import tqdm


def outliers(session):
    stmt = (
        select(
            Entry.locationId,
            Entry.last_updated,
            Entry.AQI_IN,
            Entry.lat,
            Entry.lon
        )
        .where(
            Entry.AQI_IN != None,
            Entry.city.in_([
                'Chandigarh',
                'Mohali',
                'Dera Bassi',
                'Sahibzada Ajit Singh Nagar',
                'Zirakpur'
            ])
        )
    )

    result = session.execute(stmt)

    df = pd.DataFrame(result.mappings().all())

    df['time'] = pd.to_datetime(df["last_updated"])
    df['hour'] = df['time'].dt.floor("h")

    df = df.drop_duplicates(subset=["lat", "lon", "hour"])
    results = []

    for _, group in tqdm(df.groupby("hour")):
        if len(group) < 5:
            continue

        X = group[["lat", "lon", "AQI_IN"]]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lof = LocalOutlierFactor(
            n_neighbors=5,
            contamination=0.05
        )

        group = group.copy()
        group["lof_label"] = lof.fit_predict(X_scaled)
        group["lof_score"] = lof.negative_outlier_factor_ * -1

        results.append(group)

    final_df = pd.concat(results)

    summary = (
        final_df
        .assign(is_outlier=lambda x: x["lof_label"] == -1)
        .groupby("locationId")
        .agg(
            avg_lof_score=("lof_score", "mean")
        )
        .reset_index()
    )

    print(summary.to_string())