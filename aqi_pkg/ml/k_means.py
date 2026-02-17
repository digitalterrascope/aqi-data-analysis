from sklearn.cluster import KMeans
from aqi_pkg.filters import *
import numpy as np
import time
import matplotlib.pyplot as plt
from sqlalchemy import select
import pandas as pd

measurements = ["AQI_IN", "AQI_US", "CO_PPB", "NO2_PPB", "O3_PPB", "SO2_PPB", "PM1_UGM3", "PM2_5_UGM3", "PM10_UGM3", "H_PERCENT", "T_C", "TVOC_PPM", "Noise_DB"]

def k_means_clustering(session, K=range(2, 20)):
    data = get_all_measurements(session, measurements, city="Chandigarh")
    data = np.array(data)
    print(data.shape)

    for i in range(data.shape[1]):
        print(f"Column {i}: {np.sum(data[:, i] == None)}")

    data = np.delete(data, [1, 2, 6, 11, 12], axis=1)
    print(data.shape)

    data = data[~np.any(data == None, axis=1)]

    data = data.astype(float)
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    for i in range(data.shape[1]):
        print(f"Column {i}: {np.sum(data[:, i] == None)}")

    print(data.shape)

    inertias = []
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

        print(f"Fitted KMeans with k={k}, inertia={kmeans.inertia_:.2f}")

    return K, inertias


def k_means_spatial_clustering(session, K=6):
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
                'Zirakpur',
                'Panchkula'
            ])
        )
    )

    result = session.execute(stmt)

    df = pd.DataFrame(result.mappings().all())
    df['time'] = pd.to_datetime(df["last_updated"])

    target_date = '2-10-2026'
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()
    df = df[df['time'].dt.date == target_date]

    df_avg = df.groupby(['locationId', 'lat', 'lon'], as_index=False)['AQI_IN'].mean()

    coords = df_avg[['lat', 'lon', 'AQI_IN']].values
    coords = (coords - np.mean(coords, axis=0)) / np.std(coords, axis=0)
    kmeans = KMeans(n_clusters=K, random_state=42)
    df_avg['cluster'] = kmeans.fit_predict(coords)

    plt.figure(figsize=(8, 6))
    for cluster in range(K):
        cluster_data = df_avg[df_avg['cluster'] == cluster]
        plt.scatter(
            cluster_data['lon'], cluster_data['lat'],
            label=f'Cluster {cluster}', s=50
        )
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'KMeans Spatial Clustering ({K} clusters) - {target_date}')
    plt.legend()
    plt.savefig('test_img/k_means_cluster')


def plot_inertia(K, inertias, filename=None):
    plt.figure(figsize=(8, 6))
    plt.plot(K, inertias, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("K-Means Inertia for Different k")
    plt.grid(True)

    plt.savefig(filename or "test_img/k_means_inertia.png")