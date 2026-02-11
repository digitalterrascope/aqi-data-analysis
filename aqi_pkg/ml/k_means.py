from sklearn.cluster import KMeans
from aqi_pkg.filters import *
import numpy as np
import time
import matplotlib.pyplot as plt

measurements = ["AQI_IN", "AQI_US", "CO_PPB", "NO2_PPB", "O3_PPB", "SO2_PPB", "PM1_UGM3", "PM2_5_UGM3", "PM10_UGM3", "H_PERCENT", "T_C", "TVOC_PPM", "Noise_DB"]

def k_means_clustering(session, K=range(2, 20)):
    data = get_all_measurements(session, measurements, city="Chandigarh". start=datetime(2025, 12, 1), end=datetime(2025, 12, 7))
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
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

        print(f"Fitted KMeans with k={k}, inertia={kmeans.inertia_:.2f}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")


    return K, inertias



def plot_inertia(K, inertias, filename=None):
    plt.figure(figsize=(8, 6))
    plt.plot(K, inertias, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("K-Means Inertia for Different k")
    plt.grid(True)

    plt.savefig(filename or "test_img/k_means_inertia.png")