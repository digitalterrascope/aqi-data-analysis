from aqi_pkg.filters import *
import matplotlib.pyplot as plt
import numpy as np

def plot_locations_on_map(locations, filename=None):
    lats = [lat for _, lat, _ in locations]
    lons = [lon for _, _, lon in locations]

    plt.figure(figsize=(8, 6))
    plt.scatter(lons, lats, alpha=0.7, s=20)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Locations of AQI Measurements")
    plt.grid(True)

    plt.savefig(filename or "test_img/locations_map.png")


def plot_measurement_frequency_hist(measurements, bins, range, filename=None):
    plt.figure(figsize=(8, 6))
    plt.hist(measurements, alpha=0.7, range=range, bins=bins)
    plt.xlabel("Measurement Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Measurement Values")
    plt.grid(True)

    plt.savefig(filename or "test_img/measurement_frequency.png")


def plot_measurement_frequency_polygon(measurements, bins, range, filename=None, normalize=False):
    counts, bin_edges, _ = plt.hist(measurements, bins=bins, range=range, alpha=0)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    if normalize:
        counts = counts / counts.sum()

    plt.figure(figsize=(8, 6))
    plt.plot(bin_midpoints, counts, alpha=1)
    plt.xlabel("Measurement Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Measurement Values (Polygon)")
    
    plt.savefig(filename or "test_img/measurement_frequency_polygon.png")


def plot_multiple_frequency_polygons(measurements, labels, bins, range, filename=None, normalize=False):
    plt.figure(figsize=(8, 6))

    for m, label in zip(measurements, labels):
        counts, bin_edges = np.histogram(m, bins=bins, range=range)
        
        if normalize:
            counts = counts / counts.sum()

        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_midpoints, counts, label=label, )

    plt.xlabel("Measurement Value")
    plt.ylabel("Probability" if normalize else "Frequency")
    plt.title("Distribution of Measurement Values (Polygon)")
    plt.legend()
    plt.savefig(filename or "test_img/multiple_measurement_frequency_polygons.png")



def plot_measurement_over_time(measurements, filename=None):
    timestamps = [m[0] for m in measurements]
    values = [m[1] for m in measurements]
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, measurements, marker='o', linestyle='-', alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Measurement Value")
    plt.title("Measurement Over Time")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.savefig(filename or "test_img/measurement_over_time.png")