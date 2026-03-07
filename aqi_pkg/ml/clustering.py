import aqi_pkg as ap
from aqi_pkg.filters import Filter, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_elbow(K, inertias, fname=None):
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertias, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.xticks(K)
    plt.grid()

    if fname:
        plt.savefig(f"{fname}_elbow.png", dpi=300)

    plt.show()

def load_data(filter: Filter | str) -> pl.DataFrame:
    if isinstance(filter, str):
        filter = Filter(locationId=filter)

    loader = DataLoader(filter)
    df = loader.get_df(remove_duplicates=True, hourly_data_only=True)

    if "last_updated" in df.columns:
        df = df.with_columns(
            pl.col("last_updated").cast(pl.Datetime)
        )

    return df


def select_feature_columns(df: pl.DataFrame) -> list[str]:
    # metadata_cols = [
    #     "scrape_id", "lat", "lon",
    #     "locationId", "city", "state", "country",
    #     "last_updated"
    # ]
    # feature_cols = [c for c in df.columns if c not in metadata_cols]

    aqi_cols = ['AQI_CO_MGM3', 'AQI_NO2_UGM3', 'AQI_O3_UGM3', 'AQI_PM10_UGM3', 'AQI_PM2_5_UGM3', 'AQI_SO2_UGM3']
    feature_cols = [c for c in df.columns if c in aqi_cols]

    return feature_cols


def remove_high_null_columns(df: pl.DataFrame, feature_cols: list[str], threshold: float = 0.5) -> list[str]:
    n = df.height

    null_ratios = df.select(
        [(pl.col(c).null_count() / n).alias(c) for c in feature_cols]
    ).row(0)

    valid_cols = [
        col for col, ratio in zip(feature_cols, null_ratios)
        if ratio <= threshold
    ]

    return valid_cols


def filter_valid_rows(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    if not feature_cols:
        print("No feature cols, returning df")
        return df
    
    mask = pl.all_horizontal(
        [pl.col(c).is_not_null() for c in feature_cols]
    )

    return df.filter(mask)


def scale_features(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    return df.select(feature_cols).with_columns(
        [(pl.col(c) - pl.col(c).mean()) / pl.col(c).std()
         for c in feature_cols]
    )


def compute_inertias(df_scaled: pl.DataFrame, k_range=range(2, 25)):
    inertias = []

    for k in k_range:
        labels, kmeans, X_whitened = fit_kmeans(df_scaled, k)
        inertias.append(kmeans.inertia_)

    return list(k_range), inertias


def fit_kmeans(df_scaled: pl.DataFrame, k: int):
    # X = df_scaled.to_numpy()

    # kmeans = KMeans(n_clusters=k, random_state=42)
    # labels = kmeans.fit_predict(X)

    # return labels

    # Transform data to mahanobolis distance 
    X = df_scaled.to_numpy()
    # center data
    mean = X.mean(axis=0)
    X_centered = X - mean
    # covariance matrix
    cov = np.cov(X_centered, rowvar=False)
    # Cholesky decomposition
    L = np.linalg.cholesky(cov)
    # whiten transform
    X_whitened = np.linalg.solve(L, X_centered.T).T
    # run KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_whitened)

    return labels, kmeans, X_whitened


def compute_cluster_metrics(df_scaled: pl.DataFrame, k_range=range(2, 15)):
    X = df_scaled.to_numpy()

    metrics = {
        "k": [],
        "inertia": [],
        "silhouette": [],
        "calinski_harabasz": [],
        "davies_bouldin": []
    }

    for k in k_range:
        labels, kmeans, Xw = fit_kmeans(df_scaled, k)

        metrics["k"].append(k)
        metrics["inertia"].append(kmeans.inertia_)

        if len(set(labels)) > 1:
            metrics["silhouette"].append(
                silhouette_score(Xw, labels)
            )
            metrics["calinski_harabasz"].append(
                calinski_harabasz_score(Xw, labels)
            )
            metrics["davies_bouldin"].append(
                davies_bouldin_score(Xw, labels)
            )
        else:
            metrics["silhouette"].append(np.nan)
            metrics["calinski_harabasz"].append(np.nan)
            metrics["davies_bouldin"].append(np.nan)

    return metrics


def cluster_filter(filter: Filter | str, k: int | None = None, null_threshold: float = 0.5):

    df = load_data(filter)

    # Calculate subindices and metrics
    from aqi_pkg.data_scripts.create_subindicies import convert_units_and_calculate_subindicies, calculate_aqi_metrics
    df = convert_units_and_calculate_subindicies(df)
    df = calculate_aqi_metrics(df)

    feature_cols = select_feature_columns(df)

    feature_cols = remove_high_null_columns(
        df, feature_cols, threshold=null_threshold
    )

    df_clean = filter_valid_rows(df, feature_cols)

    # Nomralize features
    df_scaled = scale_features(df_clean, feature_cols)

    metrics = compute_cluster_metrics(df_scaled)

    if k is None:
        return metrics, plot_k_metrics(metrics)


    labels, _, _ = fit_kmeans(df_scaled, k)

    df_clustered = df_clean.with_columns(
        pl.Series("cluster", labels)
    )

    return df_clustered


def plot_k_metrics(metrics):
    k = metrics["k"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")

    axes[0,0].plot(k, metrics["inertia"], marker="o")
    axes[0,0].set_title("Elbow (Inertia)")

    axes[0,1].plot(k, metrics["silhouette"], marker="o")
    axes[0,1].set_title("Silhouette Score")

    axes[1,0].plot(k, metrics["calinski_harabasz"], marker="o")
    axes[1,0].set_title("Calinski-Harabasz")

    axes[1,1].plot(k, metrics["davies_bouldin"], marker="o")
    axes[1,1].set_title("Davies-Bouldin")

    for ax in axes.flatten():
        ax.set_xlabel("k")
        ax.grid(True)

    return fig


def plot_clusters(filter: Filter | str, k: int, location: str | None = None, display_now: bool = False):
    
    from aqi_pkg.aqi_standards.in_cbcp import BREAKPOINTS as CPCB_BREAKPOINTS

    CPCB_COLORS = [
        "#00e400",  # Good
        "#92d050",  # Satisfactory
        "#ffff00",  # Moderate
        "#ff7e00",  # Poor
        "#ff0000",  # Very Poor
        "#7e0023",  # Severe
    ]

    # Strip "AQI_" prefix to match BREAKPOINTS keys e.g. AQI_CO_MGM3 -> CO_MGM3
    def _cpcb_color(col: str, value) -> str:
        if value is None or (isinstance(value, float) and value != value):
            return "#ffffff"
        key = col.removeprefix("AQI_")
        breakpoints = CPCB_BREAKPOINTS.get(key)
        if breakpoints is None:
            return "#ffffff"
        
        def hex_to_rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        
        def rgb_to_hex(r, g, b):
            return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
        
        def lerp(a, b, t):
            return a + (b - a) * t
        
        for i, (lo, hi) in enumerate(breakpoints):
            if lo <= value <= hi:
                t = (value - lo) / (hi - lo) if hi != lo else 0
                c1 = hex_to_rgb(CPCB_COLORS[i])
                c2 = hex_to_rgb(CPCB_COLORS[min(i + 1, len(CPCB_COLORS) - 1)])
                return rgb_to_hex(lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t))
        
        return CPCB_COLORS[-1]

    df_clustered = cluster_filter(filter, k=k)

    metadata_cols = [
        "scrape_id", "lat", "lon",
        "locationId", "city", "state",
        "country", "last_updated", "cluster"
    ]
    aqi_cols = ['AQI_CO_MGM3', 'AQI_NO2_UGM3', 'AQI_O3_UGM3', 'AQI_PM10_UGM3', 'AQI_PM2_5_UGM3', 'AQI_SO2_UGM3']
    metric_cols = [c for c in df_clustered.columns if c in aqi_cols]

    # ---------------- PIE ----------------
    cluster_counts = (
        df_clustered
        .group_by("cluster")
        .agg(pl.len().alias("count"))
        .sort("cluster")
    )

    # ---------------- MEANS TABLE ----------------
    cluster_means = (
        df_clustered
        .group_by("cluster")
        .agg([pl.col(c).mean().alias(c) for c in metric_cols])
        .sort("cluster")
    )
    cluster_means = cluster_means.select(
        [c for c in cluster_means.columns if cluster_means[c].is_not_null().any()]
    )
    metric_cols = [c for c in metric_cols if c in cluster_means.columns]

    table_header = ["cluster"] + metric_cols
    table_values = (
        [cluster_means["cluster"].to_list()]
        + [cluster_means[c].round(3).to_list() for c in metric_cols]
    )

    # Per-cell fill colors: one list per column
    table_fill_colors = [["#d0d0d0"] * len(cluster_means)]  # cluster id column
    for c in metric_cols:
        table_fill_colors.append([
            _cpcb_color(c, v) for v in cluster_means[c].to_list()
        ])

    # ---------------- WEEKLY PERCENTAGES ----------------
    weekly = (
        df_clustered
        .with_columns(pl.col("last_updated").dt.truncate("1w").alias("week"))
        .group_by(["week", "cluster"])
        .agg(pl.len().alias("count"))
    )
    all_weeks = weekly.select("week").unique()
    all_clusters = df_clustered.select("cluster").unique()
    weekly = (
        all_weeks.join(all_clusters, how="cross")
        .join(weekly, on=["week", "cluster"], how="left")
        .with_columns(pl.col("count").fill_null(0))
        .sort(["week", "cluster"])
        .with_columns(
            (pl.col("count") / pl.sum("count").over("week") * 100).alias("percentage")
        )
    )

    # ---------------- BUILD FIGURE ----------------
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.7, 0.3],
        row_heights=[0.4, 0.6],
        specs=[
            [{"type": "domain"}, {"type": "xy", "rowspan": 2}],
            [{"type": "table"}, None],
        ]
    )

    fig.add_trace(
        go.Pie(
            labels=cluster_counts["cluster"].to_list(),
            values=cluster_counts["count"].to_list(),
            hole=0.5,
            showlegend=False
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=table_header,
                fill_color="#404040",
                font=dict(color="white", size=12),
                align="center",
            ),
            cells=dict(
                values=table_values,
                fill_color=table_fill_colors,
                font=dict(color="black", size=11),
                align="center",
            )
        ),
        row=2, col=1
    )

    for cluster in sorted(df_clustered["cluster"].unique().to_list()):
        subset = weekly.filter(pl.col("cluster") == cluster)
        fig.add_trace(
            go.Bar(
                x=subset["week"].to_list(),
                y=subset["percentage"].to_list(),
                name=f"Cluster {cluster}"
            ),
            row=1, col=2
        )

    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        height=900,
        width=1600,
        title=f"{location} Cluster Overview (k={k})"
    )
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
    if display_now:
        fig.show()
        return df_clustered, fig
    else:
        fig_html = fig.to_html(full_html=False, include_plotlyjs="True")
        return df_clustered, fig_html