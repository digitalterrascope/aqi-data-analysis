from aqi_pkg.ml.clustering import load_data, select_feature_columns, remove_high_null_columns, filter_valid_rows, scale_features
import polars as pl


def get_heatmap(df):
    feature_cols = select_feature_columns(df)

    feature_cols = remove_high_null_columns(
        df, feature_cols, threshold=0.5
    )

    df_clean = filter_valid_rows(df, feature_cols)

    df_scaled = scale_features(df_clean, feature_cols)

    corr_matrix = df_scaled.corr()
    return corr_matrix

def correlate_data(filter):
    df = load_data(filter)
    print(get_heatmap(df))