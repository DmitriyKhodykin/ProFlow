"""Data representation patterns."""

import numpy
import pandas as pd
import tensorflow as tf


def df_series_to_hash(
    dataframe: pd.DataFrame,
    col_name: str,
    num_of_buckets: int,
):
    pass


def _string_to_hash_bucket(col_name: str, num_of_buckets: int):
    fingerprint = tf.strings.to_hash_bucket_fast(
        col_name,
        num_of_buckets,
    ).numpy()
    return fingerprint


f = tf.feature_column.categorical_column_with_hash_bucket('artist_name', hash_bucket_size=200000)