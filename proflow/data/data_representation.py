"""Patterns for data presentation."""

import numpy
import pandas as pd
import tensorflow as tf


def hashed_text_in_column(
    dataframe: pd.DataFrame,
    col_name: str,
    num_of_buckets: int,
):
    """All string values of the series are converted to hash."""
    column = dataframe[col_name].copy()
    column = column.apply(
        lambda x: _string_to_hash(x, num_of_buckets)
    )
    return column


def _string_to_hash(input_string: str, hash_bucket_size: int):
    """Converts string to hash number using deterministic methods.
    If you enter the same input_string you will get the same number.
    """
    hash_value = tf.strings.to_hash_bucket_fast(
        input_string,
        hash_bucket_size,
    ).numpy()
    return hash_value


if __name__ == "__main__":
    s = "Hello world"
    print(f"Hash value for {s}", _string_to_hash(s, hash_bucket_size=1000))
