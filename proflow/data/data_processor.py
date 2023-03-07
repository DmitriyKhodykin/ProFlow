"""Data processor module."""

import numpy
import pandas as pd
import tensorflow as tf

from proflow import config


class DataProcessor:
    
    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe
        self.num_of_buckets = config.BUCKETS_FOR_HASH

    def transform(self) -> None:
        self._text_cols_to_hashed_cols()

    def _text_cols_to_hashed_cols(self) -> pd.DataFrame:
        """All text columns are converted to columns 
        with a determinated hash.
        """
        dataframe = self.dataframe.copy()
        data_types = dataframe.dtypes
        for col_name in dataframe.columns:
            if data_types[col_name] == "object":
                print(col_name, "type:", data_types[col_name])
                dataframe[col_name] = self._hashed_text_in_column(
                    dataframe,
                    col_name,
                    self.num_of_buckets
                )
        return dataframe

    def _hashed_text_in_column(
        self, 
        dataframe: pd.DataFrame, 
        col_name: str, 
        num_of_buckets: int,
    ) -> pd.Series:
        """
        All string values of the series are converted to hash.
        """
        column = dataframe[col_name].copy()
        column = column.apply(
            lambda x: self._string_to_hash(x, num_of_buckets)
        )
        return column

    def _string_to_hash(
        self,
        input_string: str, 
        hash_bucket_size: int,
    ) -> int:
        """Converts string to hash number using deterministic methods.
        If you enter the same input_string you will get the same number.
        """
        hash_value = tf.strings.to_hash_bucket_fast(
            input_string,
            hash_bucket_size,
        ).numpy()
        return hash_value
