import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

from proflow import config


class Imputer:

    def __init__(self):
        pass

    def imput_data(self, df: pd.DataFrame):      

        # Use the apply() method to detect the data types of each column
        detected_data_types = self.data_types_detector(df)

        # Define a dictionary of fill values for each column based on its data type
        fill_values = {}
        for column, dtype in detected_data_types.items():
            if dtype == 'integer':
                fill_values[column] = 0
            elif dtype == 'floating':
                fill_values[column] = df[column].mean()
            else:
                fill_values[column] = df[column].value_counts().idxmax()

        # Use the fillna() method to fill in the gaps
        df.fillna(value=fill_values, inplace=True)
        return df

    def data_types_detector(self, df: pd.DataFrame):
        data_types = df.apply(
            lambda x: pd.api.types.infer_dtype(x.values)
        )
        return data_types
