"""Module for loading data from CSV."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable

from proflow.data.data_imputer import Imputer
from proflow import config


class DataLoader:

    def __init__(self, df_dir: str, dtype: dict, label: str):
        self.df_dir = df_dir
        self.label = label
        self.test_partition = config.TEST_SIZE
        self.seed = config.SEED
        self.sampling = config.SAMPLING
        self.df = pd.read_csv(self.df_dir, dtype=dtype).sample(self.sampling)

    def data_load(self):
        """Loads data and fills in data blanks relative to their type. 
        Returns two sets of data: training and test.
        """

        train_df, test_df = train_test_split(
            self.df,
            test_size=self.test_partition, 
            random_state=self.seed,
            shuffle=True,
        )

        shape_table = PrettyTable()
        impputer = Imputer()

        shape_table.field_names = ["Partition", "[0]_shape", "[1]_shape"]
        
        train_df_imputed = impputer.imput_data(train_df.drop([self.label], axis=1))
        test_df_imputed = impputer.imput_data(test_df.drop([self.label], axis=1))

        train_df_imputed = pd.DataFrame(train_df_imputed).assign(label = train_df[self.label].values)
        test_df_imputed = pd.DataFrame(test_df_imputed).assign(label = test_df[self.label].values)
        
        shape_table.add_row(["train_df", train_df_imputed.shape[0], train_df_imputed.shape[1]])
        shape_table.add_row(["test_df", test_df_imputed.shape[0], test_df_imputed.shape[1]])

        print(shape_table)
        print(train_df_imputed.head())
        return train_df_imputed, test_df_imputed
