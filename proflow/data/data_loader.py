"""Data load module."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable

from proflow.data.data_imputer import Imputer
from proflow import config


class DataLoader:

    def __init__(self, df_dir: str, label: str):
        self.df_dir = df_dir
        self.label = label
        self.test_partition = config.TEST_SIZE
        self.seed = config.SEED
        self.sampling = config.SAMPLING
        self.df = pd.read_csv(self.df_dir, dtype={"index_oper": "str"}).sample(self.sampling)

    def data_load(
        self, 
        data_imputer: bool,
    ):
        self._data_types_detector()

        train_df, test_df = train_test_split(
            self.df,
            test_size=self.test_partition, 
            random_state=self.seed,
            shuffle=True,
        )

        shape_table = PrettyTable()
        impputer = Imputer(type="iterative")

        shape_table.field_names = ["Partition", "[0]_shape", "[1]_shape"]
        shape_table.add_row(["train_df", train_df.shape[0], train_df.shape[1]])
        shape_table.add_row(["test_df", test_df.shape[0], test_df.shape[1]])

        if data_imputer == True:
            train_df_imputed = impputer.imput_data(train_df.drop([self.label], axis=1))
            test_df_imputed = impputer.imput_data(test_df.drop([self.label], axis=1))

            train_df_imputed = pd.DataFrame(train_df_imputed).assign(label = train_df[self.label].values)
            test_df_imputed = pd.DataFrame(test_df_imputed).assign(label = test_df[self.label].values)
            
            shape_table.add_row(["train_df_imputed", train_df.shape[0], train_df.shape[1]])
            shape_table.add_row(["test_df_imputed", test_df.shape[0], test_df.shape[1]])

            print(shape_table)
            print(train_df_imputed.head())
            return train_df_imputed, test_df_imputed
        
        else:
            print(shape_table)
            print(train_df.head())
            return train_df, test_df

    def _data_types_detector(self):
        columns = self.df.columns
        cardinality_list = []
        unique_list = []
        
        for col in columns:
            cardinality = len(set(self.df[col])) / len(self.df[col])
            cardinality_list.append(cardinality)

            uniqueness = len(set(self.df[col]))
            unique_list.append(uniqueness)
        
        data_table = PrettyTable()
        data_table.field_names = ["Column", "Cardinality", "Uniqueness", "Dtype", "Mean Len"]

        if len(columns) == len(cardinality_list) == len(unique_list):
            for i in range(len(columns)):
                l_temp = self._object_lenght_detector(self.df[columns[i]])
                print("Len:", )
                data_table.add_row(
                    [columns[i], cardinality_list[i], unique_list[i], self.df[columns[i]].dtype, l_temp]
                )
        else:
            print("The lenght of columns and data params are not equal")
        
        print(data_table)

    def _object_lenght_detector(self, col: pd.Series):
        print(type(col.dtype))
        if isinstance(col.dtype, object):
            res = (len(col.iloc[0]), len(col.iloc[100]), len(col.iloc[500])) / 3
            print(res)
            return res
        else:
            return 0

