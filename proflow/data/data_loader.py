"""Data load module."""

import pandas as pd
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable

from proflow.data.data_imputer import Imputer
from proflow import config


class DataLoader:

    def __init__(self):
        self.test_partition = config.TEST_SIZE
        self.seed = config.SEED

    def data_load(
        self, 
        df_dir: str,
        label: str,
        data_imputer: bool,
    ):
        df = pd.read_csv(df_dir, dtype={"index_oper": "str"}).sample(500_000)

        df = df[["index_oper" ,"priority", "weight", "label"]]
        
        train_df, test_df = train_test_split(
            df,
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
            train_df_imputed = impputer.imput_data(train_df.drop([label], axis=1))
            test_df_imputed = impputer.imput_data(test_df.drop([label], axis=1))

            train_df_imputed = pd.DataFrame(train_df_imputed).assign(label = train_df[label].values)
            test_df_imputed = pd.DataFrame(test_df_imputed).assign(label = test_df[label].values)
            
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
        pass

