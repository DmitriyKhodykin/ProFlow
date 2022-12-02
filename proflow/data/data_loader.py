"""Data load module."""

import pandas as pd
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable

import yaml
from yaml.loader import SafeLoader


class DataLoader:

    def __init__(self):
        with open("../proflow/data/data_config.yaml") as _config:
            params = yaml.load(_config, Loader=SafeLoader)
            self.test_partition = params["data"]["preparation"]["test_size"]
            self.seed = params["seed"]

    def data_load(
        self, 
        df_dir: str,
        split_partitions: int,
        data_imputer: bool,
    ):
        df = pd.read_csv(df_dir, dtype={"index_oper": "str"}).sample(500_000)
        
        train_df, temp_df = train_test_split(
            df,
            test_size=self.test_partition, 
            random_state=self.seed,
            shuffle=True,
        )

        shape_table = PrettyTable()

        if split_partitions == 2:
            shape_table.field_names = ["Partition", "[0]_shape", "[1]_shape"]
            shape_table.add_row(["train_df", train_df.shape[0], train_df.shape[1]])
            shape_table.add_row(["test_df", temp_df.shape[0], temp_df.shape[1]])
            print(shape_table)

            print(train_df.head())
            return train_df, temp_df

        elif split_partitions == 3:
            test_df, valid_df = train_test_split(
                temp_df,
                test_size=0.5,
                random_state=self.seed,
                shuffle=True,
            )

            shape_table.field_names = ["Partition", "[0]_shape", "[1]_shape"]
            shape_table.add_row(["train_df", train_df.shape[0], train_df.shape[1]])
            shape_table.add_row(["test_df", test_df.shape[0], test_df.shape[1]])
            shape_table.add_row(["valid_df", valid_df.shape[0], valid_df.shape[1]])
            print(shape_table)

            print(train_df.head())
            return train_df, test_df, valid_df
        
        else:
            print("The number of partitions should be 2 or 3")

    def _data_types_detector(self):
        pass

