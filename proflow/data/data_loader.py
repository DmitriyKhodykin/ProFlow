"""Data load module."""

import pandas as pd
from sklearn.model_selection import train_test_split

import yaml
from yaml.loader import SafeLoader


class DataLoader:

    def __init__(self):
        pass

    def data_load(self, df_dir: str, label: str):

        with open('proflow/data/data_config.yaml') as _config:
            params = yaml.load(_config, Loader=SafeLoader)
            test_partition = params["data"]["preparation"]["test_size"]
            seed = params["seed"]

        df = pd.read_csv("../../datasets/pochta_rf/train_dataset.csv")
        
        train_df, temp_df = train_test_split(
            df.drop([label], axis=1), 
            df[label],
            test_size=test_partition, 
            random_state=seed,
        )
        
        test_df, valid_df = train_test_split(
            temp_df.drop([label], axis=1),
            temp_df[label],
            test_size=50,
            random_state=seed,
        )
        
        print(train_df.shape, test_df.shape, valid_df.shape)

        return train_df, test_df, valid_df

    def data_types_detector(self):
        pass


if __name__=="__main__":
    c = DataLoader()
    c.data_load("", "label")
