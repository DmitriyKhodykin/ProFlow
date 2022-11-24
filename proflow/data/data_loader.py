import pandas as pd
from sklearn.model_selection import train_test_split

import yaml
from yaml.loader import SafeLoader




class DataLoader:

    def __init__(self):
        pass

    def data_load(self, df_dir: str, label: str):

        with open('data_config.yaml') as _config:
            params = yaml.load(_config, Loader=SafeLoader)
            train_size = params["data"]["preparation"]["train_size"]

        df = pd.read_csv("../../datasets/pochta_rf/train_dataset.csv")
        x_train, x_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=1-train_size, 
            random_state=42
        )
        return x_train, x_test, y_train, y_test

    def data_types_detector(self):
        pass
