import pandas as pd
from sklearn.model_selection import train_test_split

import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as _config:
    data = yaml.load(_config, Loader=SafeLoader)
    print(data['data']["data_dir"])



df = pd.read_csv("../../datasets/pochta_rf/train_dataset.csv")
print(df.head())
