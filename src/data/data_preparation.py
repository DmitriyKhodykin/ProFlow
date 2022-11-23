import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as _config:
    data = yaml.load(_config, Loader=SafeLoader)
    print(data['data']["data_dir"])
