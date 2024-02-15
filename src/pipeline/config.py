import json


class Config:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)

    def __getitem__(self, key):
        return self.config[key]
