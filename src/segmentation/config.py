import json
import os


class Config:
    """Class to handle config.json."""

    def __init__(self, config_path):
        cwd = os.getcwd()
        print(f"{cwd=} {config_path=}")
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # delete items with None values after loading in ctor
        for key, value in self.config['masks'].copy().items():
            if value is None:
                del self.config['masks'][key]

    def get(self, key):
        return self.config[key]

    def get_masks(self):
        return self.config['masks']

    def get_image(self, key):
        return self.config['image'][key]
