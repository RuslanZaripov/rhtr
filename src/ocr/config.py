import yaml


class Config:
    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as stream:
            self.config = yaml.safe_load(stream)

    def get(self, key):
        return self.config.get(key)

    def get_image(self, key):
        return self.config.get('image').get(key)

    def get_optimizer(self, key):
        return self.config.get('optimizer').get(key)

    def get_train(self, key):
        return self.config.get('train').get(key)

    def get_val(self, key):
        return self.config.get('val').get(key)

    def get_test(self, key):
        return self.config.get('test').get(key)

    def get_dataloader(self, key):
        return self.config.get('dataloader').get(key)
