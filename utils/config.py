import yaml
import os

class Config:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self):
        return {
            'model': {
                'input_dim': 256,
                'hidden_dims': [512, 256, 128],
                'output_dim': 10,
                'latent_dim': 100
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 100,
                'validation_split': 0.2
            },
            'data': {
                'data_path': './data/materials.csv',
                'feature_columns': [],
                'target_columns': ['target']
            },
            'api': {
                'host': 'localhost',
                'port': 8000,
                'debug': True
            }
        }
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default
    
    def save(self, config_path=None):
        path = config_path or self.config_path
        with open(path, 'w') as f:
            yaml.dump(self.config, f)