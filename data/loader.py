import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import os

class DataLoader:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.raw_data = None
        self.processed_data = None
        
    def load_csv_data(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        return pd.read_csv(filepath)
    
    def load_excel_data(self, filename, sheet_name=0):
        filepath = os.path.join(self.data_dir, filename)
        return pd.read_excel(filepath, sheet_name=sheet_name)
    
    def load_json_data(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def create_data_loader(self, features, targets, batch_size=32, shuffle=True):
        dataset = MaterialDataset(features, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def split_data(self, features, targets, train_ratio=0.8, val_ratio=0.1):
        total_size = len(features)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        indices = np.random.permutation(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return (features[train_indices], targets[train_indices],
                features[val_indices], targets[val_indices],
                features[test_indices], targets[test_indices])

class MaterialDataManager:
    def __init__(self):
        self.datasets = {}
        
    def register_dataset(self, name, features, targets):
        self.datasets[name] = (features, targets)
        
    def get_dataset(self, name):
        return self.datasets.get(name, (None, None))