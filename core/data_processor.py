import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import json

class MaterialDataset(Dataset):
    def __init__(self, features, targets, transform=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

class DataProcessor:
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        
    def load_material_data(self, filepath):
        data = pd.read_csv(filepath)
        features = data.drop(['material_id', 'target'], axis=1)
        targets = data['target']
        self.feature_names = features.columns.tolist()
        return features.values, targets.values
    
    def normalize_features(self, features):
        from sklearn.preprocessing import StandardScaler
        self.scalers['features'] = StandardScaler()
        return self.scalers['features'].fit_transform(features)
    
    def process_composition(self, composition_str):
        elements = composition_str.split('_')
        encoding = np.zeros(118)
        for elem in elements:
            parts = elem.split(':')
            if len(parts) == 2:
                atomic_num = int(parts[1])
                encoding[atomic_num-1] = 1
        return encoding
    
    def create_material_fingerprint(self, properties_dict):
        fingerprint = []
        if 'band_gap' in properties_dict:
            fingerprint.append(properties_dict['band_gap'])
        if 'formation_energy' in properties_dict:
            fingerprint.append(properties_dict['formation_energy'])
        if 'density' in properties_dict:
            fingerprint.append(properties_dict['density'])
        return np.array(fingerprint)
    
    def save_processor(self, filepath):
        state = {
            'scalers': self.scalers,
            'feature_names': self.feature_names
        }
        torch.save(state, filepath)
    
    def load_processor(self, filepath):
        state = torch.load(filepath)
        self.scalers = state['scalers']
        self.feature_names = state['feature_names']