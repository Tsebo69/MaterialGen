import torch
from torch.utils.data import Dataset
import numpy as np

class MaterialDataset(Dataset):
    def __init__(self, features, targets, transform=None):
        self.features = torch.FloatTensor(features) if not torch.is_tensor(features) else features
        self.targets = torch.FloatTensor(targets) if not torch.is_tensor(targets) else targets
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

class CrystalDataset(Dataset):
    def __init__(self, crystal_data, targets):
        self.crystal_data = crystal_data
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.crystal_data[idx], self.targets[idx]

class MultiTargetMaterialDataset(Dataset):
    def __init__(self, features, targets_dict):
        self.features = torch.FloatTensor(features)
        self.targets = {}
        for key, value in targets_dict.items():
            self.targets[key] = torch.FloatTensor(value)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        target = {key: value[idx] for key, value in self.targets.items()}
        return feature, target