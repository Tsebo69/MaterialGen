import torch
import torch.nn as nn
import torch.nn.functional as F

class MaterialPredictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dims=[512, 256, 128], output_dim=10):
        super(MaterialPredictor, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, output_dim)
        
    def forward(self, x):
        x = self.features(x)
        return self.output(x)

class MaterialGenerator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=256):
        super(MaterialGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(), 
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
        
    def forward(self, z):
        return self.net(z)

class CrystalGraphNN(nn.Module):
    def __init__(self, node_dim=92, edge_dim=64, hidden_dim=256):
        super(CrystalGraphNN, self).__init__()
        self.node_embedding = nn.Embedding(100, node_dim)
        self.edge_network = nn.Sequential(
            nn.Linear(edge_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        self.gcn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, graph_data):
        node_features = self.node_embedding(graph_data['atom_types'])
        edge_features = self.edge_network(graph_data['edge_attr'])
        x = node_features
        for gcn in self.gcn_layers:
            x = F.relu(gcn(x))
        x = torch.mean(x, dim=1)
        return self.output_layer(x)