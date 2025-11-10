import torch
import numpy as np
import torch.nn as nn
from .models import MaterialGenerator

class MaterialDesigner:
    def __init__(self, generator_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = MaterialGenerator()
        self.latent_dim = 100
        if generator_path:
            self.load_generator(generator_path)
        self.generator.to(self.device)
        self.generator.eval()
        
    def load_generator(self, generator_path):
        checkpoint = torch.load(generator_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        
    def generate_materials(self, num_samples, latent_vector=None):
        self.generator.eval()
        with torch.no_grad():
            if latent_vector is None:
                latent_vector = torch.randn(num_samples, self.latent_dim).to(self.device)
            generated_materials = self.generator(latent_vector)
            return generated_materials.cpu().numpy()
    
    def generate_with_constraints(self, target_properties, num_samples=10):
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        z.requires_grad = True
        optimizer = torch.optim.Adam([z], lr=0.01)
        
        for epoch in range(100):
            materials = self.generator(z)
            loss = self._calculate_property_loss(materials, target_properties)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return self.generator(z).detach().cpu().numpy()
    
    def _calculate_property_loss(self, materials, target_properties):
        loss = 0
        return loss

class ConditionalMaterialGenerator:
    def __init__(self, num_classes=10):
        self.generator = MaterialGenerator(latent_dim=100 + num_classes)
        self.num_classes = num_classes
        
    def generate_by_class(self, material_class, num_samples):
        z = torch.randn(num_samples, 100)
        class_embedding = torch.eye(self.num_classes)[material_class].repeat(num_samples, 1)
        latent = torch.cat([z, class_embedding], dim=1)
        return self.generator(latent)