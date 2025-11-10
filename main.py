import torch
import numpy as np
import pandas as pd
import logging
import os
import argparse

from core.models import MaterialPredictor, MaterialGenerator
from core.data_processor import DataProcessor
from core.predictor import PropertyPredictor
from core.generator import MaterialDesigner
from data.loader import DataLoader
from training.trainer import ModelTrainer, GANTrainer
from utils.config import Config
from utils.helpers import setup_logging, set_seed, create_directories
from api.server import app

class MaterialGen:
    def __init__(self, config_path='config.yaml'):
        self.config = Config(config_path)
        self.logger = setup_logging()
        set_seed(42)
        create_directories()
        
    def train_predictor(self, data_path=None):
        self.logger.info("Training material property predictor...")
        
        data_path = data_path or self.config.get('data.data_path')
        data_loader = DataLoader()
        
        try:
            features, targets = data_loader.load_csv_data(data_path)
        except:
            features, targets = self._create_sample_data()
            
        processor = DataProcessor()
        features_normalized = processor.normalize_features(features)
        
        train_features, train_targets, val_features, val_targets, _, _ = \
            data_loader.split_data(features_normalized, targets)
            
        train_dataset = MaterialDataset(train_features, train_targets)
        val_dataset = MaterialDataset(val_features, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = MaterialPredictor()
        trainer = ModelTrainer(model, train_loader, val_loader, self.config)
        best_loss = trainer.train()
        
        self.logger.info(f"Training completed. Best validation loss: {best_loss:.4f}")
        return model
        
    def train_generator(self):
        self.logger.info("Training material generator...")
        
        generator = MaterialGenerator()
        from core.models import MaterialPredictor as Discriminator
        discriminator = Discriminator(output_dim=1)
        
        sample_data = torch.randn(1000, 256)
        sample_dataset = MaterialDataset(sample_data, sample_data)
        data_loader = DataLoader(sample_dataset, batch_size=32, shuffle=True)
        
        gan_trainer = GANTrainer(generator, discriminator, self.config)
        gan_trainer.train(data_loader, epochs=50)
        
        self.logger.info("Generator training completed")
        return generator
        
    def run_api(self):
        self.logger.info("Starting MaterialGen API...")
        app.run(
            host=self.config.get('api.host', 'localhost'),
            port=self.config.get('api.port', 8000),
            debug=self.config.get('api.debug', True)
        )
        
    def predict(self, material_features):
        predictor = PropertyPredictor()
        return predictor.predict_single_material(material_features)
        
    def generate(self, num_samples=5):
        designer = MaterialDesigner()
        return designer.generate_materials(num_samples)
        
    def _create_sample_data(self):
        self.logger.info("Creating sample data for demonstration...")
        np.random.seed(42)
        n_samples = 1000
        n_features = 256
        
        features = np.random.randn(n_samples, n_features)
        targets = np.random.randn(n_samples, 10)
        
        sample_data = np.column_stack([features, targets.mean(axis=1)])
        columns = [f'feature_{i}' for i in range(n_features)] + ['target']
        df = pd.DataFrame(sample_data, columns=columns)
        df.to_csv('./data/sample_materials.csv', index=False)
        
        return features, targets.mean(axis=1)

def main():
    parser = argparse.ArgumentParser(description='MaterialGen: AI for Advanced Materials Discovery')
    parser.add_argument('--mode', choices=['train', 'api', 'predict', 'generate'], 
                       default='api', help='Operation mode')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    material_gen = MaterialGen(args.config)
    
    if args.mode == 'train':
        material_gen.train_predictor()
        material_gen.train_generator()
    elif args.mode == 'api':
        material_gen.run_api()
    elif args.mode == 'predict':
        sample_features = np.random.randn(256)
        prediction = material_gen.predict(sample_features)
        print(f"Prediction: {prediction}")
    elif args.mode == 'generate':
        materials = material_gen.generate(5)
        print(f"Generated materials shape: {materials.shape}")

if __name__ == '__main__':
    main()