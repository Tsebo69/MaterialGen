import torch
import numpy as np
import pandas as pd
from .models import MaterialPredictor, CrystalGraphNN

class PropertyPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MaterialPredictor()
        if model_path:
            self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    def predict_properties(self, material_features):
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(material_features).to(self.device)
            predictions = self.model(features_tensor)
            return predictions.cpu().numpy()
    
    def predict_single_material(self, feature_vector):
        if isinstance(feature_vector, list):
            feature_vector = np.array(feature_vector)
        if len(feature_vector.shape) == 1:
            feature_vector = feature_vector.reshape(1, -1)
        return self.predict_properties(feature_vector)[0]
    
    def batch_predict(self, material_list):
        predictions = self.predict_properties(material_list)
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'material_id': i,
                'band_gap': pred[0],
                'formation_energy': pred[1],
                'stability': pred[2],
                'conductivity': pred[3],
                'hardness': pred[4]
            }
            results.append(result)
        return results

class MultiTargetPredictor:
    def __init__(self):
        self.predictors = {}
        self.target_names = [
            'band_gap', 'formation_energy', 'stability',
            'conductivity', 'hardness', 'thermal_conductivity',
            'youngs_modulus', 'piezoelectric_coeff'
        ]
        
    def add_predictor(self, target_name, predictor):
        self.predictors[target_name] = predictor
        
    def predict_all_properties(self, material_features):
        results = {}
        for target_name, predictor in self.predictors.items():
            results[target_name] = predictor.predict_single_material(material_features)
        return results