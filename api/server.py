from flask import Flask, request, jsonify
import numpy as np
import torch
import sys
import os

sys.path.append('..')
from core.predictor import PropertyPredictor
from core.generator import MaterialDesigner
from utils.helpers import setup_logging

app = Flask(__name__)
logger = setup_logging()

predictor = None
designer = None

def load_models():
    global predictor, designer
    try:
        predictor = PropertyPredictor('./models/predictor.pth')
        designer = MaterialDesigner('./models/generator.pth')
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'MaterialGen API'})

@app.route('/predict', methods=['POST'])
def predict_properties():
    try:
        data = request.get_json()
        features = np.array(data['features'])
        
        if predictor is None:
            load_models()
            
        predictions = predictor.predict_single_material(features)
        
        result = {
            'band_gap': float(predictions[0]),
            'formation_energy': float(predictions[1]),
            'stability': float(predictions[2]),
            'conductivity': float(predictions[3]),
            'hardness': float(predictions[4])
        }
        
        return jsonify({'predictions': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/generate', methods=['POST'])
def generate_materials():
    try:
        data = request.get_json()
        num_samples = data.get('num_samples', 5)
        
        if designer is None:
            load_models()
            
        materials = designer.generate_materials(num_samples)
        
        return jsonify({
            'generated_materials': materials.tolist(),
            'num_samples': num_samples
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/design', methods=['POST'])
def design_material():
    try:
        data = request.get_json()
        target_properties = data.get('target_properties', {})
        num_samples = data.get('num_samples', 1)
        
        if designer is None:
            load_models()
            
        materials = designer.generate_with_constraints(target_properties, num_samples)
        
        return jsonify({
            'designed_materials': materials.tolist(),
            'target_properties': target_properties
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=8000, debug=True)