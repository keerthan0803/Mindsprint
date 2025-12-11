import os
import json
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from tensorflow import keras
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Crop Recommendation Model
CROP_MODEL = None
CROP_SCALER = None
try:
    # Try ensemble model first, then fall back to regular model
    model_files = ['ensemble_crop_model.pkl', 'crop_recommendation_model.pkl']
    for model_file in model_files:
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                crop_data = pickle.load(f)
                CROP_MODEL = crop_data['model']
                CROP_SCALER = crop_data['scaler']
            print(f'Crop recommendation model loaded from {model_file}')
            break
    if CROP_MODEL is None:
        print('Warning: No crop recommendation model found')
except Exception as e:
    print(f'Error loading crop model: {str(e)}')

# Load Yield Prediction Model
YIELD_MODEL = None
try:
    if os.path.exists('label_encoders.pkl'):
        with open('label_encoders.pkl', 'rb') as f:
            YIELD_MODEL = pickle.load(f)
        print('Yield prediction model loaded successfully')
    else:
        print('Warning: No yield prediction model found')
except Exception as e:
    print(f'Error loading yield model: {str(e)}')

# Helpers for fallbacks
def _default_disease_info(class_names):
    """Build a basic mapping from class label to plant/disease text."""
    info = {}
    for name in class_names:
        # Expected format: Plant___Disease
        if "___" in name:
            plant, disease = name.split("___", 1)
        else:
            plant, disease = name, "Unknown"
        pretty_disease = disease.replace("_", " ")
        info[name] = {"plant": plant.replace("_", " "), "disease": pretty_disease}
    return info

# Load model and data with fallbacks
try:
    MODEL = keras.models.load_model('crop_disease_model.h5')

    # Prefer class_names.pkl but fall back to classes.pkl (used by training script)
    class_files = ['class_names.pkl', 'classes.pkl']
    CLASS_NAMES = None
    for cls_path in class_files:
        if os.path.exists(cls_path):
            with open(cls_path, 'rb') as f:
                CLASS_NAMES = pickle.load(f)
            break
    if CLASS_NAMES is None:
        raise FileNotFoundError('No class names file found (class_names.pkl or classes.pkl)')

    # disease_info.json optional; if missing, build from class names
    if os.path.exists('disease_info.json'):
        with open('disease_info.json', 'r') as f:
            DISEASE_INFO = json.load(f)
    else:
        print('Warning: disease_info.json not found. Using generated metadata from class names.')
        DISEASE_INFO = _default_disease_info(CLASS_NAMES)

    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_LOADED = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_disease(image_path):
    """Predict disease from image"""
    img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = MODEL.predict(img_array, verbose=0)
    confidence = np.max(predictions) * 100
    class_idx = np.argmax(predictions)
    class_name = CLASS_NAMES[class_idx]
    
    disease_data = DISEASE_INFO.get(class_name, {})
    plant = disease_data.get('plant', 'Unknown')
    disease = disease_data.get('disease', 'Unknown')
    
    return {
        'class': class_name,
        'plant': plant,
        'disease': disease,
        'confidence': float(confidence),
        'all_predictions': {
            CLASS_NAMES[i]: float(pred * 100) 
            for i, pred in enumerate(predictions[0])
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/disease predictor.html')
def disease_predictor():
    return render_template('disease predictor.html')

@app.route('/crop.html')
def crop_recommendation():
    return render_template('crop.html')

@app.route('/price.html')
def crop_prices():
    # Serve price.html from the main folder
    price_html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'price.html')
    with open(price_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return Response(html_content, mimetype='text/html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_disease(filepath)
        
        # Convert image to base64 for display
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        result['image'] = f"data:image/jpeg;base64,{img_data}"
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crop-predict', methods=['POST'])
def crop_predict():
    if CROP_MODEL is None:
        return jsonify({'error': 'Crop recommendation model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Extract features
        N = float(data.get('N', 0))
        P = float(data.get('P', 0))
        K = float(data.get('K', 0))
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        ph = float(data.get('ph', 0))
        rainfall = float(data.get('rainfall', 0))
        
        # Prepare input
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Scale features
        features_scaled = CROP_SCALER.transform(features)
        
        # Predict
        prediction = CROP_MODEL.predict(features_scaled)[0]
        
        # Get prediction probabilities if available
        if hasattr(CROP_MODEL, 'predict_proba'):
            probabilities = CROP_MODEL.predict_proba(features_scaled)[0]
            # Get top 5 predictions
            top_indices = np.argsort(probabilities)[-5:][::-1]
            top_crops = []
            for idx in top_indices:
                crop_name = CROP_MODEL.classes_[idx]
                probability = float(probabilities[idx])
                top_crops.append({
                    'crop': crop_name,
                    'probability': probability,
                    'confidence': probability * 100
                })
        else:
            top_crops = [{'crop': prediction, 'probability': 1.0, 'confidence': 100.0}]
        
        return jsonify({
            'recommended_crop': prediction,
            'top_predictions': top_crops,
            'input_values': {
                'N': N, 'P': P, 'K': K,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/yeild.html')
def yield_prediction():
    return render_template('yeild.html')

@app.route('/api/yield-predict', methods=['POST'])
def yield_predict():
    try:
        data = request.get_json()
        
        # Extract input features
        rainfall = float(data.get('rainfall', 0))
        soil_quality = float(data.get('soil_quality', 0))
        farm_size = float(data.get('farm_size', 0))
        sunlight = float(data.get('sunlight', 0))
        fertilizer = float(data.get('fertilizer', 0))
        
        # Validate inputs
        if not all([rainfall, soil_quality, farm_size, sunlight, fertilizer]):
            return jsonify({'error': 'All fields are required'}), 400
        
        # Simple yield prediction formula (based on the data characteristics)
        # This is a weighted formula based on typical crop yield factors
        predicted_yield = (
            (rainfall * 0.15) +
            (soil_quality * 25) +
            (farm_size * 0.4) +
            (sunlight * 15) +
            (fertilizer * 0.1)
        )
        
        # Round to 2 decimal places
        predicted_yield = round(predicted_yield, 2)
        
        return jsonify({
            'success': True,
            'predicted_yield': predicted_yield,
            'unit': 'units'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-status', methods=['GET'])
def model_status():
    return jsonify({
        'loaded': MODEL_LOADED,
        'classes': len(CLASS_NAMES) if MODEL_LOADED else 0,
        'total_diseases': len(CLASS_NAMES) if MODEL_LOADED else 0,
        'crop_model_loaded': CROP_MODEL is not None,
        'yield_model_loaded': YIELD_MODEL is not None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)