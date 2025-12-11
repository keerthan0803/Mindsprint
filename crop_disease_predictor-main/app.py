import os
import json
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
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

@app.route('/api/model-status', methods=['GET'])
def model_status():
    return jsonify({
        'loaded': MODEL_LOADED,
        'classes': len(CLASS_NAMES) if MODEL_LOADED else 0,
        'total_diseases': len(CLASS_NAMES) if MODEL_LOADED else 0
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
