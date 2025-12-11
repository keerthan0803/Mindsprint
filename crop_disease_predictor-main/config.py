"""
Configuration file for Crop Disease Predictor
Modify these settings as needed
"""

import os

# ==================== MODEL TRAINING ====================
# Dataset path
DATASET_PATH = r"plantvillage dataset\color"

# Image preprocessing
IMAGE_SIZE = 224  # EfficientNetB0 standard size
BATCH_SIZE = 32   # Reduce to 16 or 8 if out of memory
EPOCHS = 50       # Total epochs (20 frozen + 30 fine-tuning)
VALIDATION_SPLIT = 0.2  # 80% train, 20% validation

# Training phases
PHASE1_EPOCHS = 20      # Frozen base model training
PHASE2_EPOCHS = 30      # Fine-tuning epochs
LEARNING_RATE_PHASE1 = 0.001   # Adam optimizer rate for phase 1
LEARNING_RATE_PHASE2 = 0.0001  # Lower rate for fine-tuning

# ==================== MODEL ARCHITECTURE ====================
BASE_MODEL = "EfficientNetB0"  # Pre-trained backbone
DROPOUT_RATE_1 = 0.5   # First dropout layer
DROPOUT_RATE_2 = 0.3   # Second dropout layer
DENSE_1_UNITS = 256    # First dense layer
DENSE_2_UNITS = 128    # Second dense layer

# ==================== DATA AUGMENTATION ====================
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'shear_range': 0.2,
}

# ==================== WEB APPLICATION ====================
FLASK_DEBUG = True
FLASK_PORT = 5000
FLASK_HOST = '127.0.0.1'
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Upload configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ==================== MODEL FILES ====================
MODEL_FILE = 'crop_disease_model.h5'
CLASS_NAMES_FILE = 'class_names.pkl'
DISEASE_INFO_FILE = 'disease_info.json'

# ==================== LOGGING ====================
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = 'app.log'

# ==================== PERFORMANCE ====================
# GPU configuration
USE_GPU = True
GPU_MEMORY_FRACTION = 0.8  # Use up to 80% of GPU memory

# ==================== PREDICTION ====================
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for predictions (0-1)
TOP_N_PREDICTIONS = 5       # Number of top predictions to show

# ==================== DATABASE (Future) ====================
# SQLALCHEMY_DATABASE_URI = 'sqlite:///predictions.db'
# SQLALCHEMY_TRACK_MODIFICATIONS = False
