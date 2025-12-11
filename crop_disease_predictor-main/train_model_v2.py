import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle
import json
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATASET_PATH = r"plantvillage dataset\color"
IMAGE_SIZE = 224
BATCH_SIZE = 64  # Increased for faster processing
EPOCHS = 20  # Reduced epochs for faster training
MODEL_SAVE_PATH = "crop_disease_model.h5"
CLASSES_PATH = "classes.pkl"
HISTORY_PATH = "training_history.json"

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def create_data_generators():
    """
    Create efficient data generators for training and validation
    """
    print("Creating data generators...")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 80% training, 20% validation
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Save class names
    class_names = list(train_generator.class_indices.keys())
    with open(CLASSES_PATH, 'wb') as f:
        pickle.dump(class_names, f)
    
    print(f"\nDataset Information:")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Batch size: {BATCH_SIZE}")
    
    return train_generator, validation_generator, len(class_names)

def build_model(num_classes):
    """
    Build an efficient CNN model using MobileNetV2 as base
    """
    print("\nBuilding model...")
    
    # Use MobileNetV2 as base model (pre-trained on ImageNet)
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Build the complete model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    return model

def train_model():
    """
    Main training function
    """
    print("="*60)
    print("Starting Crop Disease Classification Model Training")
    print("="*60)
    
    # Create data generators
    train_gen, val_gen, num_classes = create_data_generators()
    
    # Build model
    model = build_model(num_classes)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Calculate steps
    steps_per_epoch = train_gen.samples // BATCH_SIZE
    validation_steps = val_gen.samples // BATCH_SIZE
    
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10
    )
    
    # Save training history
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Classes saved to: {CLASSES_PATH}")
    print(f"Training history saved to: {HISTORY_PATH}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Model Evaluation")
    print("="*60)
    
    val_loss, val_accuracy = model.evaluate(val_gen, steps=validation_steps)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    return model, history

if __name__ == "__main__":
    try:
        model, history = train_model()
        print("\n✓ Model training completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
