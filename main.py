import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# CONFIG
# ==============================
DATA_DIR = "data/malaria_dataset_split"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5

# ==============================
# DATA LOADING
# ==============================
def load_data():
    print("Loading dataset...")

    train_dir = os.path.join(DATA_DIR, "Train")
    val_dir = os.path.join(DATA_DIR, "Validation")
    test_dir = os.path.join(DATA_DIR, "Test")

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    val_gen = ImageDataGenerator(rescale=1./255)
    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_data = val_gen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_data, val_data, test_data


# ==============================
# MODEL
# ==============================
def build_model():
    print("Building MobileNetV2 model...")

    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==============================
# TRAINING
# ==============================
def train_model(model, train_data, val_data):
    print("Training model...")

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    return history


# ==============================
# EVALUATION
# ==============================
def evaluate_model(model, test_data):
    print("Evaluating model...")

    preds = model.predict(test_data)
    preds = (preds > 0.5).astype(int)

    y_true = test_data.classes

    print("\nClassification Report:")
    print(classification_report(y_true, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, preds))


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("🚀 Malaria Cell Classification Project")

    train_data, val_data, test_data = load_data()

    model = build_model()

    train_model(model, train_data, val_data)

    evaluate_model(model, test_data)

    print("✅ Done!")