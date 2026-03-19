import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import argparse

# Load trained model
model = tf.keras.models.load_model("model.keras")

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to image")
args = parser.parse_args()

# Load and preprocess image
img_path = args.image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

# Output result
if prediction > 0.5:
    print(f"Prediction: Parasitized ({prediction*100:.2f}%)")
else:
    print(f"Prediction: Uninfected ({(1-prediction)*100:.2f}%)")