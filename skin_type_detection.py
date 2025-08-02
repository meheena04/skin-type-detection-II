import tensorflow as tf
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
MODEL_PATH = "skin_type_model.h5"  # Path to the trained model

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_skin_type(image_path):
    """Predicts skin type (Oily or Dry) from an image."""
    # Load and preprocess the image
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    return "Oily" if prediction[0][0] > 0.5 else "Dry"

def display_image_with_prediction(image_path, skin_type):
    """Displays the image with the predicted skin type."""
    # Load the image using OpenCV
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Show the image with predicted label
    plt.figure(figsize=(6, 6))
    plt.imshow(original_img)
    plt.title(f"Predicted Skin Type: {skin_type}", fontsize=14, color='blue')
    plt.axis('off')
    plt.show()

# ====== Run prediction on a sample image ======
if __name__ == "__main__":
    test_image_path = "C:/project sem 6/DL PROJECT/Real-TIme-Skin-Type-Detection-main/gayass/skin-dataset/oily/oily_0d852556c21686e16906_jpg.rf.a2b255a4ff59743f63d37f4f0794e987 - Copy.jpg"  # Change this to your test image
    result = predict_skin_type(test_image_path)
    display_image_with_prediction(test_image_path, result)
