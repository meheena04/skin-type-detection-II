from flask import Flask, request, jsonify
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import product
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load model and data once
model = tf.keras.models.load_model('skin_type_classifier_1.h5')
df = pd.read_csv(r'C:\project sem 6\DL PROJECT\Real-TIme-Skin-Type-Detection-main\webpage2\skincare_products.csv')
df['inr'] = pd.to_numeric(df['inr'].astype(str).str.replace('₹', '', regex=False).str.strip(), errors='coerce')
class_labels = ['dry', 'oily']

def predict_skin_type(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img, verbose=0)[0][0]
    return class_labels[int(round(prediction))]

def recommend_best_combo(skin_type, budget):
    filtered_df = df[df['suitability'].str.contains(skin_type, case=False, na=False)]
    cleansers = filtered_df[filtered_df['product_type'].str.contains('cleanser', case=False, na=False)]
    serums = filtered_df[filtered_df['product_type'].str.contains('serum', case=False, na=False)]
    moisturisers = filtered_df[filtered_df['product_type'].str.contains('moisturiser', case=False, na=False)]

    best_combo, best_total = None, float('inf')
    for c, s, m in product(cleansers.iterrows(), serums.iterrows(), moisturisers.iterrows()):
        c_row, s_row, m_row = c[1], s[1], m[1]
        total = c_row['inr'] + s_row['inr'] + m_row['inr']
        if total <= budget and total < best_total:
            best_combo = (c_row, s_row, m_row)
            best_total = total

    if best_combo:
        c, s, m = best_combo
        return {
            "cleanser": {"name": c['product_name'], "price": c['inr']},
            "serum": {"name": s['product_name'], "price": s['inr']},
            "moisturiser": {"name": m['product_name'], "price": m['inr']},
            "total": best_total
        }
    else:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    budget = int(request.form['budget'])

    image_bytes = image.read()
    skin_type = predict_skin_type(image_bytes)
    recommendation = recommend_best_combo(skin_type, budget)

    return jsonify({
        "skin_type": skin_type,
        "recommendation": recommendation
    })

if __name__ == '__main__':
    app.run(debug=True)
