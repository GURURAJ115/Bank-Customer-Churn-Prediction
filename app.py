import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load models with error handling
try:
    rf_model = joblib.load('models/nate_random_forest.sav')
    svm_model = joblib.load('models/SVM_model.sav')
    xgb_model = joblib.load('models/XGBoost_model.sav')

    loaded_models = {
        'svm': svm_model,
        'rf': rf_model,
        'xgb': xgb_model
    }
except Exception as e:
    print(f"Error loading models: {e}")
    loaded_models = {}

# Decode prediction results
def decode(pred):
    return 'Customer Exits' if pred == 1 else 'Customer Stays'

@app.route('/')
def home():
    maind = {
        'customer': {},
        'predictions': [
            {'model': 'SVM', 'prediction': ''},
            {'model': 'Random Forest', 'prediction': ''},
            {'model': 'XGBoost', 'prediction': ''}
        ]
    }
    return render_template('index.html', maind=maind)

@app.route('/predict', methods=['POST'])
def predict():
    # Get and validate input
    try:
        values = [x for x in request.form.values()]
        if len(values) != 10:
            return jsonify({"error": "Invalid input length. Expected 10 values."}), 400

        new_array = np.array(values).reshape(1, -1)
    except Exception as e:
        return jsonify({"error": f"Invalid input format: {e}"}), 400

    # Map inputs to customer dictionary
    cols = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    custd = {k: v for k, v in zip(cols, values)}

    # Convert binary fields to Yes/No
    for field in ['HasCrCard', 'IsActiveMember']:
        custd[field] = 'Yes' if custd[field] == '1' else 'No'

    # Generate predictions
    if not loaded_models:
        return jsonify({"error": "Models are not loaded."}), 500

    predl = [decode(m.predict(new_array)[0]) for m in loaded_models.values()]

    result = [
        {'model': 'SVM', 'prediction': predl[0]},
        {'model': 'Random Forest', 'prediction': predl[1]},
        {'model': 'XGBoost', 'prediction': predl[2]}
    ]

    maind = {
        'customer': custd,
        'predictions': result
    }

    return render_template('index.html', maind=maind)

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1", "yes"]
    app.run(debug=debug_mode)
