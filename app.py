import numpy as np
from flask import Flask, request, jsonify, render_template

import joblib

app = Flask(__name__)

rf_model = joblib.load('models/nate_random_forest.sav')
svm_model = joblib.load('models/SVM_model.sav')
xgb_model = joblib.load('models/XGBoost_model.sav')

loaded_models = {
    'svm': svm_model,
    'rf': rf_model,
    'xgb': xgb_model
}

def decode(pred):
    if pred == 1: return 'Customer Exits'
    else: return 'Customer Stays'

@app.route('/')
def home():
    result = [{'model': 'SVM', 'prediction': ' '},
                {'model': 'Random Forest', 'prediction': ' '},
                {'model': 'XGBoost', 'prediction': ' '}]
    
    maind = {}
    maind['customer'] = {}
    maind['predictions'] = result

    return render_template('index.html', maind=maind)

@app.route('/predict', methods=['POST'])
def predict():

    values = [x for x in request.form.values()]

    new_array = np.array(values).reshape(1, -1)
    print(new_array)
    print(values)
    
    cols = ['CreditScore',
            'Geography',
            'Gender',
            'Age',
            'Tenure',
            'Balance',
            'NumOfProducts',
            'HasCrCard',
            'IsActiveMember',
            'EstimatedSalary']

    custd = {}
    for k, v in  zip(cols, values):
        custd[k] = v

    yn_val = ['HasCrCard', 'IsActiveMember']
    for val in  yn_val:
        if custd[val] == '1': custd[val] = 'Yes'
        else: custd[val] = 'No'

    predl = []
    for m in loaded_models.values():
        predl.append(decode(m.predict(new_array)[0]))

    result = [
            {'model': 'SVM', 'prediction': predl[0]},
            {'model': 'Random Forest', 'prediction': predl[1]},
            {'model': 'XGBoost', 'prediction': predl[2]}
            ]

    maind = {}
    maind['customer'] = custd
    maind['predictions'] = result

    return render_template('index.html', maind=maind)


if __name__ == "__main__":
    app.run(debug=True)