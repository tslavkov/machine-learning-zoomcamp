import pickle
from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model2.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in_model:
    model = pickle.load(f_in_model)

with open(dv_file, 'rb') as f_in_dv:
    dv = pickle.load(f_in_dv)

app = Flask('predict_flask_docker')


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
   

    result = {
        'get_credit_probability': float(y_pred)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
