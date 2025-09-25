from flask import Flask, render_template, request, jsonify
import random
from predict import predict_fraud  # Importing the prediction function

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        trans_type = request.form['type']
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        # Automatically assign step and isFlaggedFraud
        step = random.randint(1, 95)  # Random step between 1 and 95
        isFlaggedFraud = 0  # Always set to 0

        # Create transaction dictionary
        transaction_data = {
            "step": step,
            "type": trans_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "isFlaggedFraud": isFlaggedFraud
        }

        # Predict fraud
        prediction = predict_fraud(transaction_data)

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
