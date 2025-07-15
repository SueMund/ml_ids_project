from flask import Flask, request, jsonify
import pandas as pd
import joblib

# 1: Initialize Flask app
app = Flask(__name__)

# 2: Load the trained Random Forest model
model = joblib.load('models/random_forest_ids_20k.pkl')

# 3: Basic route to test if it's working
@app.route('/')
def home():
    return "IDS Flask API is live!"

# 4: Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        input_data = request.get_json()

        # Handle single record or batch
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])  # One record
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)    # Multiple records
        else:
            return jsonify({'error': 'Input must be a dict or list of dicts'}), 400

        # Clean input
        df.columns = df.columns.str.strip()
        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        df = df.dropna()

        # Predict using model
        predictions = model.predict(df)

        # Return predictions as a list
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 5: Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
