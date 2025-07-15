import requests
import json
import pandas as pd
from datetime import datetime
import os

# === Load CSV input file ===
df = pd.read_csv('realworld_input.csv')
data = df.to_dict(orient='records')

# === Send POST request to the Flask API ===
url = 'http://127.0.0.1:5000/predict'
response = requests.post(url, json=data)

# === Handle the response ===
if response.status_code == 200:
    predictions = response.json()['predictions']
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Handle single or multiple records
    if isinstance(data, dict):
        output_df = pd.DataFrame([data])
        print(f"Prediction: {predictions[0]}")
    else:
        output_df = pd.DataFrame(data)
        for i, pred in enumerate(predictions):
            print(f"Prediction for flow {i + 1}: {pred}")

    output_df['Prediction'] = predictions
    output_df['Timestamp'] = timestamp

    # === Save or append to CSV ===
    file_exists = os.path.isfile('results.csv')
    output_df.to_csv('results.csv', mode='a', index=False, header=not file_exists)

    print("Prediction(s) logged to results.csv")

else:
    print("Error:", response.status_code)
    print(response.json())