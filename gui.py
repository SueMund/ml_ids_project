import gradio as gr
import pandas as pd
import requests

# === Function: Takes uploaded CSV, sends to Flask API, returns prediction DataFrame ===
def classify_flows(file):
    try:
        df = pd.read_csv(file)
        url = "http://127.0.0.1:5000/predict"  # Make sure Flask API is running
        response = requests.post(url, json=df.to_dict(orient='records'))

        if response.status_code == 200:
            preds = response.json()['predictions']
            df['Prediction'] = preds
            return df
        else:
            return f"API Error {response.status_code}: {response.json().get('error')}"
    except Exception as e:
        return f"Error: {str(e)}"

# === Build the Gradio interface ===
gui = gr.Interface(
    fn=classify_flows,
    inputs=gr.File(label="Upload CSV File"),
    outputs=gr.Dataframe(label="Predictions"),
    title="Machine Learning Intrusion Detection System",
    description="Upload network traffic flow data to detect benign or malicious behavior using a trained Random Forest model.",
)

# Launch the GUI ===
if __name__ == "__main__":
    gui.launch(share=True)
