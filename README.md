**Machine Learning-Based Intrusion Detection System (IDS)**
This project is a machine learning-based Intrusion Detection System that uses a Random Forest classifier to detect malicious network traffic. It includes a Flask API and a Gradio GUI for demonstration and testing.


**Features**
Trained using the CICIDS2017 dataset

Random Forest classifier for classification

Flask API to serve model predictions

Gradio GUI for local or public interactive use (with share=True)

Model is saved using joblib for reuse

Includes evaluation with accuracy, precision, recall, and F1 score


**How to Run the Project**
Install the required dependencies by running:

pip install -r requirements.txt

Run the Flask API:

python api.py

The API will be available at: http://127.0.0.1:5000/

Run the Gradio GUI:

python gui.py

The GUI already uses share=True, so it will generate a public URL for remote testing or demonstration.


**Files in This Project**
train.py – Trains and saves the Random Forest model

evaluate.py – Evaluates the model’s accuracy, precision, recall, and F1 score

api.py – Flask-based REST API to serve predictions

gui.py – Gradio GUI for uploading and testing CSV samples

model.joblib – The saved trained model

requirements.txt – Lists all required Python packages

.gitignore – Prevents large dataset files from being committed

README.md – This file

All files are located in a single directory.


**Dataset Used**
This project uses the CICIDS2017 dataset from the Canadian Institute for Cybersecurity. Due to GitHub's file size limits, the dataset is not included in this repository.

You can download the dataset from:

https://www.unb.ca/cic/datasets/ids-2017.html

Once downloaded, place the required CSV files inside a folder named data/. This folder is listed in .gitignore to prevent accidental upload of large files.


**Model Performance**

Random Forest Classification Report:

              precision    recall  f1-score   support

      Benign       1.00      1.00      1.00    242174
   Malicious       1.00      0.99      0.99     57809

    accuracy                           1.00    299983
   macro avg       1.00      1.00      1.00    299983
weighted avg       1.00      1.00      1.00    299983


**Optional Improvements**
Add real-time packet capture

Add anomaly detection models

Improve feature selection or try other classifiers

Deploy using Docker

Build a dashboard for live monitoring


**Author**
Sarah Mund
GitHub: https://github.com/SueMund


**Disclaimer**
This project is for educational purposes only. It should not be used in production environments without further validation, testing, and security hardening.
