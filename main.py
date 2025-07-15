# 1. Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 2. Load and combine CSVs
csv_files = [
    'data/MachineLearningCSV/MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv',
    'data/MachineLearningCSV/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'data/MachineLearningCSV/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'data/MachineLearningCSV/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'data/MachineLearningCSV/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
]

dataframes = [pd.read_csv(file) for file in csv_files]
df = pd.concat(dataframes, ignore_index=True)
df.columns = df.columns.str.strip()

# 3. Binary label conversion
df['Label'] = df['Label'].apply(lambda x: 'Benign' if x == 'BENIGN' else 'Malicious')

# 4. Handle infinite and missing values
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df = df.dropna()

# 5. Split features and labels
X = df.drop('Label', axis=1)
y = df['Label']

print("\n🔹 Binary Label distribution:")
print(y.value_counts())
print("🔹 Feature matrix shape:", X.shape)
print("🔹 Target vector shape:", y.shape)

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n🔹 Training set size:", X_train.shape)
print("🔹 Test set size:", X_test.shape)

# 7. Sample training set (20k)
X_sample = X_train.sample(n=20000, random_state=1)
y_sample = y_train.loc[X_sample.index]

# 8. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(X_sample, y_sample)
y_pred_rf = rf.predict(X_test)

print("\n🔹 Random Forest Classification Report:")
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
print(classification_report(y_test, y_pred_rf))
print("\n🔹 Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# 9. Save Random Forest model
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/random_forest_ids_20k.pkl')
print("Random Forest model saved successfully.")

# 10. Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_sample, y_sample)
y_pred_knn = knn.predict(X_test)

print("\n🔹 KNN Classification Report:")
knn_report = classification_report(y_test, y_pred_knn, output_dict=True)
print(classification_report(y_test, y_pred_knn))
print("\n🔹 KNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# 11. Create Comparison Graph
# Extract F1 scores
rf_df = pd.DataFrame(rf_report).transpose().loc[['Benign', 'Malicious'], ['precision', 'recall', 'f1-score']]
knn_df = pd.DataFrame(knn_report).transpose().loc[['Benign', 'Malicious'], ['precision', 'recall', 'f1-score']]
rf_df['model'] = 'Random Forest'
knn_df['model'] = 'KNN'
comparison_df = pd.concat([rf_df, knn_df]).reset_index()

# Plot F1-score comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=comparison_df, x='index', y='f1-score', hue='model', palette='Set2')
plt.title('F1-Score Comparison: Random Forest vs KNN')
plt.ylabel('F1 Score')
plt.ylim(0.8, 1.01)
plt.xlabel('Class')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('models/rf_vs_knn_f1.png')
plt.show()