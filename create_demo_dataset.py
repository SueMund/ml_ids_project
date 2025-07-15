import pandas as pd

source_file = "data/MachineLearningCSV/MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"

# Load it
df = pd.read_csv(source_file)

# Strip extra whitespace from column names
df.columns = df.columns.str.strip()

# Drop the label column
df = df.drop(columns=['Label'])

# Sample 5 flows
sample = df.sample(5, random_state=42)

# Save to CSV
sample.to_csv("realworld_input.csv", index=False)

print("completed")
