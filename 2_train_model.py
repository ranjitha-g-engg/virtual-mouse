import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

DATA_FILE = r'E:\htracker\gestures.csv'
MODEL_FILE = 'gesture_model.pkl'

# --- 1. Load the Dataset ---
print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# Check for empty or invalid data
if df.empty or 'label' not in df.columns:
    print("Error: The CSV file is empty or is missing the 'label' column.")
    print("Please run 1_data_collector.py to collect data.")
    exit()

print("Data loaded successfully.")
print(f"Dataset has {df.shape[0]} samples.")
print("\nGesture counts in your dataset:")
print(df['label'].value_counts())
print("-" * 30)


# --- 2. Prepare the Data ---
# X contains the features (all the coordinate columns)
# y contains the target label (the 'label' column)
X = df.drop('label', axis=1) 
y = df['label']


# --- 3. Split Data into Training and Testing Sets ---
# stratify=y is important here! It ensures that the training and testing sets have
# the same proportion of samples for each gesture, which is crucial for imbalanced datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- 4. Train the Model ---
print("\nTraining the RandomForestClassifier model...")
# n_estimators=100 means the model is built from 100 "decision trees"
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")
print("-" * 30)


# --- 5. Evaluate the Model ---
print("\nEvaluating the model on the test data...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
# This detailed report shows the performance for each individual gesture
print(classification_report(y_test, y_pred))
print("-" * 30)


# --- 6. Save the Trained Model ---
print(f"\nSaving the trained model to {MODEL_FILE}...")
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)
print("Model saved successfully. You are ready for the final step!")