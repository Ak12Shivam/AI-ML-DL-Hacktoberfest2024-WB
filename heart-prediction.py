import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset (you'll need to provide your own dataset or use a publicly available one)
data = pd.read_csv('heart_disease_data.csv')

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Function to predict heart disease for new data
def predict_heart_disease(new_data):
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    probability = model.predict_proba(new_data_scaled)[:, 1]
    return prediction[0], probability[0]

# Example usage of the prediction function
new_patient = np.array([[60, 1, 0, 140, 289, 0, 1, 172, 0, 0, 1, 0, 2]])  # Sample data
prediction, probability = predict_heart_disease(new_patient)
print(f"\nPrediction for new patient: {'Positive' if prediction == 1 else 'Negative'}")
print(f"Probability of heart disease: {probability:.2f}")
