# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # For saving the model
import warnings

# Ignore any warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Step 1: Load and Prepare Data ---
try:
    df = pd.read_csv('Crop_recommendation.csv')
except FileNotFoundError:
    print("Error: 'Crop_recommendation.csv' not found.")
    exit()

# Separate features (X) and the target variable (y)
X = df.drop('label', axis=1)
y = df['label']

# --- Step 2: Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Step 3: Feature Scaling ---
# Scale features to have a mean of 0 and a standard deviation of 1
# This helps the model perform better.
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler object for use in prediction
joblib.dump(scaler, 'scaler.joblib')
print("Scaler has been saved to 'scaler.joblib'\n")

# --- Step 4: Build and Train the Random Forest Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)

print("Training the Random Forest model...")
model.fit(X_train_scaled, y_train)
print("Model training complete.\n")

# --- Step 5: Save the Trained Model ---
# Save the model to a file so we can use it later
joblib.dump(model, 'crop_model.joblib')
print("Model has been saved to 'crop_model.joblib'\n")

# --- Step 6: Evaluate the Model ---
print("Evaluating model performance...")
y_pred = model.predict(X_test_scaled)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Print the classification report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Generate and display the confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
