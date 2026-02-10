import joblib
import numpy as np

# --- Load the Saved Model and Scaler ---
try:
    model = joblib.load('crop_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler file not found. Please run the training script first.")
    exit()

# --- Make a Prediction on New Data ---

# Get new data from the user or another source
# Format: [N, P, K, temperature, humidity, ph, rainfall]
print("\nEnter the environmental and soil conditions to get a crop recommendation.")

try:
    n = float(input("Enter Nitrogen (N) content: "))
    p = float(input("Enter Phosphorus (P) content: "))
    k = float(input("Enter Potassium (K) content: "))
    temp = float(input("Enter temperature (°C): "))
    humidity = float(input("Enter humidity (%): "))
    ph = float(input("Enter pH value: "))
    rainfall = float(input("Enter rainfall (mm): "))

    # Create a numpy array from the inputs
    new_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])

    # Scale the new data using the loaded scaler
    new_data_scaled = scaler.transform(new_data)

    # Use the loaded model to predict the crop
    prediction = model.predict(new_data_scaled)

    # Display the result
    print("\n" + "="*30)
    print(f"🌾 Recommended Crop: {prediction[0].capitalize()}")
    print("="*30)

except ValueError:
    print("\nInvalid input. Please enter numerical values only.")
