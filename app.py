from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

# Load trained model & encoders
dt_model = joblib.load("fertilizer_dt_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Load dataset just to fetch available Soil & Crop types
df = pd.read_csv(r"C:\Users\DELL\Desktop\internship\Fertilizer_Prediction.csv")
soil_types = sorted(df["Soil Type"].unique())
crop_types = sorted(df["Crop Type"].unique())

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", soil_types=soil_types, crop_types=crop_types)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        temperature = float(request.form["Temperature"])
        humidity = float(request.form["Humidity"])
        moisture = float(request.form["Moisture"])
        soil_type = request.form["Soil Type"]
        crop_type = request.form["Crop Type"]
        nitrogen = float(request.form["Nitrogen"])
        potassium = float(request.form["Potassium"])
        phosphorous = float(request.form["Phosphorous"])

        # Encode categorical values
        soil_encoded = label_encoders["Soil Type"].transform([soil_type])[0]
        crop_encoded = label_encoders["Crop Type"].transform([crop_type])[0]

        # Prepare input for model
        input_data = np.array([[temperature, humidity, moisture,
                                soil_encoded, crop_encoded,
                                nitrogen, potassium, phosphorous]])

        # Predict fertilizer
        prediction = dt_model.predict(input_data)[0]
        fertilizer_name = label_encoders["Fertilizer Name"].inverse_transform([prediction])[0]

        # Show only result page
        return render_template("result.html", fertilizer=fertilizer_name)

    except Exception as e:
        return render_template("result.html", fertilizer=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
