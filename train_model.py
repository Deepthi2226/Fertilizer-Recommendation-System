import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\DELL\Desktop\internship\Fertilizer_Prediction.csv")

# Encode categorical columns
label_encoders = {}
for col in ["Soil Type", "Crop Type", "Fertilizer Name"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42, criterion="entropy")
dt_model.fit(X_train, y_train)

# Save model & encoders
joblib.dump(dt_model, "fertilizer_dt_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Model training completed & saved successfully!")

accuracy = dt_model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")