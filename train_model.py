import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("sales_data.csv")

# Create features
df["Gross_Sales"] = df["Quantity"] * df["Unit_Price"]

df["Discount_Amount"] = (
    df["Gross_Sales"] * df["Discount"]
)

df["Net_Sales"] = (
    df["Gross_Sales"] -
    df["Discount_Amount"]
)

# Encode target
df["Returned"] = df["Returned"].map({
    "No": 0,
    "Yes": 1
})

# Select features
features = [
    "Age",
    "Quantity",
    "Unit_Price",
    "Discount",
    "Customer_Rating",
    "Net_Sales"
]

X = df[features]
y = df["Returned"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")

print("\nModel saved successfully!")