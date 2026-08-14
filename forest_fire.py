import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================
# Load Dataset
# ==========================

data = pd.read_csv("forestfires.csv")

print("First 5 Rows:\n")
print(data.head())

# ==========================
# Convert Target Variable
# ==========================

# 0 = No Fire
# 1 = Fire

data["fire"] = data["area"].apply(lambda x: 0 if x == 0 else 1)

# ==========================
# Encode Categorical Columns
# ==========================

label_encoder = LabelEncoder()

data["month"] = label_encoder.fit_transform(data["month"])
data["day"] = label_encoder.fit_transform(data["day"])

# ==========================
# Select Features
# ==========================

X = data.drop(["area", "fire"], axis=1)

y = data["fire"]

# ==========================
# Split Dataset
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train Random Forest Model
# ==========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Prediction
# ==========================

predictions = model.predict(X_test)

# ==========================
# Accuracy
# ==========================

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

# ==========================
# Confusion Matrix
# ==========================

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, predictions))

# ==========================
# Classification Report
# ==========================

print("\nClassification Report:\n")

print(classification_report(y_test, predictions))

# ==========================
# Feature Importance
# ==========================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:\n")

print(importance)

# ==========================
# User Prediction
# ==========================

'''print("\n========== Forest Fire Prediction ==========\n")

X_value = int(input("Enter X coordinate: "))
Y_value = int(input("Enter Y coordinate: "))

month = input("Enter Month (jan-dec): ").lower()
day = input("Enter Day (mon-sun): ").lower()

FFMC = float(input("Enter FFMC: "))
DMC = float(input("Enter DMC: "))
DC = float(input("Enter DC: "))
ISI = float(input("Enter ISI: "))

temp = float(input("Enter Temperature: "))
RH = int(input("Enter Relative Humidity: "))
wind = float(input("Enter Wind Speed: "))
rain = float(input("Enter Rainfall: "))

month = label_encoder.fit_transform(data["month"].astype(str))'''