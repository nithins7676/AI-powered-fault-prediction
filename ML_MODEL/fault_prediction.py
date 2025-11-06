import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import joblib
import matplotlib.pyplot as plt
import os

# ===============================
# Step 1: Loading Dataset
# ===============================
df = pd.read_csv("synthetic_5g_fault_dataset.csv")

print("Initial Data Info:")
print(df.info())
print("\nFault Status Distribution:\n", df['fault_status'].value_counts())

# =======================================================
# Step 2: Adding Realism - Simulating Sensor Noise
# =======================================================
noise_scale = {
    'rssi_dbm': 2,                # ±2 dBm variation
    'sinr_db': 1.5,               # ±1.5 dB variation
    'throughput_mbps': 5,         # ±5 Mbps variation
    'latency_ms': 5,              # ±5 ms variation
    'jitter_ms': 2,               # ±2 ms variation
    'packet_loss_percent': 0.5,   # ±0.5% variation
}

for col, scale in noise_scale.items():
    if col in df.columns:
        df[col] = df[col] + np.random.normal(0, scale, len(df))

# =======================================================
# Step 3: Added Derived Features (Hidden Dependencies)
# =======================================================
df['efficiency_score'] = df['throughput_mbps'] / (df['latency_ms'] + 1)
df['signal_ratio'] = df['sinr_db'] / (abs(df['rssi_dbm']) + 1)
df['network_load_factor'] = df['active_users'] / (df['cpu_usage_percent'] + 1)

# =======================================================
# Step 4: Added Label Uncertainty (Simulate Real Data)
# =======================================================
flip_prob = 0.07  # 7% chance of wrong label
mask = np.random.rand(len(df)) < flip_prob
df.loc[mask, 'fault_status'] = df['fault_status'].map({'Normal': 'Faulty', 'Faulty': 'Normal'})

# =======================================================
# Step 5: Encoded Target Variable
# =======================================================
label_encoder = LabelEncoder()
df['fault_status'] = label_encoder.fit_transform(df['fault_status'])  # Normal=0, Faulty=1

# =======================================================
# Step 6: Cleaning & Spliting Train/Test Data
# =======================================================
# Drop obvious non-numeric or identifier columns if present
drop_cols = [col for col in df.columns if df[col].dtype == 'object' or 'time' in col.lower() or 'date' in col.lower() or 'id' in col.lower()]

if drop_cols:
    print(f"\n🧹 Dropping non-numeric columns: {drop_cols}")
    df = df.drop(columns=drop_cols)

# Separate features and target
X = df.drop(columns=['fault_status'])
y = df['fault_status']

# Ensure only numeric values remain
X = X.select_dtypes(include=[np.number])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n Train shape: {X_train.shape}, Test shape: {X_test.shape}")


# =======================================================
# Step 7: Training using Random Forest
# =======================================================
print("\n Training Random Forest ")
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_rf))
print("Validation Accuracy:", accuracy_score(y_test, y_pred_rf))

# =======================================================
# step 8: Train XGBoost
# =======================================================
print("\n Training XGBoost ")
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("\nXGBoost Results:")
print(classification_report(y_test, y_pred_xgb))
print("Validation Accuracy:", accuracy_score(y_test, y_pred_xgb))

# =======================================================
# Step 9: Saving Best Model
# =======================================================
best_model = xgb_model if accuracy_score(y_test, y_pred_xgb) > accuracy_score(y_test, y_pred_rf) else rf_model
with open("fault_prediction_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n Best model saved as: fault_prediction_model.pkl")

# ==========================
# Step 10: Real-Time Fault Prediction
# ==========================

print("\n Real-Time Fault Prediction : ")

# Load saved model
model = joblib.load("fault_prediction_model.pkl")

# Get model training feature names
if hasattr(model, "feature_names_in_"):
    expected_features = model.feature_names_in_
else:
    expected_features = None

# Example input (you can modify this)
sample_input = {
    "rssi_dbm": -88,
    "sinr_db": 8,
    "throughput_mbps": 45,
    "latency_ms": 60,
    "jitter_ms": 20,
    "packet_loss_percent": 4,
    "cpu_usage_percent": 70,
    "memory_usage_percent": 65,
    "active_users": 35,
    "temperature_celsius": 45,
    "hour": 15,
    "day_of_week": 3,
    "is_peak_hour": 1,
    "network_quality_score": 0.6,
    "resource_stress": 0.7,
    "efficiency_score": 0.5,
    "network_load_factor": 0.65,
    "signal_ratio": 0.8
}

input_df = pd.DataFrame([sample_input])

# Align columns
if expected_features is not None:
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0  # add missing feature
    input_df = input_df[expected_features]  # reorder columns

# Predict
prediction = model.predict(input_df)[0]
status = "Normal" if prediction == 0 else "Faulty"

print(f"Predicted Fault Status: {status}")


# ==========================
# Step 11: Model Performance Graph 
# ==========================

print("\n Plotting Model Performance ")

# Storing accuracy values directly here
# (Using the same printed accuracies from your previous step)
rf_acc = accuracy_score(y_test, y_pred_rf)
xgb_acc = accuracy_score(y_test, y_pred_xgb)

# Preparing the data for plotting
models = ["Random Forest", "XGBoost"]
accuracies = [rf_acc, xgb_acc]

# Creating and formating the bar chart
plt.figure(figsize=(6, 4))
bars = plt.bar(models, accuracies, color=["skyblue", "lightgreen"], width=0.5, edgecolor='black')

# Annotating the bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.005, f"{acc:.3f}", ha='center', fontsize=10, fontweight='bold')

plt.ylim(0, 1)
plt.ylabel("Validation Accuracy", fontsize=12)
plt.title("Model Comparison - Fault Prediction System", fontsize=13, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# Save and show 
save_path = os.path.join(os.getcwd(), "model_performance.png")
plt.savefig(save_path, dpi=300)
plt.show()

print(f"Graph saved successfully at: {save_path}")

# ==========================
# Step 12: Feature Importance Graphs
# ==========================
print("\n Feature Importance Graphs ")

# Random Forest
rf_importances = rf_model.feature_importances_
rf_features = X_train.columns

plt.figure(figsize=(8, 5))
indices = np.argsort(rf_importances)[::-1][:10]  # Top 10 features
plt.barh(np.array(rf_features)[indices][::-1],
         np.array(rf_importances)[indices][::-1],
         color='skyblue', edgecolor='black')
plt.xlabel("Importance Score", fontsize=12)
plt.title("Top 10 Feature Importances - Random Forest", fontsize=13, fontweight="bold")
plt.tight_layout()

rf_path = os.path.join(os.getcwd(), "rf_feature_importance.png")
plt.savefig(rf_path, dpi=300)
plt.show()
print(f" Random Forest feature importance graph saved: {rf_path}")

# XGBoost
xgb_importances = xgb_model.feature_importances_
xgb_features = X_train.columns

plt.figure(figsize=(8, 5))
indices = np.argsort(xgb_importances)[::-1][:10]
plt.barh(np.array(xgb_features)[indices][::-1],
         np.array(xgb_importances)[indices][::-1],
         color='lightcoral', edgecolor='black')
plt.xlabel("Importance Score", fontsize=12)
plt.title("Top 10 Feature Importances - XGBoost", fontsize=13, fontweight="bold")
plt.tight_layout()

xgb_path = os.path.join(os.getcwd(), "xgb_feature_importance.png")
plt.savefig(xgb_path, dpi=300)
plt.show()
print(f" XGBoost feature importance graph saved: {xgb_path}")


# ==========================
# Step13: Saved Model Performance Reports
# ==========================
rf_report = classification_report(y_test, y_pred_rf)
xgb_report = classification_report(y_test, y_pred_xgb)

with open("rf_performance_report.txt", "w") as f:
    f.write("Random Forest Performance Report\n")
    f.write("================================\n")
    f.write(rf_report)
    f.write(f"\nValidation Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}\n")

with open("xgb_performance_report.txt", "w") as f:
    f.write("XGBoost Performance Report\n")
    f.write("===========================\n")
    f.write(xgb_report)
    f.write(f"\nValidation Accuracy: {accuracy_score(y_test, y_pred_xgb):.3f}\n")

print("\n Performance reports saved: rf_performance_report.txt, xgb_performance_report.txt")
