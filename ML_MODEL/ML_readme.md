AI-Powered Fault Prediction in 5G Testbed

Project Overview :

The AI-Powered Fault Prediction in 5G Testbed is a collaborative project designed to monitor and predict network faults in a 5G environment. The system involves dataset creation, preprocessing, model evaluation, backend integration, and frontend visualization.

The project aims to provide real-time insights into network performance and identify potential faults efficiently.

Member 2 – Responsible for assisting in model training, evaluation, and optimization to support the system.

Project Steps
1. Load Dataset : 

  Read CSV into a pandas DataFrame.
  Inspect data types and fault_status distribution.

2. Add Realism : 

  Simulate sensor noise using normal distribution to replicate real-world variations.

3. Add Derived Features : 

  Compute additional features capturing hidden dependencies:
  efficiency_score
  signal_ratio
  network_load_factor

4. Add Label Uncertainty

  Flip ~7% of labels randomly to simulate mislabeling in real datasets.

5. Encode Target Variable

  Convert fault_status to numeric using LabelEncoder (Normal = 0, Faulty = 1).

6. Train/Test Split

  Drop non-numeric columns. 
  Split data into train and test sets (80/20) with stratification to maintain class balance.

7. Train Random Forest

  200 estimators, random_state=42.
  Evaluate using:
  classification_report
  accuracy_score

8. Train XGBoost

  300 estimators, learning rate 0.05, max depth 6.
  Evaluate metrics and compare with Random Forest.

9. Save Best Model

  Saved the model with highest accuracy using pickle.

10. Real-Time Fault Prediction
  
  Load the saved model using joblib.
  Predict new samples with all required features.
  Output: "Normal" or "Faulty".

11. Model Performance Graph

  Compare Random Forest and XGBoost accuracies in a bar chart.
  Saved as: model_performance.png.

12. Feature Importance

  Plot top 10 features for both models:
  Random Forest → rf_feature_importance.png
  XGBoost → xgb_feature_importance.png
  Helps understand which metrics impact fault prediction most.

Key Results: 
  Random Forest Accuracy: 0.931
  XGBoost Accuracy: 0.931

Notes: 
  Plots are saved with DPI=100 for fast execution in VS Code.
  Avoid plt.show() to prevent execution blocking.
  Example of real-time prediction:
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
prediction = model.predict(pd.DataFrame([sample_input]))[0]
status = "Normal" if prediction == 0 else "Faulty"

Output: Predicted Fault Status: Faulty


Folder Structure: 
ML_FAULT_PREDICTION/
│
├─ synthetic_5g_fault_dataset.csv
├─ fault_prediction_model.pkl
├─ model_performance.png
├─ rf_feature_importance.png
├─ xgb_feature_importance.png
├─ rf_performance_report.txt
├─ xgb_performance_report.txt
└─ fault_prediction_system.md


