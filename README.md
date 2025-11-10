# AI Fault Prediction — Quick Start

A streamlined guide to run, use, and understand the project.

- Frontend: Streamlit dashboard (professional blue/slate UI)
- Backend: FastAPI REST API (/predict, /health)
- Model: XGBoost/RandomForest saved as fault_prediction_model.pkl
- Data: Uses raw, unscaled network metrics; engineered features are computed in the API

## Prerequisites
- Python 3.9+
- Install dependencies: `pip install -r requirements.txt`

## Run the services
1) Backend (FastAPI)
```
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
- Docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/

2) Frontend (Streamlit)
```
streamlit run frontend-enhanced/app_enhanced.py --server.port 8501
```
- Dashboard: http://localhost:8501

## API
- POST /predict — returns prediction, fault probability, and confidence
- GET / — returns basic health info

### Request body (raw, unscaled values)
```json
{
  "RSSI": -75.0,
  "SINR": 18.0,
  "throughput": 95.0,
  "latency": 15.0,
  "jitter": 3.0,
  "packet_loss": 0.5,
  "cpu_usage_percent": 65.0,
  "memory_usage_percent": 60.0,
  "active_users": 350,
  "temperature_celsius": 45.0,
  "hour": 14,
  "day_of_week": 3,
  "is_peak_hour": 1,
  "network_quality_score": 0.75,
  "resource_stress": 65.0
}
```

### Response body (example)
```json
{
  "prediction": "Normal",
  "probability_faulty": 0.185,
  "confidence_percent": 81.5
}
```

## Frontend usage (Streamlit)
- Manual Input tab: enter metrics with real-time status hints
- JSON Input tab: paste the full JSON payload (as above)
- Results: main status card, confidence gauge, fault probability bar, metrics analysis, and prediction history
- Settings: update API base URL if backend runs on a different host/port

## How it works (pipeline)
1. Frontend sends raw metrics to the API
2. API maps fields to training feature names and computes engineered features:
   - efficiency_score = throughput_mbps / (latency_ms + 1)
   - signal_ratio = sinr_db / (abs(rssi_dbm) + 1)
   - network_load_factor = active_users / (cpu_usage_percent + 1)
3. API aligns feature order to the model’s expected features
4. Model predicts and API returns label, probability, and confidence

## Project structure (key files)
- app.py — FastAPI backend
- frontend-enhanced/app_enhanced.py — Streamlit dashboard
- ML_MODEL/fault_prediction.py — training script (saves fault_prediction_model.pkl)
- scripts/generate_synthetic_data.py — synthetic dataset generator
- requirements.txt — dependencies

## Troubleshooting
- API Unreachable badge: ensure backend is running on port 8000
- 422 errors: check JSON shape and field names
- 500 errors: verify model file exists at ML_MODEL/fault_prediction_model.pkl
- Unexpected predictions: validate input ranges (RSSI, SINR, latency, jitter, packet loss) and consider retraining the model with updated data

---

# AI-Powered Fault Prediction in 5G Testbed

## 📋 Project Overview

An AI-powered network management system that predicts faults in 5G testbed environments using machine learning. This project aims to proactively detect and prevent network failures, improving service quality and reducing downtime.

---

## 👥 Team Structure

| Member | Role | Responsibilities |
|--------|------|-----------------|
| **Member 1** | Data Engineer | Dataset creation, preprocessing, and validation |
| **Member 2** | ML Engineer | Model training, optimization, and evaluation |
| **Member 3** | Backend Developer | API development and ML model integration |
| **Member 4** | Frontend Developer | Dashboard creation and visualization |

---

## 📁 Project Structure

```
AI-powered-fault-prediction/
│
├── data/                          # Dataset storage
│   └── synthetic_5g_fault_dataset.csv
│
├── scripts/                       # Data generation & preprocessing scripts
│   ├── generate_synthetic_data.py
│   └── data_preprocessing.py (Day 2)
│
├── notebooks/                     # Jupyter notebooks for analysis
│   └── eda_report.ipynb (Day 3)
│
├── models/                        # Trained ML models
│   └── fault_prediction_model.pkl
│
├── api/                          # Backend API code
│   └── app.py
│
├── dashboard/                    # Frontend dashboard
│   └── streamlit_app.py
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Navigate to the project directory:**
```bash
cd AI-powered-fault-prediction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Generate and preprocess data (Days 1-2 Completed ✅):**
```bash
cd scripts
python generate_synthetic_data.py
python data_preprocessing.py
```

### Processed Data Ready for ML Training
- `data/train.csv` - 8,000 samples for training
- `data/test.csv` - 2,000 samples for testing  
- `data/scaler.pkl` - StandardScaler for deployment
- `data/label_encoder.pkl` - Label encoder for predictions

---

## 📊 Dataset Information

### Processed Dataset Ready for ML

**Training Set:** `data/train.csv`
- **Samples:** 8,000
- **Features:** 17 (scaled and encoded)
- **Class Distribution:** 70.6% Faulty, 29.4% Normal

**Test Set:** `data/test.csv`
- **Samples:** 2,000  
- **Features:** 17 (scaled and encoded)
- **Class Distribution:** 70.7% Faulty, 29.3% Normal

**Original Dataset:** `data/synthetic_5g_fault_dataset.csv` (10,000 samples)

### Features (19 total)

#### Network Performance Metrics
- `rssi_dbm`: Received Signal Strength Indicator (dBm)
- `sinr_db`: Signal-to-Interference-plus-Noise Ratio (dB)
- `throughput_mbps`: Data throughput (Mbps)
- `latency_ms`: Network latency (milliseconds)
- `jitter_ms`: Packet delay variation (milliseconds)
- `packet_loss_percent`: Packet loss percentage

#### Infrastructure Metrics
- `cpu_usage_percent`: CPU utilization
- `memory_usage_percent`: Memory utilization
- `temperature_celsius`: Equipment temperature
- `active_users`: Number of connected users

#### Contextual Features
- `timestamp`: Time of measurement
- `base_station_id`: Base station identifier
- `cell_id`: Cell tower identifier
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0-6)
- `is_peak_hour`: Peak hour indicator (9 AM - 5 PM)

#### Derived Features
- `network_quality_score`: Composite network health metric (0-1)
- `resource_stress`: Average CPU and memory utilization

#### Target Variable
- `fault_status`: Normal or Faulty

---

## 📅 Development Timeline

### ✅ Day 1 - Dataset Creation (Completed)
- [x] Synthetic dataset generation with 10,000 samples
- [x] 19 features including network metrics and fault labels
- [x] Data validation (5/5 checks passed)
- **Deliverables:** `synthetic_5g_fault_dataset.csv`, `generate_synthetic_data.py`

### ✅ Day 2 - Data Preprocessing (Completed)
- [x] Data cleaning and validation
- [x] Feature scaling (StandardScaler) and encoding
- [x] Train-test split (80-20, stratified)
- [x] Saved preprocessing artifacts
- **Deliverables:** `data_preprocessing.py`, `train.csv` (8K), `test.csv` (2K), `scaler.pkl`, `label_encoder.pkl`

### ✅ Day 3 - Exploratory Data Analysis (Completed)
- [x] Feature distribution analysis
- [x] Correlation analysis and heatmap
- [x] Class balance visualization
- [x] Feature importance identification
- [x] Temporal pattern analysis
- **Deliverables:** `eda_report.ipynb` with 15+ visualizations

### ✅ Day 4 - Documentation & Handoff (Completed)
- [x] Final dataset documentation
- [x] Model training guidelines and sample code
- [x] API integration specifications
- [x] Complete ML team handoff documentation
- **Deliverables:** `HANDOFF_TO_ML_TEAM.md` - Complete guide for ML Engineer

---

## 🔧 Usage

### Data Pipeline (Completed ✅)
```bash
# Generate dataset
cd scripts
python generate_synthetic_data.py

# Preprocess data
python data_preprocessing.py
```

### Next Steps for ML Engineer (Member 2)
```python
import pandas as pd
import pickle

# Load preprocessed data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Load scaler and encoder for deployment
with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('data/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Features and target
X_train = train_df.drop('fault_status', axis=1)
y_train = train_df['fault_status']

# Start model training...
```

---

## 📈 Model Development (Member 2)

The preprocessed data will be used to train:
- Random Forest Classifier
- XGBoost
- Support Vector Machine (SVM)
- Neural Networks

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## 🌐 API Development (Member 3)

Backend API will provide:
- `/predict` - Real-time fault prediction
- `/upload` - Bulk data upload
- `/health` - System health check
- `/metrics` - Network metrics dashboard

---

## 📊 Dashboard Features (Member 4)

Interactive dashboard will display:
- Real-time network health status
- Fault probability visualization
- Alert notifications
- Historical trend analysis
- Network KPI monitoring

---

## 🤝 Contributing

Each team member works on their designated area:
1. Create feature branch from main
2. Commit changes with clear messages
3. Test thoroughly before merge
4. Document all changes

---

## 📝 License

This is an academic project for 5G network fault prediction research.

---

## 📞 Contact

**Team Members:**
- Data Engineer: Dataset & Preprocessing
- ML Engineer: Model Development
- Backend Developer: API Integration
- Frontend Developer: Dashboard & UI

---

**Last Updated:** November 4, 2025  
**Status:** Days 1-4 Complete ✅ | Data Engineering Finished | Ready for ML Training 🚀

---

## 🎯 Data Engineering Complete!

All data work is finished! The ML team has everything needed:
- ✅ Clean, preprocessed datasets (train/test)
- ✅ Comprehensive EDA with insights
- ✅ Deployment artifacts (scaler, encoder)
- ✅ Complete handoff documentation

**👉 ML Team: Start with `HANDOFF_TO_ML_TEAM.md`**
