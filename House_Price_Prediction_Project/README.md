# 🏡 House Price Prediction Project

An end-to-end Machine Learning system for real estate price valuation in Indian Rupees (₹), complete with dataset generation, model comparison, FastAPI REST API backend, and an interactive Streamlit frontend.

---

## 📂 Project Directory Structure

```text
House_Price_Prediction_Project/
├── data/
│   ├── generate_data.py       # Dataset generator script
│   └── housing_data.csv       # Synthetic housing dataset (1,500 rows)
├── models/
│   ├── train_model.py         # Model training & evaluation pipeline
│   └── house_price_model.pkl  # Trained Gradient Boosting model bundle
├── backend/
│   └── main_api.py            # FastAPI REST API server
├── frontend/
│   └── app.py                 # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Models
```bash
python data/generate_data.py
python models/train_model.py
```

### 3. Launch Backend REST API (FastAPI)
```bash
python backend/main_api.py
```
- Interactive Swagger API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Launch Frontend Web App (Streamlit)
```bash
streamlit run frontend/app.py
```
- Interactive Web App: [http://localhost:8501](http://localhost:8501)
