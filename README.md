# Customer Churn Prediction (Machine Learning Project)

A production-ready **Machine Learning application** that predicts whether a telecom customer is likely to churn using an **XGBoost classifier**. The project demonstrates an end-to-end ML pipeline including data preprocessing, model training, evaluation, and deployment via a web application.

---

## 📌 Project Overview

Customer churn is a critical problem for telecom businesses. This project uses machine learning to:

* Identify customers at risk of leaving
* Provide actionable insights for retention strategies
* Deliver real-time predictions through a web interface

---

## 📁 Project Structure

```
customer-churn-prediction/
├── app.py                          # Streamlit web application
├── src/
│   └── churn_prediction.py        # Script-based prediction module
├── notebooks/
│   └── churn_analysis.ipynb       # Model training and experimentation
├── models/
│   ├── churn_model.pkl            # Trained XGBoost model
│   ├── model_columns.pkl          # Feature column names
│   └── scaler.pkl                 # StandardScaler for preprocessing
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── dashboard/
    └── churn_dashboard.twbx       # Tableau dashboard
```

---

## 🚀 Features

The model predicts churn probability based on:

* **Tenure** → Customer lifetime (months)
* **Monthly Charges** → Monthly billing amount
* **Total Charges** → Total spending

---

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing

* Missing value handling
* Categorical encoding using one-hot encoding
* Feature scaling using **StandardScaler**

### 2. Model Training

* Logistic Regression (baseline)
* Random Forest (comparison)
* **XGBoost (final model)**

### 3. Model Evaluation

* Accuracy Score
* Confusion Matrix
* Classification Report
* ROC-AUC Score (~0.85+)

---

## 🏆 Final Model

* **Algorithm:** XGBoost Classifier
* **Performance:** ~80–85% Accuracy
* **AUC Score:** ~0.86

---

## 📊 Business Insights

* Customers with **month-to-month contracts** churn the most
* **Higher monthly charges** increase churn probability
* **New customers (low tenure)** are more likely to churn
* **Fiber optic users** show higher churn behavior

---

## ▶️ How to Run the Project

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn
```

---

### 2. Train the Model (Optional)

```bash
cd notebooks
jupyter notebook churn_analysis.ipynb
```

Run all cells to generate:

* `churn_model.pkl`
* `model_columns.pkl`
* `scaler.pkl`

---

### 3. Run Streamlit App

```bash
cd ..
python -m streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

### 4. Run Script Version

```bash
python src/churn_prediction.py
```

---

### 5. View Dashboard

Open in Tableau:

```
dashboard/churn_dashboard.twbx
```

---

## 💻 Web Application

The Streamlit app allows users to:

* Enter customer details
* Predict churn in real-time
* View probability of churn

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Tableau

---

## 🔧 Key Improvements

* Consistent preprocessing using saved scaler
* Feature alignment with training data
* Probability-based predictions
* End-to-end ML pipeline integration

---

## 🎯 Business Impact

This solution helps businesses:

* Reduce customer churn
* Improve retention strategies
* Increase customer lifetime value

---

## 🔮 Future Enhancements

* Add more features (contract type, payment method)
* Deploy application on cloud (Streamlit Cloud / AWS)
* Hyperparameter tuning
* Real-time data integration

---

## 📬 Contact

Open for collaboration and discussions on ML/Data Science projects.

---
