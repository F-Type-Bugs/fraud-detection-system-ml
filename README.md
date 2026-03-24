# 💳 Fraud Detection System (Machine Learning)

A Machine Learning web application that detects whether a transaction is **fraudulent or legitimate** using classification models, with a focus on handling **highly imbalanced data**.

---

## 🚀 Project Overview

This project aims to solve a real-world financial problem — detecting fraudulent transactions from a highly imbalanced dataset.

The system processes transaction data and predicts:

- ⚠️ Fraudulent Transaction  
- ✅ Legitimate Transaction  

This project demonstrates an **end-to-end ML pipeline**, from data preprocessing to deployment.

---

## 🧠 Key Features

- ✔ Handles extreme class imbalance using **SMOTE**
- ✔ Uses **Logistic Regression** for classification
- ✔ Applies **ROC Curve & AUC** for performance evaluation
- ✔ Implements **threshold tuning** for real-world decision making
- ✔ Analyzes **feature importance** for model explainability
- ✔ Deployed with **Streamlit** for interactive prediction

---

## 📊 Model Performance & Insights

- Achieved **AUC Score ≈ 0.97**, indicating strong model performance  
- Tuned classification threshold to improve fraud detection recall  
- Achieved high recall (~90%) for fraud detection with trade-off in precision  
- Demonstrated real-world trade-offs between **precision and recall**

---

## ⚠️ Important Note

This is a **demo application**.

The dataset uses **PCA-transformed features (V1–V28)**, which means:

- Features are not human-readable  
- Inputs are not user-friendly  

👉 The purpose of this project is to demonstrate:
- ML system design  
- Model evaluation strategies  
- Deployment skills  

---

## 🔄 Machine Learning Workflow

1. Data Loading & Exploration  
2. Train-Test Split (Stratified)  
3. Handling Imbalanced Data using SMOTE  
4. Model Training (Logistic Regression)  
5. Model Evaluation:
   - Confusion Matrix  
   - Precision, Recall, F1-score  
   - ROC Curve & AUC  
6. Threshold Tuning (0.5 → 0.3)  
7. Feature Importance Analysis  
8. Deployment using Streamlit  

---

## 🛠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Matplotlib, Seaborn  
- Streamlit  
- Joblib  

---

## 📁 Project Structure


fraud-detection-system-ml/
│
├── train_model.py
├── app.py
├── fraud_model.pkl
├── fraud_feature_columns.pkl
├── requirements.txt
└── README.md


---

## 📊 Dataset

Dataset used: Credit Card Fraud Detection  

🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

---

## ⚙️ Installation & Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/fraud-detection-system-ml.git
cd fraud-detection-system-ml

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py


##🎯 Key Learning Outcomes
Handling highly imbalanced datasets
Understanding precision vs recall trade-offs
Applying ROC-AUC for model evaluation
Implementing threshold tuning for business decisions
Building and deploying ML applications


##🚀 Future Improvements
Improve model using advanced algorithms (XGBoost, LightGBM)
Build user-friendly input system (non-PCA features)
Deploy application on cloud (AWS / Streamlit Cloud)
Integrate real-time transaction monitoring

##👨‍💻 Author
Farhan Tanvir
