# 💳 Fraud Detection System (ML)

A Machine Learning web application that predicts whether a transaction is **fraudulent or legitimate** using classification models.

---

## 🚀 Project Overview

This project focuses on detecting fraudulent financial transactions using a highly imbalanced dataset.

The system analyzes transaction features and predicts:

- ⚠️ Fraudulent Transaction  
- ✅ Legitimate Transaction  

---

## 🧠 Key Features

- Handles **extreme class imbalance** using SMOTE  
- Uses classification models for fraud detection  
- Evaluates performance using:
  - Precision
  - Recall
  - F1-score
- Deployed using **Streamlit** for real-time predictions  

---

## ⚠️ Note

This is a **demo application**.  
The dataset uses PCA-transformed features (`V1–V28`), so inputs are not user-friendly.  

The goal of this project is to demonstrate:

- End-to-end ML pipeline  
- Model training & evaluation  
- Deployment skills  

---

## 📊 Machine Learning Workflow

1. Data Loading & Exploration  
2. Handling Imbalanced Data (SMOTE)  
3. Model Training (Logistic Regression)  
4. Model Evaluation  
5. Deployment using Streamlit  

---

## 🛠 Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Streamlit  
- Joblib  

---

## Dataset link
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download?datasetVersionNumber=3

## ⚙️ Installation & Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fraud-detection-system-ml.git
cd fraud-detection-system-ml