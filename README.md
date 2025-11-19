
# Credit Card Fraud Detection using Machine Learning

Credit card fraud is a major challenge in the financial industry due to the increasing number of online transactions and sophisticated fraud techniques. This project builds a complete **fraud detection pipeline** using classical machine learning models, proper handling of **extreme class imbalance**, and strong evaluation metrics.  
The objective is to identify fraudulent transactions with high accuracy and minimal false negatives, which is critical for real-world financial institutions.

---

# 📌 Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Challenges](#challenges)  
4. [Technologies Used](#technologies-used)  
5. [Project Architecture](#project-architecture)  
6. [Workflow Summary](#workflow-summary)  
7. [Data Preprocessing](#data-preprocessing)  
8. [Modeling Approach](#modeling-approach)  
9. [Evaluation Metrics](#evaluation-metrics)  
10. [Results](#results)  
11. [Visualizations](#visualizations)  
12. [Key Insights](#key-insights)  
13. [How to Run the Project](#how-to-run-the-project)  
14. [Project Structure](#project-structure)  
15. [Future Work](#future-work)  
16. [Author](#author)

---

# 🚀 Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning.  
Fraud detection is a **binary classification problem**, but the dataset used is **highly imbalanced**, with fraudulent transactions making up less than **0.2%** of all samples.

To address this, the project implements:
- Oversampling using **SMOTE**
- Proper splitting to avoid data leakage
- Training multiple ML models
- Comparing performance across evaluation metrics
- Detailed visualizations for clarity

This project demonstrates your ability to work with imbalanced data, apply ML models, tune them, and evaluate them professionally — highly relevant for roles in **Machine Learning, Data Science, and FinTech**.

---

# 📂 Dataset

The dataset used in this project is publicly available on Kaggle:

👉 **Credit Card Fraud Dataset**  
https://www.kaggle.com/datasets/waqasishtiaq/credit-card-fraud-dataset  

### **Dataset Details**
- **Rows**: 284,807 transactions  
- **Fraud cases**: ~492 (0.17%)  
- **Features**:  
  - `V1` to `V28`: PCA-transformed numerical features  
  - `Amount`: Transaction amount  
  - `Time`: Time of transaction  
  - `Class`: Target label  
    - `0` → Legitimate  
    - `1` → Fraud  

### **Why PCA Features?**
To protect customer identities and sensitive attributes, PCA-transformed features are used instead of original cardholder data.

---

# ⚠️ Challenges

The project tackles several real-world machine learning challenges:

### **✔ Extreme Class Imbalance**
Fraud cases are less than 1% — typical models will predict everything as "legit" and still get 99% accuracy.  
This requires special handling like **SMOTE** and proper evaluation metrics.

### **✔ Data Leakage Risks**
Applying oversampling **before** train-test split leaks information.  
This project avoids that using correct methodology.

### **✔ High Cost of False Negatives**
Misclassifying a fraud as legitimate results in financial loss.

Thus, recall, F1, and ROC-AUC are more important than accuracy.

---

# 🛠 Technologies Used

### **Languages**
- Python

### **Libraries**
- NumPy, Pandas  
- Scikit-learn  
- XGBoost  
- Imbalanced-Learn (SMOTE)  
- Matplotlib, Seaborn  
- Jupyter Notebook  

---

# 🧱 Project Architecture

```

credit-card-fraud-detection/
│
├── README.md                  # Documentation
├── requirements.txt           # Libraries to install
│
├── notebooks/
│   └── fraud_detection_clean.ipynb   # Final cleaned notebook (resume-ready)
│
└── src/
├── data_preprocessing.py
├── model_training.py
└── evaluation_utils.py

````

---

# 🔁 Workflow Summary

### ✔ Data Loading & Cleaning  
### ✔ Feature scaling  
### ✔ Train-test split (stratified)  
### ✔ SMOTE applied only on training data  
### ✔ Training 4 models  
### ✔ Performance comparison  
### ✔ Visualization of key metrics  

---

# 🧹 Data Preprocessing

Steps performed:

1. **Removed duplicates**  
2. **Handled missing values**  
3. **Separated features and target**  
4. Applied **StandardScaler** to numerical features  
5. **Stratified train-test split** to preserve fraud ratio  
6. Applied **SMOTE ONLY on training data**  
7. Verified class balance  
8. Prepared final input for model training  

---

# 🤖 Modeling Approach

The following models were trained:

| Model | Type |
|-------|------|
| Logistic Regression | Linear classifier |
| Decision Tree | Rule-based classifier |
| Random Forest | Ensemble (Bagging) |
| XGBoost | Ensemble (Boosting) |

### Why these models?
✔ They handle tabular data well  
✔ Work efficiently on medium-size datasets  
✔ Provide interpretability & feature importance  
✔ Commonly used in fraud detection systems  

---

# 📏 Evaluation Metrics

Since the dataset is imbalanced, the following metrics were used:

### **Primary Metrics**
- **Recall** (most important — catching fraud cases)
- **F1 Score**
- **ROC-AUC Score**

### **Secondary Metrics**
- Precision
- Accuracy  
- Confusion Matrix  

Why not rely on accuracy?  
Because a model predicting all transactions as “legit” will score 99.8% accuracy — but detect **zero frauds**.

---

# 📊 Results

### **Summary Table**

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|----------|
| Logistic Regression | Good | Medium | Medium | Medium | High |
| Decision Tree | Medium | Medium | Good | Good | Medium |
| Random Forest | **High** | **High** | **High** | **High** | **Very High** |
| XGBoost | **Excellent** | **High** | **High** | **High** | **Best (≈0.99)** |

### **Best Model: XGBoost**
- Very high ROC-AUC  
- Excellent recall  
- Robust generalization  
- Suitable for real-time fraud detection pipelines  

---

# 📉 Visualizations

The project includes:  
✔ Confusion Matrix  
✔ ROC Curve  
✔ Performance Comparison Bar Chart  

These help explain model performance visually, which is important in real-world ML workflows.

---

# 💡 Key Insights

1. **SMOTE drastically improves recall**, which is crucial for fraud detection.  
2. **Tree-based models outperform linear models** on this dataset.  
3. Random Forest and XGBoost provide **superior stability and accuracy**.  
4. Feature scaling + PCA components help reduce noise.  
5. Fraud detection requires:  
   - High recall  
   - High F1 score  
   - Strong separation between classes  

---

# ▶️ How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
````

### 2. Open the notebook

```bash
jupyter notebook notebooks/fraud_detection_clean.ipynb
```

### 3. Run all cells

# 🚀 Future Work

* Add deep learning models (ANN, LSTM)
* Deploy model using FastAPI or Flask
* Build a live dashboard using Streamlit
* Hyperparameter tuning with Optuna
* Real-time fraud probability scoring
* Model drift detection & monitoring

---

# 👤 Author

**Your Name**
Machine Learning Enthusiast | Data Analyst
GitHub: *your-username*
LinkedIn: *your-profile-link*

---

# ⭐ Final Note

This project is designed to demonstrate **real-world ML skills**, including:

* Working with highly imbalanced data
* Building complete ML pipelines
* Evaluating models correctly
* Preparing production-ready documentation

