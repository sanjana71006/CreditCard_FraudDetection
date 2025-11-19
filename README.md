# Credit Card Fraud Detection – Machine Learning Project

This project builds a machine learning-based system to detect fraudulent credit card transactions using classical ML models while handling extreme class imbalance.

---

## 📂 Dataset
- **Source:** [Kaggle – Credit Card Fraud Dataset](https://www.kaggle.com/datasets/waqasishtiaq/credit-card-fraud-dataset)  
- **Rows:** 284,807  
- **Features:** `Time`, `V1..V28`, `Amount`, `Class`  
- **Target labels:**  
  - `0` → Legit  
  - `1` → Fraud  
- **Challenge:** Highly imbalanced dataset (fraud ≈ 0.17%)

---

## 🛠 Tools & Technologies
- Python, NumPy, Pandas  
- Scikit-Learn  
- SMOTE (Imbalanced-Learn)  
- XGBoost  
- Matplotlib, Seaborn  
- Jupyter Notebook  

---

## 🚀 Project Workflow

### **1️⃣ Data Preprocessing**
- Removed duplicates and missing entries  
- Standardized numerical features using `StandardScaler`

### **2️⃣ Train-Test Split**
- Stratified split to preserve class ratio  
- **IMPORTANT:** SMOTE applied **only on training data** (avoids data leakage)

### **3️⃣ Oversampling**
- SMOTE used to balance training data  
- Fraud class increased from 0.17% → 50%

### **4️⃣ Models Used**
- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  

### **5️⃣ Evaluation Metrics**
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- Confusion Matrix  
- ROC Curve  

---

## 📊 Key Results (Test Set)
Best-performing models:
- **XGBoost**
  - ROC-AUC ≈ **0.99**
  - High recall (fraud detection)
- **Random Forest**
  - Stable performance
  - Good balance of precision & recall

---

## 🖼 Output Visualizations
- Confusion Matrix  
- ROC Curve  
- Comparison Bar Chart (Accuracy, Precision, Recall, F1)

---

## 📌 File Structure

