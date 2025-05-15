# 🏦 Bank Customer Churn Prediction

This project uses machine learning to predict whether a customer will **churn** (leave the bank). It is based on structured customer data and implemented using Python in a Google Colab notebook.

---

## 📁 Dataset

The dataset used is:

**Bank Customer Churn Prediction.csv**

### Features:
- `customer_id`: Unique ID for each customer
- `credit_score`: Credit score of the customer
- `country`: Customer's country
- `gender`: Gender of the customer
- `age`: Age of the customer
- `tenure`: Number of years with the bank
- `balance`: Account balance
- `products_number`: Number of products used
- `credit_card`: Whether the customer has a credit card (0/1)
- `active_member`: Whether the customer is active (0/1)
- `estimated_salary`: Customer’s estimated salary
- `churn`: Target variable (1 = churn, 0 = no churn)

---

## 🧠 Goal

To build a machine learning model that predicts customer churn using supervised classification techniques.

---

## 🔧 Tools & Libraries Used

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn (Label Encoding, Scaling, Model Building)

---

## 📊 Model Used

**Random Forest Classifier**  
- Chosen for its robustness and ability to handle both numerical and categorical features.

---

## 📈 Results

- **Accuracy:** ~86.4%
- **F1-Score (Churn Class):** ~0.57
- **Confusion Matrix:**
