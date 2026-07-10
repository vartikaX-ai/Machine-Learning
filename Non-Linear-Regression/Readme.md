# Non-Linear Regression

## About this Repository

This repository contains my implementations and practical projects on **Non-Linear Regression** algorithms using **Python** and **Scikit-learn**.

The objective of this repository is to understand how different non-linear regression algorithms model complex relationships between features and target variables through practical implementation, experimentation, and model evaluation.

---

## Algorithms Covered

- Polynomial Regression
- Decision Tree Regression

---

## Topics Covered

- Polynomial Regression
- Decision Tree Regression
- Polynomial Features
- Feature Engineering
- One-Hot Encoding
- Feature Scaling
- Correlation Analysis
- Outlier Detection
- Train-Test Split
- Decision Tree Visualization
- Feature Importance
- Model Training
- Prediction
- Model Evaluation
- Model Comparison

---

## Evaluation Metrics

The models are evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## Projects

### 1. Car Price Prediction using Polynomial Regression

**Objective**

- Predict car prices using Polynomial Regression.
- Compare the performance of Polynomial Regression with Linear Regression.

**Dataset**

- CarPrice_Assignment.csv

**Concepts Practiced**

- One-Hot Encoding
- Feature Scaling
- Polynomial Feature Engineering
- Correlation Analysis
- Outlier Detection
- Model Training
- Model Evaluation
- Model Comparison

---

### 2. Medical Insurance Cost Prediction using Decision Tree Regression

**Objective**

- Predict medical insurance charges using Decision Tree Regression.

**Dataset**

- medical_insurance.csv

**Concepts Practiced**

- One-Hot Encoding
- Correlation Analysis
- Decision Tree Regression
- Feature Importance
- Decision Tree Visualization
- Model Evaluation

---

## Key Learnings

### Polynomial Regression

During experimentation, Polynomial Regression did not outperform Linear Regression on the Car Price dataset.

Generating polynomial and interaction features significantly increased the number of input features, resulting in overfitting and poorer generalization.

**Key Takeaway**

> A more complex model does not always produce better results. Choosing the appropriate model depends on the characteristics of the dataset.

---

### Decision Tree Regression

Decision Tree Regression successfully captured non-linear relationships without requiring feature scaling.

This project also demonstrated how feature importance can be used to identify the most influential variables affecting the prediction.

---

## Repository Structure

```text
Non-Linear-Regression/
│
├── datasets/
│   ├── CarPrice_Assignment.csv
│   └── medical_insurance.csv
│
├── projects/
│   ├── Car_Price_Prediction.py
│   └── Medical_Insurance_Cost_Prediction_Decision_Tree.py
│
├── README.md
└── requirements.txt
```

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/Machine-Learning.git
```

Move into the project directory

```bash
cd Machine-Learning/Non-Linear-Regression
```

Install the required libraries

```bash
pip install -r requirements.txt
```

---

## Learning Outcomes

Through these implementations, I learned:

- When to use Non-Linear Regression algorithms
- Creating Polynomial Features
- Decision Tree Regression
- Feature Engineering
- One-Hot Encoding
- Data Preprocessing
- Correlation Analysis
- Detecting Outliers
- Decision Tree Visualization
- Feature Importance
- Training Regression Models
- Comparing Regression Models
- Evaluating Regression Models using multiple metrics
- Understanding overfitting caused by high-dimensional feature expansion
- Selecting the appropriate model based on the dataset

---

## About This Repository

This repository is part of my **Machine Learning learning journey**, where I implement machine learning algorithms using **Python** and **Scikit-learn**, solve practical problems, and build projects to strengthen both my theoretical understanding and practical coding skills.
