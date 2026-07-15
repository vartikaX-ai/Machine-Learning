# Classification

## About this Repository

This repository contains my implementations and practical projects on **Classification** algorithms using **Python** and **Scikit-learn**.

The objective of this repository is to understand the complete workflow of classification models, including data preprocessing, feature engineering, model training, prediction, probability estimation, threshold tuning, tree-based learning, model evaluation, and applying classification algorithms to real-world datasets.

---

## Algorithms Covered

- Logistic Regression
- Decision Tree Classifier

---

## Topics Covered

### Classification Fundamentals

- Introduction to Classification
- Binary Classification
- Multiclass Classification

### Logistic Regression

- Logistic Regression
- Sigmoid Function
- Decision Boundary
- Log Loss
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC Score
- Threshold Tuning
- Precision-Recall Trade-off
- One-vs-Rest (OVR)
- Multinomial Logistic Regression
- Probability Prediction (`predict_proba()`)

### Decision Tree

- Decision Tree Classifier
- Gini Impurity
- Entropy
- Information Gain
- Decision Tree Visualization
- Tree Depth
- Feature Importance

### Data Preprocessing

- Data Cleaning
- Handling Missing Values
- One-Hot Encoding
- Label Encoding
- Feature Scaling (StandardScaler)
- Train-Test Split
- Model Training
- Prediction
- Model Evaluation
- Class Imbalance Handling

---

## Evaluation Metrics

The models are evaluated using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- ROC Curve
- AUC Score
- Feature Importance

---

# Projects

## 1. Titanic Survival Prediction (Logistic Regression)

### Objective

- Predict whether a passenger survived the Titanic disaster using Logistic Regression.

### Features Used

- Passenger Class
- Sex
- Age
- Number of Siblings/Spouses
- Number of Parents/Children
- Fare
- Embarked
- and other passenger-related features.

### Concepts Practiced

- Missing Value Handling
- One-Hot Encoding
- Feature Scaling
- Logistic Regression
- Binary Classification
- Train-Test Split
- Probability Prediction (`predict_proba()`)
- Confusion Matrix
- ROC Curve
- AUC Score
- Classification Report
- Model Evaluation

---

## 2. Bank Marketing Prediction (Decision Tree)

### Objective

- Predict whether a customer will subscribe to a term deposit using a Decision Tree Classifier.

### Features Used

- Age
- Job
- Marital Status
- Education
- Account Balance
- Housing Loan
- Personal Loan
- Contact Type
- Contact Month
- Campaign Details
- Previous Campaign Information
- Previous Campaign Outcome

### Concepts Practiced

- One-Hot Encoding
- Label Encoding
- Decision Tree Classification
- Gini Impurity
- Entropy
- Decision Tree Visualization
- Feature Importance
- Confusion Matrix
- Classification Report
- ROC Curve
- AUC Score
- Model Evaluation

---

## Repository Structure

```text
Classification/
│
├── datasets/
│   ├── Titanic-Dataset.csv
│   ├── bank_marketing_train.csv
│   └── bank_marketing_test.csv
│
├── projects/
│   ├── Titanic_data_Logistic.py
│   └── bank_marketing_decision_tree.py
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
- Imbalanced-learn

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/Machine-Learning.git
```

Move into the project directory

```bash
cd Machine-Learning/Classification
```

Install the required libraries

```bash
pip install -r requirements.txt
```

---

## Learning Outcomes

Through these implementations, I learned:

- Understanding binary and multiclass classification.
- Implementing Logistic Regression and Decision Tree Classifiers using Scikit-learn.
- Understanding Sigmoid Function and Decision Boundary.
- Understanding Gini Impurity and Entropy.
- Comparing Gini and Entropy criteria.
- Working with probability predictions using `predict_proba()`.
- Applying threshold tuning.
- Comparing One-vs-Rest (OVR) and Multinomial Logistic Regression.
- Building and visualizing Decision Trees.
- Interpreting Feature Importance.
- Handling missing values and categorical data.
- Applying One-Hot Encoding and Label Encoding.
- Feature Scaling using StandardScaler.
- Handling class imbalance using sampling techniques.
- Training classification models.
- Making predictions.
- Evaluating model performance using multiple classification metrics.
- Interpreting ROC Curve and AUC Score.

---

## About This Repository

This repository is part of my **Machine Learning learning journey**, where I implement machine learning algorithms using **Python** and **Scikit-learn**, solve practical problems, and build projects to strengthen both my theoretical understanding and practical machine learning skills.
