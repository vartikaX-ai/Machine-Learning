# Classification

## About this Repository

This repository contains my implementations and practical projects on **Classification** algorithms using **Python** and **Scikit-learn**.

The objective of this repository is to understand the complete workflow of classification models, including data preprocessing, feature engineering, model training, prediction, probability estimation, threshold tuning, distance-based learning, tree-based learning, ensemble learning, margin-based learning, model evaluation, and applying classification algorithms to real-world datasets.

---

## Algorithms Covered

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN) Classifier
- Support Vector Machine (SVM) Classifier

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

### Random Forest

- Random Forest Classifier
- Ensemble Learning
- Bootstrap Sampling
- Bagging
- Majority Voting
- Number of Trees (`n_estimators`)
- Tree Depth (`max_depth`)
- Feature Importance
- Model Comparison

### K-Nearest Neighbors (KNN)

- K-Nearest Neighbors Classification
- Distance-Based Learning
- Euclidean Distance
- Majority Voting
- Choosing the Best K (`n_neighbors`)
- Uniform vs Distance Weights
- Feature Scaling
- Model Comparison

### Support Vector Machine (SVM)

- Support Vector Machine Classification
- Hyperplane
- Support Vectors
- Margin Maximization
- Kernel Trick
- Linear Kernel
- Polynomial Kernel
- RBF Kernel
- Sigmoid Kernel
- Hyperparameter Tuning (`C`, `gamma`)
- Probability Prediction (`predict_proba()`)
- Binary Classification
- Multiclass Classification
- Model Comparison

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
- Class Imbalance Handling (Random Over Sampling & Random Under Sampling)

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

## 3. Adult Income Prediction (Random Forest)

### Objective

- Predict whether a person's annual income is greater than **$50K** using a Random Forest Classifier.

### Features Used

- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Gender
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country
- Final Weight (`fnlwgt`)
- and other demographic features.

### Concepts Practiced

- One-Hot Encoding
- Label Encoding
- Random Forest Classification
- Ensemble Learning
- Bootstrap Sampling
- Bagging
- Feature Importance
- Confusion Matrix
- Classification Report
- ROC Curve
- AUC Score
- Model Evaluation

---

## 4. Heart Disease Prediction (K-Nearest Neighbors)

### Objective

- Predict whether a patient has heart disease using the K-Nearest Neighbors (KNN) algorithm.

### Features Used

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression
- Slope
- Number of Major Vessels
- Thalassemia
- and other medical features.

### Concepts Practiced

- Data Preprocessing
- Random Under Sampling
- StandardScaler
- K-Nearest Neighbors Classification
- Choosing the Best K (`n_neighbors`)
- Distance-Based Learning
- Uniform vs Distance Weights
- Confusion Matrix
- Classification Report
- ROC Curve
- AUC Score
- Model Evaluation

---

## 5. Diabetes Prediction (Support Vector Machine)

### Objective

- Predict whether a patient has diabetes using a Support Vector Machine (SVM) classifier.

### Features Used

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

### Concepts Practiced

- Data Preprocessing
- Random Over Sampling
- StandardScaler
- Support Vector Machine Classification
- Kernel Selection
- Hyperparameter Tuning (`C`, `gamma`)
- Probability Prediction (`predict_proba()`)
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
│   ├── bank_marketing_test.csv
│   ├── adult.csv
│   ├── heart_disease.csv
│   └── diabetes.csv
│
├── projects/
│   ├── Titanic_data_Logistic.py
│   ├── bank_marketing_decision_tree.py
│   ├── adult_income_random_forest.py
│   ├── heart_disease_KNN_Classifier.py
│   └── diabetes_SVC.py
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
- Implementing Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, and Support Vector Machine classifiers using Scikit-learn.
- Understanding Sigmoid Function and Decision Boundary.
- Understanding Gini Impurity and Entropy.
- Comparing Gini and Entropy criteria.
- Understanding Ensemble Learning, Bootstrap Sampling, and Bagging.
- Understanding distance-based learning and the K-Nearest Neighbors algorithm.
- Selecting the optimal value of **K** (`n_neighbors`).
- Comparing uniform and distance-based voting in KNN.
- Understanding Support Vector Machines (SVM), hyperplanes, margins, and support vectors.
- Understanding the Kernel Trick.
- Comparing Linear, Polynomial, RBF, and Sigmoid kernels.
- Selecting appropriate values of `C` and `gamma`.
- Applying SVM to binary and multiclass classification problems.
- Working with probability predictions using `predict_proba()`.
- Applying threshold tuning.
- Comparing One-vs-Rest (OVR) and Multinomial Logistic Regression.
- Building and visualizing Decision Trees.
- Understanding Feature Importance.
- Handling missing values and categorical data.
- Applying One-Hot Encoding and Label Encoding.
- Feature Scaling using StandardScaler.
- Handling class imbalance using Random Over Sampling and Random Under Sampling.
- Training classification models.
- Comparing Decision Tree and Random Forest performance.
- Making predictions on real-world datasets.
- Evaluating model performance using multiple classification metrics.
- Interpreting ROC Curve and AUC Score.

---

## About This Repository

This repository is part of my **Machine Learning learning journey**, where I implement machine learning algorithms using **Python** and **Scikit-learn**, solve practical problems, and build projects to strengthen both my theoretical understanding and practical machine learning skills.
