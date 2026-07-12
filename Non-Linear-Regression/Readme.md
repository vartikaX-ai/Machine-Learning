# Non-Linear Regression

## About this Repository

This repository contains my implementations and practical projects on **Non-Linear Regression** algorithms using **Python** and **Scikit-learn**.

The objective of this repository is to understand how different non-linear regression algorithms model complex relationships between features and target variables through practical implementation, experimentation, model comparison, hyperparameter tuning, and evaluation.

---

## Algorithms Covered

- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- K-Nearest Neighbors (KNN) Regression
- Support Vector Regression (SVR)

---

## Topics Covered

- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- K-Nearest Neighbors (KNN) Regression
- Support Vector Regression (SVR)
- Polynomial Features
- Feature Engineering
- One-Hot Encoding
- Feature Scaling
- Target Variable Scaling
- Correlation Analysis
- Outlier Detection
- Train-Test Split
- Decision Tree Visualization
- Feature Importance
- Distance Metrics (Euclidean & Manhattan)
- Kernel Functions (Linear, Polynomial, RBF)
- Hyperparameter Tuning
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
- Decision Tree Visualization
- Feature Importance
- Model Evaluation

---

### 3. Diamond Price Prediction using Random Forest Regression

**Objective**

- Predict diamond prices using Random Forest Regression.
- Analyze feature importance.
- Tune model hyperparameters for better performance.

**Dataset**

- Diamonds Prices2022.csv

**Concepts Practiced**

- One-Hot Encoding
- Correlation Analysis
- Random Forest Regression
- Feature Importance
- Hyperparameter Tuning (`n_estimators`, `max_depth`)
- Model Evaluation

---

### 4. Fish Weight Prediction using KNN Regression

**Objective**

- Predict fish weight using K-Nearest Neighbors Regression.
- Compare different values of **K** and distance metrics.
- Identify the best-performing model configuration.

**Dataset**

- Fish_dataset.csv

**Concepts Practiced**

- One-Hot Encoding
- Feature Scaling
- Correlation Analysis
- KNN Regression
- Distance Metrics (Euclidean & Manhattan)
- Hyperparameter Tuning (`n_neighbors`)
- Model Evaluation
- Model Comparison

---

### 5. Concrete Compressive Strength Prediction using Support Vector Regression

**Objective**

- Predict concrete compressive strength using Support Vector Regression.
- Compare different kernel functions and tune SVR hyperparameters to improve performance.

**Dataset**

- concrete_data.csv

**Concepts Practiced**

- Feature Scaling
- Target Variable Scaling
- Correlation Analysis
- Support Vector Regression (SVR)
- Kernel Functions (Linear, RBF, Polynomial)
- Hyperparameter Tuning (`C`, `epsilon`)
- Model Evaluation
- Model Comparison

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

This project demonstrated how feature importance can be used to identify the most influential variables affecting predictions.

---

### Random Forest Regression

Random Forest Regression achieved excellent predictive performance by combining the predictions of multiple Decision Trees.

Different values of **`n_estimators`** and **`max_depth`** were tested to understand their impact on model performance.

**Key Takeaway**

> Ensemble learning improves prediction accuracy and reduces overfitting compared to using a single Decision Tree.

---

### K-Nearest Neighbors (KNN) Regression

KNN Regression predicts continuous values by averaging the target values of the **K nearest neighbors**.

Different values of **`n_neighbors`** and different **distance metrics (Euclidean and Manhattan)** were tested to identify the best-performing model.

Feature scaling was essential because KNN is a distance-based algorithm.

**Key Takeaway**

> The choice of **K**, distance metric, and proper feature scaling significantly impacts the performance of KNN Regression.

---

### Support Vector Regression (SVR)

Support Vector Regression predicts continuous values by finding a function that fits the data while allowing a small prediction error inside an **ε-tube**.

Different **kernel functions** and values of **`C`** and **`epsilon`** were tested to analyze their effect on model performance.

Both feature scaling and target variable scaling were applied to improve training stability and prediction accuracy.

**Key Takeaway**

> The performance of SVR depends heavily on selecting an appropriate kernel and tuning hyperparameters such as **C** and **epsilon**, while proper feature scaling is essential for achieving good results.

---

## Repository Structure

```text
Non-Linear-Regression/
│
├── datasets/
│   ├── CarPrice_Assignment.csv
│   ├── medical_insurance.csv
│   ├── Diamonds Prices2022.csv
│   ├── Fish_dataset.csv
│   └── concrete_data.csv
│
├── projects/
│   ├── Car_Price_Prediction.py
│   ├── Medical_Insurance_Cost_Prediction_Decision_Tree.py
│   ├── Diamond_Price_RandomForest.py
│   ├── Fish_Weight_Prediction_KNN.py
│   └── concrete_data_SVR.py
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
- Random Forest Regression
- K-Nearest Neighbors (KNN) Regression
- Support Vector Regression (SVR)
- Ensemble Learning
- Distance-Based Learning
- Kernel-Based Learning
- Hyperparameter Tuning
- Distance Metrics (Euclidean & Manhattan)
- Kernel Functions (Linear, Polynomial, RBF)
- Feature Engineering
- One-Hot Encoding
- Data Preprocessing
- Feature Scaling
- Target Variable Scaling
- Correlation Analysis
- Detecting Outliers
- Decision Tree Visualization
- Feature Importance
- Training Regression Models
- Comparing Regression Models
- Evaluating Regression Models using multiple metrics
- Understanding overfitting caused by high-dimensional feature expansion
- Improving model performance using ensemble methods
- Selecting the appropriate model based on the dataset

---

## About This Repository

This repository is part of my **Machine Learning learning journey**, where I implement machine learning algorithms using **Python** and **Scikit-learn**, solve practical problems, and build projects to strengthen both my theoretical understanding and practical coding skills.
