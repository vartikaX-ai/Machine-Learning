# Principal Component Analysis (PCA) with Logistic Regression

## About this Project

This project demonstrates how to use **Principal Component Analysis (PCA)** for **dimensionality reduction** before training a **Logistic Regression** classifier on the **Breast Cancer Wisconsin Dataset**.

The workflow includes data preprocessing, feature standardization, determining the optimal number of principal components using a **Scree Plot**, transforming the dataset into a lower-dimensional space, and evaluating the classification model.

---

## Objectives

- Reduce the dimensionality of the dataset while preserving maximum information.
- Understand the explained variance of each principal component.
- Visualize the Scree Plot to determine the optimal number of principal components.
- Train a Logistic Regression model using PCA-transformed data.
- Evaluate the model using common classification metrics.

---

## Dataset

- **Dataset:** Breast Cancer Wisconsin Dataset
- **Source:** `sklearn.datasets.load_breast_cancer()`

---

## Topics Covered

- Principal Component Analysis (PCA)
- Dimensionality Reduction
- Feature Standardization
- Explained Variance Ratio
- Scree Plot
- Principal Components
- Logistic Regression
- Model Training
- Model Prediction
- Accuracy Score
- Confusion Matrix
- Classification Report

---

## Workflow

1. Load the Breast Cancer dataset.
2. Split the dataset into training and testing sets.
3. Standardize the features using `StandardScaler`.
4. Apply PCA to determine the explained variance of each principal component.
5. Plot the Scree Plot.
6. Select the first two principal components.
7. Transform both training and testing datasets.
8. Train a Logistic Regression model.
9. Make predictions on the test dataset.
10. Evaluate the model using:
    - Accuracy Score
    - Confusion Matrix
    - Classification Report

---

## Libraries Used

- pandas
- seaborn
- matplotlib
- scikit-learn

---

## Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report

---

## Output

The project generates:

- Scree Plot
- Explained Variance Ratio
- Model Predictions
- Accuracy Score
- Confusion Matrix
- Classification Report

---

## Learning Outcomes

After completing this project, you will understand:

- Why feature scaling is important before PCA.
- How PCA reduces the dimensionality of a dataset.
- How to interpret the Scree Plot.
- How to choose the number of principal components.
- How to transform data using PCA.
- How PCA can improve machine learning workflows.
- How to combine PCA with Logistic Regression for classification tasks.

---

## Project Structure

```text
PCA/
│
├── projects/
│   └── Breast_Cancer_PCA_Logistic_Regression.py
│
├── README.md
└── requirements.txt
```

---

## Requirements

```text
pandas
matplotlib
seaborn
scikit-learn
```

