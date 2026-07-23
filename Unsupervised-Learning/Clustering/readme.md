# Clustering

# About this Repository

This repository contains my implementations and practical projects on **Clustering** algorithms using **Python** and **Scikit-learn**.

The objective of this repository is to understand the complete workflow of clustering algorithms, from data preprocessing and feature scaling to cluster evaluation and visualization.

---
# Algorithms Covered

- K-Means Clustering

---

# Topics Covered

## K-Means Clustering

- Introduction to Clustering
- Supervised vs Unsupervised Learning
- K-Means Clustering
- Euclidean Distance
- Centroid
- Feature Scaling
- K-Means Algorithm
- Choosing the Optimal Number of Clusters
- Elbow Method
- Within Cluster Sum of Squares (WCSS)
- Inertia
- Silhouette Score
- Cluster Labels
- Cluster Centers
- Predicting Cluster for New Data
- Customer Segmentation
- Cluster Visualization

---

# Project

## Mall Customer Segmentation

This project groups mall customers based on their **Annual Income** and **Spending Score** using the **K-Means Clustering** algorithm.

### Workflow

- Load the dataset
- Select relevant features
- Apply feature scaling using `StandardScaler`
- Find the optimal number of clusters using the Elbow Method
- Train the K-Means model
- Visualize customer clusters and centroids
- Evaluate the clustering using Inertia and Silhouette Score
- Predict the cluster for a new customer

---

# Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

---

# Libraries Used

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
```

---

# Repository Structure

```text
Clustering/
│
├── datasets/
│   └── Mall_Customers.csv
│
├── projects/
│   └── Mall_Customer_KMeans.py
├── README.md
└── requirements.txt
```

---

# Skills Gained

- Understanding Unsupervised Learning
- Feature Scaling
- K-Means Clustering
- Customer Segmentation
- Choosing the Optimal Number of Clusters
- Elbow Method
- Inertia
- Silhouette Score
- Cluster Visualization
- Predicting Clusters for New Data

---

# Future Updates

The following clustering algorithms will be added in future updates:

- Hierarchical Clustering
- DBSCAN
