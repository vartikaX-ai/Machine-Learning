# Clustering

# About this Repository

This repository contains my implementations and practical projects on **Clustering** algorithms using **Python** and **Scikit-learn**.

The objective of this repository is to understand the complete workflow of clustering algorithms, including data preprocessing, feature scaling, distance calculation, cluster formation, cluster evaluation, hierarchical clustering, and real-world customer segmentation.

---

# Algorithms Covered

- K-Means Clustering
- Hierarchical (Agglomerative) Clustering

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

## Hierarchical Clustering

- Introduction to Hierarchical Clustering
- Agglomerative Clustering
- Divisive Clustering
- Linkage Methods
  - Single Linkage
  - Complete Linkage
  - Average Linkage
  - Ward Linkage
- Dendrogram
- Choosing the Optimal Number of Clusters using Dendrogram
- Euclidean Distance
- Feature Scaling
- Silhouette Score
- Cluster Labels
- Cluster Visualization
- Customer Segmentation

---

# Projects

## Mall Customer Segmentation using K-Means

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

## Wholesale Customer Segmentation using Hierarchical Clustering

This project segments wholesale customers based on their annual spending across different product categories using **Agglomerative Hierarchical Clustering**.

### Workflow

- Load the dataset
- Select spending-related features
- Apply feature scaling using `StandardScaler`
- Build a Dendrogram to estimate the optimal number of clusters
- Train the Agglomerative Clustering model
- Visualize customer clusters
- Evaluate the clustering using the Silhouette Score

---

# Technologies Used

- Python
- Pandas
- Matplotlib
- SciPy
- Scikit-learn

---

# Libraries Used

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sc

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
```

---

# Repository Structure

```text
Clustering/
│
├── datasets/
│   ├── Mall_Customers.csv
│   ├── Wholesale customers data.csv
│
├── projects/
│   ├── Mall_Customer_KMeans.py
│   └── Wholescale_Customer_hierarchial.py
│
├── README.md
└── requirements.txt
```

---

# Skills Gained

- Understanding Unsupervised Learning
- Data Preprocessing
- Feature Scaling
- K-Means Clustering
- Hierarchical Clustering
- Agglomerative Clustering
- Linkage Methods
- Dendrogram Analysis
- Choosing the Optimal Number of Clusters
- Elbow Method
- Inertia
- Silhouette Score
- Cluster Evaluation
- Cluster Visualization
- Customer Segmentation

---

#ering
- Spectral Clustering
