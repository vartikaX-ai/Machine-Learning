import pandas as pd
import scipy.cluster.hierarchy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\Wholesale customers data.csv")

X = df[["Fresh","Milk", "Grocery","Frozen","Detergents_Paper","Delicassen"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sc.dendrogram(sc.linkage(X_scaled,method="complete",metric='euclidean'))
plt.title("Dendogram")
plt.xlabel("No. of data points")
plt.ylabel("Distance of cluster")
plt.show()

ac = AgglomerativeClustering(n_clusters=2,linkage='complete')
labels = ac.fit_predict(X_scaled)

print(labels)
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=labels)
plt.title("Hierarchial Clustering")
plt.xlabel("Fresh")
plt.ylabel("Milk")
plt.show()

print("Silhouette Score: ",silhouette_score(X_scaled,labels))