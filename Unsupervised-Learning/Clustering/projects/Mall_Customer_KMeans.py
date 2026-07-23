import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\Mall_Customers.csv")

X = df[["Annual Income (k$)","Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
for i in range(1,15):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
print(inertia)

plt.plot(range(1,15),inertia,marker="o")
plt.xlabel("No. of cluster")
plt.ylabel("Inertia")
plt.show()

fkmeans = KMeans(n_clusters=5,n_init="auto",random_state=42)
fkmeans.fit(X_scaled)

plt.scatter(X_scaled[:,0],X_scaled[:,1],c=fkmeans.labels_,marker="o")
plt.scatter(fkmeans.cluster_centers_[:,0],fkmeans.cluster_centers_[:,1],color="red",marker="X")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

print("Labels: ",fkmeans.labels_)

X_test = [[65,82]]
X_test_scaled = scaler.transform(X_test)
pred = fkmeans.predict(X_test_scaled)
print("Predicted Cluster: ",pred)

print("Inertia: ",fkmeans.labels_)
print("Silhouette Score: ",silhouette_score(X_scaled,fkmeans.labels_))