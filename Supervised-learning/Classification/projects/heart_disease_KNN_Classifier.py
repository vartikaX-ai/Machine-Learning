import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,roc_auc_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\heart_disease.csv")

print(df.isnull().sum())

X = df.drop("target",axis=1)
y = df["target"]

print(y.value_counts())

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

us = RandomUnderSampler()
rux,ruy = us.fit_resample(X_train,y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(rux)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled,ruy)

pred = knn.predict(X_test_scaled)
print(pred[:10])

acc = accuracy_score(y_test,pred)
print("Accuracy: ",acc)

precision = precision_score(y_test,pred)
print("Precision: ",precision)

recall = recall_score(y_test,pred)
print("Recall: ",recall)

f1 = f1_score(y_test,pred)
print("F1 Score: ",f1)

cm = confusion_matrix(y_test,pred)
plt.title("Confusion Matrix")
sns.heatmap(cm,annot=True,xticklabels=["No","Yes"],yticklabels=["No","Yes"])
plt.show()

report = classification_report(y_test,pred)
print("Classification Report: ")
print(report)

predict_probability = knn.predict_proba(X_test_scaled)
fpr,tpr,threshold = roc_curve(y_test,predict_probability[:,1])
auc = roc_auc_score(y_test,predict_probability[:,1])

plt.plot(fpr,tpr,color="red",label=f"AUC: {auc : .2f}")
plt.plot([0,1],[0,1],color="blue")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()