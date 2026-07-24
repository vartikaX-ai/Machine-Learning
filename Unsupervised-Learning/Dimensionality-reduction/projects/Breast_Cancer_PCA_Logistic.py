import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

can = load_breast_cancer()

X = pd.DataFrame(can.data,columns=can.feature_names)
y = pd.Series(can.target,name="Diagnosis")

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train)

plt.plot(range(1,len(pca.explained_variance_ratio_)+1),pca.explained_variance_ratio_,marker="o")
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance")
plt.title("Scree plot")
plt.show()

print(pca.explained_variance_ratio_)

fpca = PCA(n_components=2)
X_train = fpca.fit_transform(X_train)
X_test = fpca.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train,y_train)

pred = lr.predict(X_test)
print(pred[:10])

acc = accuracy_score(y_test,pred)
print("Accuracy: ",acc)

cm = confusion_matrix(y_test,pred)
plt.title("Confusion Matrix")
sns.heatmap(cm,annot=True,xticklabels=["No","Yes"],yticklabels=["No","Yes"])
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.show()

report = classification_report(y_test,pred)
print("Classification report: ")
print(report)