import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

iris = load_iris()

X = pd.DataFrame(iris.data,columns=iris.feature_names)
y = pd.Series(iris.target,name="Flower Type")

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train,y_train)
X_test_lda = lda.transform(X_test)

svc_lda = SVC()
svc_lda.fit(X_train_lda,y_train)
svc = SVC()
svc.fit(X_train,y_train)

pred_lda = svc_lda.predict(X_test_lda)
print(pred_lda[:10])
pred = svc.predict(X_test)
print(pred[:10])

print("Variance: ",lda.explained_variance_ratio_)

print("Performance of SVC with LDA: ")
acc_lda = accuracy_score(y_test,pred_lda)
print("Accuracy: ",acc_lda)

cm_lda = confusion_matrix(y_test,pred_lda)
plt.title("Confusion Matrix with LDA")
sns.heatmap(cm_lda,annot=True,xticklabels=["Setosa","Versicolor","Virginica"],yticklabels=["Setosa","Versicolor","Virginica"])
plt.show()

report_lda = classification_report(y_test,pred_lda)
print("Classification report with LDA: ")
print(report_lda)

print("Performance of SVC: ")
acc = accuracy_score(y_test,pred)
print("Accuracy: ",acc)

cm = confusion_matrix(y_test,pred)
plt.title("Confusion Matrix")
sns.heatmap(cm,annot=True,xticklabels=["Setosa","Versicolor","Virginica"],yticklabels=["Setosa","Versicolor","Virginica"])
plt.show()

report = classification_report(y_test,pred)
print("Classification report: ")
print(report)