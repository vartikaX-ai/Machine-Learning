import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\heart_disease.csv")

print(df.isnull().sum())

X = df.drop("target",axis=1)
y = df["target"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

dc = DecisionTreeClassifier()

bc = BaggingClassifier(estimator=dc,n_estimators=100)
bc.fit(X_train,y_train)

pred = bc.predict(X_test)
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
print("Classification report: ")
print(report)