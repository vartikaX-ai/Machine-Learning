import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from sklearn.impute import SimpleImputer

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\Titanic-Dataset.csv")

print(df.isnull().sum())

si_age = SimpleImputer(strategy="median")
df["Age"] = si_age.fit_transform(df[["Age"]]).ravel()

si_emb = SimpleImputer(strategy='most_frequent')
df["Embarked"] = si_emb.fit_transform(df[["Embarked"]]).ravel()

oe = OneHotEncoder(drop='first',sparse_output=False)
encoded_df = pd.DataFrame(oe.fit_transform(df[["Sex","Embarked"]]),columns=oe.get_feature_names_out(["Sex","Embarked"]),index=df.index)
df = df.drop(["PassengerId","Name","Sex","Ticket","Cabin","Embarked"],axis=1)
new_df = pd.concat([encoded_df,df],axis=1)

X = new_df.drop("Survived",axis=1)
y = new_df["Survived"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

xgc = XGBClassifier()
xgc.fit(X_train,y_train)

pred = xgc.predict(X_test)
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
sns.heatmap(cm,annot=True,xticklabels=["Not survived","Survived"],yticklabels=["Not survived","Survived"])
plt.show()

report = classification_report(y_test,pred)
print("Classification report: ")
print(report)
