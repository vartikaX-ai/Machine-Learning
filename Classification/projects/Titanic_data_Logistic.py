import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_curve,roc_auc_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\Titanic-Dataset.csv")

print(df.isnull().sum())

si_age = SimpleImputer(strategy='median')
df["Age"] = si_age.fit_transform(df[["Age"]]).ravel()

si_emb = SimpleImputer(strategy='most_frequent')
df["Embarked"] = si_emb.fit_transform(df[["Embarked"]]).ravel()

oe = OneHotEncoder(drop='first',sparse_output=False)
encoded_df = pd.DataFrame(oe.fit_transform(df[["Sex","Embarked"]]),columns=oe.get_feature_names_out(["Sex","Embarked"]),index = df.index)
df = df.drop(["PassengerId","Name","Sex","Ticket","Cabin","Embarked"],axis=1)
new_df = pd.concat([encoded_df,df],axis=1)
print(new_df.head())

X = new_df.drop("Survived",axis=1)
y = new_df["Survived"].values

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y  
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_scaled,y_train)

pred = lr.predict(X_test_scaled)
print(pred[:10])

acc = accuracy_score(y_test,pred)
print("Accuracy: ",acc)

cm = confusion_matrix(y_test,pred)
plt.title("Confusion Matrix")
sns.heatmap(cm,annot=True,xticklabels=["not survived","survived"],yticklabels=["not survived","survived"])
plt.show()

precision = precision_score(y_test,pred)
print("Precision score: ",precision)

recall = recall_score(y_test,pred)
print("Recall score: ",recall)

f1 = f1_score(y_test,pred)
print("F1 Score: ",f1)

report = classification_report(y_test,pred)
print("Classification report: ")
print(report)

predict_probability = lr.predict_proba(X_test_scaled)

auc = roc_auc_score(y_test,predict_probability[:,1])
fpr,tpr,threshold = roc_curve(y_test,predict_probability[:,1])

plt.plot(fpr,tpr,color="red",label=f"AUC : {auc : .2f}")
plt.plot([0,1],[0,1],color="blue")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC AUC Curve")
plt.show()