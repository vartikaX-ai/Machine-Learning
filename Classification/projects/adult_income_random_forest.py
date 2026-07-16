import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score,precision_score,f1_score,confusion_matrix,classification_report

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\adult.csv")

print(df.isnull().sum())

oe = OneHotEncoder(drop='first',sparse_output=False)
encoded_df = pd.DataFrame(oe.fit_transform(df[["workclass","education","marital-status","occupation","relationship","race","gender","native-country"]]),columns=oe.get_feature_names_out(["workclass","education","marital-status","occupation","relationship","race","gender","native-country"]),index=df.index)
le = LabelEncoder()
y = pd.Series(le.fit_transform(df["income"]),name="Income")
df = df.drop(["workclass","education","marital-status","occupation","relationship","race","gender","native-country","income"],axis=1)
X = pd.concat([encoded_df,df],axis=1)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

rf = RandomForestClassifier(n_estimators=100,max_depth=6)
rf.fit(X_train,y_train)

pred = rf.predict(X_test)
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
sns.heatmap(cm,annot=True,xticklabels=["<=50",">50"],yticklabels=["<=50",">50"])
plt.show()

report = classification_report(y_test,pred)
print("Classification Report: ")
print(report)

predict_probability = rf.predict_proba(X_test)
fpr,tpr,threshold = roc_curve(y_test,predict_probability[:,1])
auc = roc_auc_score(y_test,predict_probability[:,1])

plt.plot(fpr,tpr,color="blue",label=f"AUC: {auc:.2f}")
plt.plot([0,1],[0,1],color="red")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

importance = pd.DataFrame({
    "Features":X.columns,
    "Importance":rf.feature_importances_
})

importance = importance.sort_values(by = "Importance",ascending=False)
print(importance)