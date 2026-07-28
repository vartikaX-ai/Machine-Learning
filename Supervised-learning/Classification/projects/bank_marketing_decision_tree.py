import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import classification_report,f1_score,confusion_matrix,precision_score,recall_score,accuracy_score,roc_auc_score,roc_curve
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

train_data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\train.csv")
test_data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\test.csv")

print(train_data.isnull().sum())

oe = OneHotEncoder(drop='first',sparse_output=False)
encoded_df = pd.DataFrame(oe.fit_transform(train_data[["job","marital","education","default","housing","loan","contact","month","poutcome"]]),columns = oe.get_feature_names_out(["job","marital","education","default","housing","loan","contact","month","poutcome"]),index = train_data.index)
le = LabelEncoder()
encoded_target = pd.Series(le.fit_transform(train_data["y"]),name="y")
df = train_data.drop(["job","marital","education","default","housing","loan","contact","month","poutcome","y"],axis=1)
new_df = pd.concat([encoded_df,df],axis=1)

encoded_test = pd.DataFrame(oe.transform(test_data[["job","marital","education","default","housing","loan","contact","month","poutcome"]]),columns = oe.get_feature_names_out(["job","marital","education","default","housing","loan","contact","month","poutcome"]),index = test_data.index)
encoded_ttarget = pd.Series(le.transform(test_data["y"]),name="y")
df_test = test_data.drop(["job","marital","education","default","housing","loan","contact","month","poutcome","y"],axis=1)
new_df_test = pd.concat([encoded_test,df_test],axis=1)

X_train = new_df
y_train = encoded_target

X_test = new_df_test
y_test = encoded_ttarget

dc = DecisionTreeClassifier(criterion='gini',max_depth=6,random_state=42)
dc.fit(X_train,y_train)

pred = dc.predict(X_test)
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

plt.title("Decision Tree Classification")
plot_tree(dc,feature_names=X_train.columns,max_depth=3,rounded=True,filled=True)
plt.show()

predict_probability = dc.predict_proba(X_test)
fpr,tpr,threshold = roc_curve(y_test,predict_probability[:,1])
auc = roc_auc_score(y_test,predict_probability[:,1])

plt.plot(fpr,tpr,color="red",label=f"AUC : {auc :.2f}")
plt.plot([0,1],[0,1],color="blue")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

importance = pd.DataFrame({
    "Features" : X_train.columns,
    "Importance" : dc.feature_importances_
})

print(importance.sort_values(by = "Importance",ascending=False))