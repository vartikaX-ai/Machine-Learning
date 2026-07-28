import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,roc_curve,confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\spam.csv",encoding="ISO-8859-1")

print(df.isnull().sum())

X = df["v2"]
y = df["v1"]

le = LabelEncoder()
y = pd.Series(le.fit_transform(y),name="v1")

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

cv = CountVectorizer()
X_train_count = cv.fit_transform(X_train)
X_test_count = cv.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train_count,y_train)

pred = mnb.predict(X_test_count)
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
sns.heatmap(cm,annot=True,xticklabels=["Ham","Spam"],yticklabels=["Ham","Spam"])
plt.show()

report = classification_report(y_test,pred)
print("Classification report: ")
print(report)

predict_probability = mnb.predict_proba(X_test_count)
print(predict_probability[:10])

fpr,tpr,threshold = roc_curve(y_test,predict_probability[:,1])
auc = roc_auc_score(y_test,predict_probability[:,1])

plt.plot(fpr,tpr,color="red",label=f"AUC : {auc:.2f}")
plt.plot([0,1],[0,1],color="blue")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()