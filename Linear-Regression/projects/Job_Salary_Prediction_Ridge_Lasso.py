import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,root_mean_squared_error

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\job_salary_prediction_dataset.csv")

print(df.isnull().sum())

oe = OneHotEncoder(drop='first',sparse_output=False)
encoded_df = pd.DataFrame(oe.fit_transform(df[["job_title","education_level","industry","company_size","location","remote_work"]]),columns=oe.get_feature_names_out(["job_title","education_level","industry","company_size","location","remote_work"]),index=df.index)
df = df.drop(["job_title","education_level","industry","company_size","location","remote_work"],axis=1)

new_df = pd.concat((encoded_df,df),axis=1)

plt.title("Correlation matrix")
sns.heatmap(new_df.corr(),annot=True,cmap="coolwarm")
plt.show()

X = new_df.drop("salary",axis=1)
y = new_df["salary"]

sns.boxplot(X)
plt.show()

sns.boxplot(y)
plt.show()

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rr = Ridge(alpha=10)
rr.fit(X_train,y_train)

pred = rr.predict(X_test)
print(pred[:10])

r2 = r2_score(y_test,pred)
print("R2 Score: ",r2)

mse = mean_squared_error(y_test,pred)
print("MSE: ",mse)

mae = mean_absolute_error(y_test,pred)
print("MAE: ",mae)

rmse = root_mean_squared_error(y_test,pred)
print("RMSE: ",rmse)

print("Intercept: ",rr.intercept_)
print("Coefficient: ",rr.coef_)

plt.scatter(y_test,pred)
plt.xlabel("Actual label")
plt.ylabel("predicted label")
plt.title("Job Salary Prediction")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,root_mean_squared_error

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\job_salary_prediction_dataset.csv")

print(df.isnull().sum())

oe = OneHotEncoder(drop='first',sparse_output=False)
encoded_df = pd.DataFrame(oe.fit_transform(df[["job_title","education_level","industry","company_size","location","remote_work"]]),columns=oe.get_feature_names_out(["job_title","education_level","industry","company_size","location","remote_work"]),index=df.index)
df = df.drop(["job_title","education_level","industry","company_size","location","remote_work"],axis=1)

new_df = pd.concat((encoded_df,df),axis=1)

plt.title("Correlation matrix")
sns.heatmap(new_df.corr(),annot=True,cmap="coolwarm")
plt.show()

X = new_df.drop("salary",axis=1)
y = new_df["salary"]

sns.boxplot(X)
plt.show()

sns.boxplot(y)
plt.show()

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = Lasso(alpha=1000)
lr.fit(X_train,y_train)

pred = lr.predict(X_test)
print(pred[:10])

r2 = r2_score(y_test,pred)
print("R2 Score: ",r2)

mse = mean_squared_error(y_test,pred)
print("MSE: ",mse)

mae = mean_absolute_error(y_test,pred)
print("MAE: ",mae)

rmse = root_mean_squared_error(y_test,pred)
print("RMSE: ",rmse)

print("Intercept: ",lr.intercept_)
print("Coefficient: ",lr.coef_)

plt.scatter(y_test,pred)
plt.xlabel("Actual label")
plt.ylabel("predicted label")
plt.title("Job Salary Prediction")
plt.show()