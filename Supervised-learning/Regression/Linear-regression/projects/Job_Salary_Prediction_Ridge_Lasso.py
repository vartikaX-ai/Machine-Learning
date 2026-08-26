from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error
import pandas as pd

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\job_salary_prediction_dataset.csv")

print(df.isnull().sum())

oe = OneHotEncoder(drop='first',sparse_output=False)
encoded_df = pd.DataFrame(oe.fit_transform(df[["job_title","education_level","industry","company_size","location","remote_work"]]),columns=oe.get_feature_names_out(["job_title","education_level","industry","company_size","location","remote_work"]),index=df.index)
df = df.drop(["job_title","education_level","industry","company_size","location","remote_work"],axis=1)
new_df = pd.concat([encoded_df,df],axis=1)

X = new_df.drop("salary",axis=1)
y = new_df["salary"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=10)
ridge.fit(X_train_scaled,y_train)

pred_ri = ridge.predict(X_test_scaled)
print(pred_ri[:5])

lasso = Lasso(alpha=1000)
lasso.fit(X_train_scaled,y_train)

pred_la = lasso.predict(X_test_scaled)
print(pred_la[:5])

print("Performance of Ridge regression: ")
r2 = r2_score(y_test,pred_ri)
print("R2 Score: ",r2)

mse = mean_squared_error(y_test,pred_ri)
print("Mean Squarred Error: ",mse)

mae = mean_absolute_error(y_test,pred_ri)
print("Mean Absolute Error: ",mae)

rmse = root_mean_squared_error(y_test,pred_ri)
print("Root Mean Squared Error: ",rmse)

print("Performance of Lasso Regression: ")
r21 = r2_score(y_test,pred_la)
print("R2 Score: ",r21)

mse1 = mean_squared_error(y_test,pred_la)
print("Mean Squarred Error: ",mse1)

mae1 = mean_absolute_error(y_test,pred_la)
print("Mean Absolute Error: ",mae1)

rmse1 = root_mean_squared_error(y_test,pred_la)
print("Root Mean Squared Error: ",rmse1)
