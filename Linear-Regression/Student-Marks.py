import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\Student_Marks.csv")
print("No. of rows: ",df.shape[0],"\n","No. of columns: ",df.shape[1])

print(df.isnull().sum())

print(df.corr())

sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.show()

X = df[["number_courses","time_study"]].values
y = df["Marks"].values

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = LinearRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)
print(pred)

r2 = r2_score(y_test,pred)
print("R2 Score: ",r2)

mse = mean_squared_error(y_test,pred)
print("MSE: ",mse)

mae = mean_absolute_error(y_test,pred)
print("MAE: ",mae)

rmse = root_mean_squared_error(y_test,pred)
print("RMSE: ",rmse)