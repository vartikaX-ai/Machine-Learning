import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,root_mean_squared_error,mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler,LabelEncoder

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\Housing.csv")

print(df.isnull().sum())

le = LabelEncoder()
df["mainroad"] = le.fit_transform(df["mainroad"])
df["guestroom"] = le.fit_transform(df["guestroom"])
df["basement"] = le.fit_transform(df["basement"])
df["hotwaterheating"] = le.fit_transform(df["hotwaterheating"])
df["airconditioning"] = le.fit_transform(df["airconditioning"])
df["prefarea"] = le.fit_transform(df["prefarea"])
df["furnishingstatus"] = le.fit_transform(df["furnishingstatus"])

plt.title("Correlation Matrix")
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.show()

X = df.drop("price",axis=1)
y = df["price"].values

plt.title("Outlier matrix")
sns.boxplot(data=X)
plt.show()

plt.title("Target outlier matrix")
sns.boxplot(y)
plt.show()

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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

plt.scatter(y_test,pred,color="blue")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House price prediction")
plt.show()