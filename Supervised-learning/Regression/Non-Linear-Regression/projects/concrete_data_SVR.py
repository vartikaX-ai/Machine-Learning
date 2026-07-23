import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\concrete_data.csv")

print(df.isnull().sum())

plt.title("Correlation matrix")
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.show()

X = df.drop("concrete_compressive_strength",axis=1)
y = df["concrete_compressive_strength"].values

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

Xscaler = StandardScaler()
yscaler = StandardScaler()
X_train_scaled = Xscaler.fit_transform(X_train)
X_test_scaled = Xscaler.transform(X_test)
y_train_scaled = yscaler.fit_transform(y_train.reshape(-1,1)).ravel()

sv = SVR(kernel="rbf",C=100,epsilon=1)
sv.fit(X_train_scaled,y_train_scaled)

pred = sv.predict(X_test_scaled)
pred = yscaler.inverse_transform(pred.reshape(-1,1))
print(pred[:10])

r2 = r2_score(y_test,pred)
print("R2 Score: ",r2)

mse = mean_squared_error(y_test,pred)
print("MSE: ",mse)

mae = mean_absolute_error(y_test,pred)
print("MAE: ",mae)

rmse = root_mean_squared_error(y_test,pred)
print("RMSE: ",rmse)