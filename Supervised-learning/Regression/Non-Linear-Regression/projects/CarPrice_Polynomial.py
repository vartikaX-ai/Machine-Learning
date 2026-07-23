import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\CarPrice_Assignment.csv")

print(df.isnull().sum())

oe = OneHotEncoder(drop='first',sparse_output=False)
encoded_df = pd.DataFrame(oe.fit_transform(df[["fueltype","aspiration","doornumber","carbody","drivewheel","enginelocation","enginetype","cylindernumber","fuelsystem"]]),columns=oe.get_feature_names_out(["fueltype","aspiration","doornumber","carbody","drivewheel","enginelocation","enginetype","cylindernumber","fuelsystem"]),index=df.index)
df = df.drop(["car_ID","CarName","fueltype","aspiration","doornumber","carbody","drivewheel","enginelocation","enginetype","cylindernumber","fuelsystem"],axis=1)

new_df = pd.concat([encoded_df,df],axis=1)

X = new_df.drop("price",axis=1)
y = new_df["price"].values

plt.title("Correlation matrix")
sns.heatmap(new_df.corr(),annot=True,cmap="coolwarm")
plt.show()

plt.title("Outliers of X")
sns.boxplot(X)
plt.show()

plt.title("Outliers of y")
sns.boxplot(y)
plt.show()

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pf = PolynomialFeatures(degree=2,interaction_only=True)
X_train_poly = pf.fit_transform(X_train)
X_test_poly = pf.transform(X_test)

lr_poly = LinearRegression()
lr_poly.fit(X_train_poly,y_train)

pred_poly = lr_poly.predict(X_test_poly)
print(pred_poly[:10])

r2 = r2_score(y_test,pred_poly)
print("R2 score: ",r2)

mse = mean_squared_error(y_test,pred_poly)
print("MSE: ",mse)

mae = mean_absolute_error(y_test,pred_poly)
print("MAE: ",mae)

rmse = root_mean_squared_error(y_test,pred_poly)
print("RMSE: ",rmse)

lr = LinearRegression()
lr.fit(X_train,y_train)

pred = lr.predict(X_test)
print(pred[:10])

r21 = r2_score(y_test,pred)
print("R2 Score: ",r21)

mae1 = mean_absolute_error(y_test,pred)
print("MAE: ",mae1) 

mse1 = mean_squared_error(y_test,pred)
print("MSE: ",mse1)

rmse1 = root_mean_squared_error(y_test,pred)
print("RMSE: ",rmse1)