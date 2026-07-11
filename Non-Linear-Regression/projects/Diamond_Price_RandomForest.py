import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\Diamonds Prices2022.csv")

print(df.isnull().sum())

oe = OneHotEncoder(drop='first',sparse_output=False)
encoded_df = pd.DataFrame(oe.fit_transform(df[["cut","color","clarity"]]),columns=oe.get_feature_names_out(["cut","color","clarity"]),index=df.index)
df = df.drop(["srno","cut","color","clarity"],axis=1)
new_df = pd.concat([encoded_df,df],axis=1)

plt.title("Correlation Matrix")
sns.heatmap(new_df.corr(),annot=True,cmap="coolwarm")
plt.show()

X = new_df.drop("price",axis=1)
y = new_df["price"].values

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

rf = RandomForestRegressor(n_estimators=100,max_depth=15)
rf.fit(X_train,y_train)

pred = rf.predict(X_test)
print(pred[:10])

r2 = r2_score(y_test,pred)
print("R2 Score: ",r2)

mse = mean_squared_error(y_test,pred)
print("MSE: ",mse)

mae = mean_absolute_error(y_test,pred)
print("MAE: ",mae)

rmse = root_mean_squared_error(y_test,pred)
print("RMSE: ",rmse)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)