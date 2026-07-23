import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\medical_insurance.csv")

print(df.isnull().sum())

oe = OneHotEncoder(drop='first',sparse_output=False)
encoded_df = pd.DataFrame(oe.fit_transform(df[["sex","smoker","region"]]),columns=oe.get_feature_names_out(["sex","smoker","region"]),index=df.index)
df = df.drop(["sex","smoker","region"],axis=1)
new_df = pd.concat([encoded_df,df],axis=1)

plt.title("Correlation matrix")
sns.heatmap(new_df.corr(),annot=True,cmap="coolwarm")
plt.show()

X = new_df.drop("charges",axis=1)
y = new_df["charges"].values

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

dr = DecisionTreeRegressor(max_depth=10)
dr.fit(X_train,y_train)

pred = dr.predict(X_test)
print(pred)

r2 = r2_score(y_test,pred)
print("R2 Score: ",r2)

mae = mean_absolute_error(y_test,pred)
print("MAE: ",mae)

mse = mean_squared_error(y_test,pred)
print("MSE: ",mse)

rmse = root_mean_squared_error(y_test,pred)
print("RMSE: ",rmse)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": dr.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

plt.title("Medical Insurance Cost Prediction")
plot_tree(dr,feature_names=X.columns,rounded=True,filled=True)
plt.show()