import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\Fish_dataset.csv")

print(df.isnull().sum())

oe = OneHotEncoder(drop="first",sparse_output=False)
encoded_df = pd.DataFrame(oe.fit_transform(df[["Species"]]),columns=oe.get_feature_names_out(["Species"]),index=df.index)
df = df.drop("Species",axis=1)
new_df = pd.concat([encoded_df,df],axis=1)

plt.title("Correlation Matrix")
sns.heatmap(new_df.corr(),annot=True,cmap="coolwarm")
plt.show()

X = new_df.drop("Weight",axis=1)
y = new_df["Weight"].values

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kr = KNeighborsRegressor(n_neighbors=3,metric="manhattan")
kr.fit(X_train_scaled,y_train)

pred = kr.predict(X_test_scaled)
print(pred[:10])

r2 = r2_score(y_test,pred)
print("R2 Score: ",r2)

mse = mean_squared_error(y_test,pred)
print("MSE: ",mse)

mae = mean_absolute_error(y_test,pred)
print("MAE: ",mae)

rmse = root_mean_squared_error(y_test,pred)
print("RMSE: ",rmse)