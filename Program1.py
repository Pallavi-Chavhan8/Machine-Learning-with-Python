import pandas as pd

data=pd.read_csv ("C:/Palla_ML/clean_study_hours_vs_marks.csv")
print(data.head(10))
X=data[["StudyHours"]]
Y=data["Marks"]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train, Y_train)
print("Model trained")
Y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error,r2_score
print("R square",r2_score(Y_test,Y_pred))
print("MSE :",mean_squared_error(Y_test,Y_pred))
import matplotlib.pyplot as plt
plt.scatter(X_test,Y_test,color='blue',label='Actual')
plt.plot(X_test,Y_pred,color='red',linewidth=2, label= 'Predicted')
plt.show()