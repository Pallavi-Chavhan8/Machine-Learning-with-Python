import pandas as pd

data = pd.read_csv("C:/Users/HP/PycharmProjects/Machine-Learning-with-Python/iris.csv")

print(data.head())
print(data.isnull().sum())

print(data.info())   #info of the dataset

X = data.drop("target", axis = 1)
y = data["target"]

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
print("Features Preprocessed")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size= 0.2, train_size= 0.8,
                                                    random_state=0)
print("Data Split Successful")

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


print("Model Trained Successfully")

train_score = model.score(X_train, y_train)
print("Model Accuracy is ", train_score)


import numpy as np
new_sample = np.array([[6.2, 4.8, 2.6, 0.8]])
new_scaled = scalar.transform(new_sample)
prediction = model.predict(new_scaled)
print("Target = ", prediction[0])