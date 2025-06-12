import pandas as pd

data = pd.read_csv("C:/Users/HP/PycharmProjects/Machine-Learning-with-Python/iris_naivebayes.csv")
print(data.head())
print(data.isnull().sum())
print(data.info())
print(data.describe())

X = data.drop("target", axis = 1)
y = data["target"]

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

print("Features Preprocessed")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size= 0.2, train_size= 0.8,
                                                    random_state=58)
print("Data Split Successfully")
from sklearn.naive_bayes import GaussianNB
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)
print("Model Trained Successfully")

train_accuracy = naive_bayes_model.score(X_train, y_train)
test_accuracy = naive_bayes_model.score(X_test, y_test)
print("Train Accuracy : ",train_accuracy)
print("Test Accuracy : ",test_accuracy)

y_pred = naive_bayes_model.predict(X_test)

correct_predictions = (y_pred == y_test)
wrong_predictions = (y_pred != y_test)

print("Correct Predictions : ")
print(X_test[correct_predictions])


print("Wrong Predictions : ")
print(X_test[wrong_predictions])