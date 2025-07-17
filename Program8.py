import pandas as pd

data = pd.read_csv("C:/Users/HP/Desktop/CSV FILES/give_me_credit.csv")
data = data.dropna()

X = data.drop('SeriousDlqin2yrs', axis=1)
y = data['SeriousDlqin2yrs']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=52)

from sklearn.svm import SVC
model = SVC( kernel='rbf',C = 1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score
print("Accuracy : ", accuracy_score(y_test, y_pred))
print("\n Classification Report : \n", classification_report(y_test, y_pred))

correct = X_test[y_test == y_pred]
wrong = X_test[y_test != y_pred]

print("\n Top 5 Correct Predictions : ")
print(correct.head())

print("\n Top 5 Wrong Predictions : ")
print(wrong.head())
