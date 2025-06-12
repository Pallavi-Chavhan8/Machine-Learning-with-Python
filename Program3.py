import pandas as pd

data = pd.read_csv("C:/Users/HP/PycharmProjects/Machine-Learning-with-Python/PlayTennis.csv")

print(data.head())
print(data.isnull().sum())
print(data.info())
print(data.describe())

#data preprocessing
for col in data.columns[:-1]:
    data[col] = data[col].astype('Category')
    mapping = dict(enumerate(data[col].cat.categories))
    print(f"{col}: {mapping}")
    data[col] = data[col].cat.codes
    print("Categorical to numerical conversion Successful")

    target = "Play Tennis"
    data[target] = data[target].map({'yes': 1, 'No': 0})
    print("Target Converted Successfully")

    from sklearn.model_selection import train_test_split
    x = data.drop(target, axis=1)
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8,
                                                        random_state=58)
    from sklearn.tree import  DecisionTreeClassifier

    model = DecisionTreeClassifier(criterion='entropy')

    print("\n Train Accuracy is ",model.score(x_train,y_train))
    print("The Accuracy for Testing ")