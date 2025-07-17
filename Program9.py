import pandas as pd

df = pd.read_csv("C:/Users/HP/Desktop/CSV FILES/iris_naivebayes (2).csv")

X = df.drop(columns=['target'])
y = df['target']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.8, random_state=58)
print("Data Split Successful")

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=15, random_state=42)
rf_model.fit(X_train, y_train)

sample_index = 10
sample = X_test.iloc[sample_index].values.reshape(1, -1)

tree_preds = [tree.predict(sample)[0] for tree in rf_model.estimators_]

from collections import Counter

vote_counts = Counter(tree_preds)
label_votes = {le.inverse_transform([int(k)])[0]: v for k, v in vote_counts.items()}

print("\n Class Votes : ")
for label, count in label_votes.items():
    print(f"{label}: {count} vote(s)")

majority_encoded, _ = vote_counts.most_common(1)[0]
majority_label = le.inverse_transform([int(majority_encoded)])[0]

true_label = le.inverse_transform([int(y_test[sample_index])])[0]

print(f"\n Final Prediction (Majority Vote) : {majority_label}")
print(f"Actual Label : {true_label}")