import pandas as pd
from scipy.cluster.vq import kmeans

data = pd.read_csv("C:/Users/HP/PycharmProjects/Machine-Learning-with-Python/Mall_Customers.csv")
print(data.head())

X = data[['Annual Income (k$)','Spending Score (1-100)']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features Scaled")

from sklearn.cluster import KMeans
inertia = []
for k in range(1,11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

kmeans = KMeans(n_clusters=2, random_state=42)
data['Clusters'] = kmeans.fit_predict(X_scaled)
Centroids = scaler.inverse_transform(kmeans.cluster_centers_)

print("\n Centroids ")
for i,c in enumerate(Centroids):
    print(f"Clusters{i} : Income = {c[0]}, Score = {c[1]}")

print("\n Clusters Counts:")
print(data['Clusters'].value_counts().sort_index())

import matplotlib.pyplot as plt
plt.scatter(X_scaled[:,0],X_scaled[:,1], c= data['Clusters'])
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c= 'black', marker = 'X')
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering")



