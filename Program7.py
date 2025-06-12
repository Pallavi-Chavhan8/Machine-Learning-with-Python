import pandas as pd
from matplotlib.pyplot import title

data = pd.read_csv("C:/Users/HP/PycharmProjects/Machine-Learning-with-Python/ecommerce_customers.csv")

X = data.drop(columns=['CustomerID'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features Scaled")

print(data.head())
print(pd.DataFrame(X_scaled, columns=X.columns).head())

from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=3)
data['Clusters'] = model.fit_predict(X_scaled)

print("Clusters Count is :")
print(data['Clusters'].value_counts().sort_index())

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(0,6))
sns.scatterplot(x = X_pca[:,0], y = X_pca[:,1], hue = data['Clusters'], palette='Set1')
plt.title("Hierarchical Clustering")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title= "Clusters")
plt.grid(True)
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
linked = linkage(X_scaled, method="ward")
plt.figure(figsize=(10,6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts= False)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()
