import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
from gap_statistic import OptimalK

# Load data
customer_data = pd.read_csv("C:/Users/Downloads/Mall_Customers.csv")

# Data Cleaning
customer_data.dropna()  # Drop any missing values
# Handle outliers if needed

# Customer Gender Visualization
gender_counts = customer_data['Genre'].value_counts()
plt.bar(gender_counts.index, gender_counts.values, color=['blue', 'red'])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Comparison')
plt.show()

# Age Distribution Visualization
plt.hist(customer_data['Age'], color='blue', bins=10)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

# Annual Income Visualization
plt.hist(customer_data['Annual Income (k$)'], color='orange', bins=10)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')
plt.title('Annual Income Distribution')
plt.show()

# Spending Score Visualization
plt.hist(customer_data['Spending Score (1-100)'], color='green', bins=10)
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frequency')
plt.title('Spending Score Distribution')
plt.show()

# K-means Algorithm
X = customer_data.iloc[:, 2:].values  # Extracting relevant columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scaling the data
wcss = []  # Within-cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Average Silhouette Method
range_n_clusters = range(2, 11)
for n_clusters in range_n_clusters:
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(7, 5)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X_scaled) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("Silhouette plot for K = %d" % n_clusters)
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()

# Gap Statistics
optimalK = OptimalK(parallel_backend='multiprocessing')
n_clusters = optimalK(X_scaled, cluster_array=np.arange(1, 11))
print('Optimal clusters:', n_clusters)

# K-means Clustering
kmeans = KMeans(n_clusters=6, random_state=0)
kmeans.fit(X_scaled)
customer_data['Cluster'] = kmeans.labels_

# Visualizing Clustering Results
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_scaled)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
finalDf = pd.concat([principalDf, customer_data['Cluster']], axis=1)

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=finalDf, palette='viridis')
plt.title('Clustering of Mall Customers')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
