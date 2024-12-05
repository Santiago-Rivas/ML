from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


# Load the training image
image_path = "./imagenes/messi.jpeg"  # Change to your training image path
image = io.imread(image_path)
image = image / 255.0  # Normalize pixel values to [0, 1]

# Prepare the training image data for clustering
pixel_data = image.reshape(-1, 3)


# Number of clusters
n_clusters = 2

print("kmeans")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(pixel_data)
print("fin kmeans")

print("gauss")
gmm = GaussianMixture(n_components=n_clusters, covariance_type='tied', random_state=42)
gmm_labels = gmm.fit_predict(pixel_data)
print("fin gauss")

print(gmm_labels)


# Calculate Calinski-Harabasz Index
ch_kmeans = calinski_harabasz_score(pixel_data, kmeans_labels)
ch_gmm = calinski_harabasz_score(pixel_data, gmm_labels)

print(f"Calinski-Harabasz Index - KMeans: {ch_kmeans}, GMM: {ch_gmm}")

# Reconstruct segmented images
kmeans_segmented_image = kmeans.cluster_centers_[kmeans_labels].reshape(image.shape)
gmm_cluster_colors = np.zeros((n_clusters, 3))

for i in range(n_clusters):
    gmm_cluster_colors[i] = pixel_data[gmm_labels == i].mean(axis=0)

gmm_segmented_image = gmm_cluster_colors[gmm_labels].reshape(image.shape)

# Plot
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

# KMeans Segmentation
plt.subplot(1, 3, 2)
plt.title("KMeans Segmentation")
plt.imshow(kmeans_segmented_image)
plt.axis("off")

# GMM Segmentation
plt.subplot(1, 3, 3)
plt.title("GMM Segmentation")
plt.imshow(gmm_segmented_image)
plt.axis("off")

plt.tight_layout()
plt.show()
