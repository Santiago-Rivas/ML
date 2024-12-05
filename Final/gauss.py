import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from skimage import io

# Load the training image
image_path = "./imagenes/messi.jpeg"  # Change to your training image path
image = io.imread(image_path)
image = image / 255.0  # Normalize pixel values to [0, 1]

# Prepare the training image data for clustering
pixel_data = image.reshape(-1, 3)

# Load the test image
new_image_path = "./imagenes/perito.jpeg"  # Change to your test image path
new_image = io.imread(new_image_path)
new_image = new_image / 255.0  # Normalize pixel values to [0, 1]
new_pixel_data = new_image.reshape(-1, 3)

# Configure the Gaussian Mixture Model
n_clusters = 5
gmm = GaussianMixture(n_components=n_clusters, covariance_type='tied', random_state=42)

# Train the model on the training image
gmm.fit(pixel_data)

# Predict clusters for both images
clusters = gmm.predict(pixel_data)  # Clusters for the training image
new_clusters = gmm.predict(new_pixel_data)  # Clusters for the test image

# Compute representative colors from the training image
cluster_colors = np.zeros((n_clusters, 3))
for i in range(n_clusters):
    cluster_colors[i] = pixel_data[clusters == i].mean(axis=0)

# Reconstruct segmented images using the cluster colors
segmented_image = cluster_colors[clusters].reshape(image.shape)
new_segmented_image = cluster_colors[new_clusters].reshape(new_image.shape)

# Visualize the original and segmented images
plt.figure(figsize=(10, 5))

# Training image
plt.subplot(2, 2, 1)
plt.title("Training Image (Original)")
plt.imshow(image)
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title(f"Training Image (Segmented, {n_clusters} clusters)")
plt.imshow(segmented_image)
plt.axis("off")

# Test image
plt.subplot(2, 2, 3)
plt.title("Test Image (Original)")
plt.imshow(new_image)
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title(f"Test Image (Segmented, {n_clusters} clusters)")
plt.imshow(new_segmented_image)
plt.axis("off")

plt.tight_layout()
plt.show()
