import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from skimage import io
from skimage.transform import resize
from PIL import Image

# Function to load and preprocess images (supports .webp)


def load_images_from_directory(directory, image_size=(500, 500)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
            image_path = os.path.join(directory, filename)
            # Use Pillow to open .webp images
            image = Image.open(image_path)
            # Convert to numpy array
            image = np.array(image)
            # Resize the image to a consistent size
            image_resized = resize(image, image_size, anti_aliasing=True)
            images.append(image_resized)
    return np.array(images)


# Load images from the directory
directory_path = "./imagenes/wood"  # Change this to your directory path
image_size = (300, 300)
images = load_images_from_directory(directory_path, image_size=image_size)

# Flatten all images into a 2D array for clustering
pixel_data = images.reshape(-1, 3)

n_clusters = 5

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(pixel_data)
kmeans_labels = kmeans.predict(pixel_data)

# Fit GMM
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(pixel_data)
gmm_labels = gmm.predict(pixel_data)

# Load the final test image (including .webp support)
final_image_path = "./imagenes/rock_house.webp"
final_image = Image.open(final_image_path)
# Resize to match the training images
final_image_resized = resize(
    np.array(final_image), image_size, anti_aliasing=True)
final_pixel_data = final_image_resized.reshape(-1, 3)

# Predict with KMeans and GMM on the test image
kmeans_test_labels = kmeans.predict(final_pixel_data)
gmm_test_labels = gmm.predict(final_pixel_data)

# Visualize results
kmeans_segmented = kmeans.cluster_centers_[
    kmeans_test_labels].reshape(final_image_resized.shape)
gmm_segmented = gmm.means_[gmm_test_labels].reshape(final_image_resized.shape)

# Display the original and segmented images
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(final_image_resized)
plt.title("Original Test Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(kmeans_segmented)
plt.title("KMeans Segmentation")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gmm_segmented)
plt.title("GMM Segmentation")
plt.axis('off')

plt.tight_layout()
plt.show()
