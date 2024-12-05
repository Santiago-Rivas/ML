import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from skimage import io

# Cargar la imagen
image_path = "./imagenes/cow.jpg"  # Cambia esto por la ruta de tu imagen
image = io.imread(image_path)
image = image / 255.0  # Normalizar los valores de píxeles a [0, 1]

# Preparar los datos de la imagen para clustering
pixel_data = image.reshape(-1, 3)

# Configurar el modelo Gaussian Mixture
n_clusters = 10  # Número de segmentos deseados
gmm = GaussianMixture(n_components=n_clusters, covariance_type='tied', random_state=42)

# Ajustar el modelo y predecir los clusters
gmm.fit(pixel_data)
clusters = gmm.predict(pixel_data)

# Calcular los colores promedio para cada cluster
cluster_colors = np.zeros((n_clusters, 3))
for i in range(n_clusters):
    cluster_colors[i] = pixel_data[clusters == i].mean(axis=0)

# Reconstruir la imagen segmentada con colores representativos
segmented_image = cluster_colors[clusters].reshape(image.shape)

# Visualizar la imagen original y la segmentada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Imagen Segmentada (GMM, {n_clusters} clusters)")
plt.imshow(segmented_image)
plt.axis("off")

plt.tight_layout()
plt.show()
