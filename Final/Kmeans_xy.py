import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Cargar la imagen
image = cv2.imread('./imagenes/condor.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB
plt.imshow(image)
plt.title("Imagen Original")
plt.axis('off')
plt.show()

# 2. Reducir la resolución de la imagen
scale_percent = 100  # Porcentaje de reducción (ajusta este valor según tus necesidades)
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
small_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

plt.imshow(small_image)
plt.title(f"Imagen Reducida ({width}x{height})")
plt.axis('off')
plt.show()

# 3. Aplanar la imagen, normalizar los valores y agregar coordenadas
h, w, _ = small_image.shape
pixels = small_image.reshape(-1, 3)  # Cada píxel es una fila con valores RGB
pixels_normalized = pixels / 255.0  # Normalizar a [0, 1]

# Crear una matriz con las posiciones de los píxeles
x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
positions = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)

# Normalizar las posiciones para que tengan el mismo rango que los colores
positions_normalized = positions / np.array([w, h])

# Concatenar colores y posiciones
features = np.hstack([pixels_normalized, positions_normalized])

# 4. Aplicar K-means Clustering
num_clusters = 7  # Número de clusters deseados
print(f"Aplicando K-means con {num_clusters} clusters...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(features)

# 5. Reconstruir la imagen
clustered_image = labels.reshape(small_image.shape[:2])  # Forma de la imagen reducida

# 6. Asignar colores a los clusters
colors = kmeans.cluster_centers_[:, :3]  # Solo usamos los valores RGB
colored_image = colors[clustered_image]

# 7. Mostrar la imagen segmentada
plt.imshow(colored_image)
plt.title(f"Imagen Clusterizada con K-means ({num_clusters} clusters)")
plt.axis('off')
plt.show()
