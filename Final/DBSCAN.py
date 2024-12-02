import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 1. Cargar la imagen
image = cv2.imread('./imagenes/condor.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB
plt.imshow(image)
plt.title("Imagen Original")
plt.axis('off')
plt.show()

# 2. Reducir la resolución de la imagen
scale_percent = 20  # Porcentaje de reducción (puedes ajustarlo según tus necesidades)
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
small_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

plt.imshow(small_image)
plt.title(f"Imagen Reducida ({width}x{height})")
plt.axis('off')
plt.show()

# 3. Aplanar la imagen y normalizar los valores
pixels = small_image.reshape(-1, 3)  # Cada píxel es una fila con valores RGB
pixels_normalized = pixels / 255.0  # Normalizar a [0, 1]

# 4. Escalar los datos (opcional para mejorar el rendimiento de DBSCAN)
scaler = StandardScaler()
pixels_scaled = scaler.fit_transform(pixels_normalized)

# 5. Aplicar DBSCAN
eps = 0.3  # Distancia máxima entre puntos para considerarlos vecinos
min_samples = 50  # Mínimo número de puntos para formar un cluster
print(f"Aplicando DBSCAN con eps={eps}, min_samples={min_samples}...")
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(pixels_scaled)

# 6. Reconstruir la imagen
clustered_image = labels.reshape(small_image.shape[:2])  # Forma de la imagen reducida

# 7. Asignar colores a los clusters
unique_labels = np.unique(labels)
n_clusters = len(unique_labels[unique_labels >= 0])  # Excluye el ruido (-1)
print(f"Número de clusters encontrados (excluyendo ruido): {n_clusters}")

# Asignar colores aleatorios a cada cluster (y gris para ruido)
random_colors = np.random.rand(len(unique_labels), 3)
random_colors[labels == -1] = [0.5, 0.5, 0.5]  # Color para el ruido (gris)
colored_image = random_colors[labels + 1]  # labels puede ser -1, ajustamos con +1

# 8. Mostrar la imagen segmentada
plt.imshow(colored_image.reshape(small_image.shape))
plt.title(f"Imagen Segmentada con DBSCAN (eps={eps}, min_samples={min_samples})")
plt.axis('off')
plt.show()
