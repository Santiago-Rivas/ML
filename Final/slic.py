import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb

# 1. Cargar la imagen
image = cv2.imread('./imagenes/condor.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB

# 2. Aplicar SLIC para segmentación en superpíxeles
num_segments = 7  # Número de superpíxeles deseados
segments = slic(image, n_segments=num_segments, compactness=10, enforce_connectivity=False, start_label=1)

# 3. Visualizar superpíxeles como colores promedio
segmented_image = label2rgb(segments, image, kind='avg')  # Asignar colores promedio a cada superpíxel

# 4. Mostrar la imagen segmentada
plt.figure(figsize=(10, 10))
plt.imshow(segmented_image)
plt.title(f"Segmentación con SLIC ({num_segments} superpíxeles)")
plt.axis('off')
plt.show()
