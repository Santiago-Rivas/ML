import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb

import concurrent.futures
from threading import Lock


def calculate_segment_mse(image, segments):
    unique_labels = np.unique(segments)
    total_mse = 0
    num_segments = len(unique_labels)
    for label in unique_labels:
        # Get the mask for this segment
        mask = segments == label
        segment_pixels = image[mask]

        # Calculate the mean color of the segment
        mean_color = np.mean(segment_pixels, axis=0)

        # Calculate squared error for each pixel in the segment
        mse_segment = np.mean(
            np.sum((segment_pixels - mean_color) ** 2, axis=1))

        total_mse += mse_segment

    # Calculate average MSE across all segments
    average_mse = total_mse / num_segments
    return average_mse


# 1. Cargar la imagen
image = cv2.imread('./imagenes/condor.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB

# 2. Aplicar SLIC para segmentación en superpíxeles
# Número de superpíxeles deseados

# Lock to ensure print statements are not mixed
print_lock = Lock()


def process_segmentation(num_segments, compactness, max_num_iter, sigma, convert2lab):
    # Apply SLIC segmentation
    segments = slic(
        image,
        n_segments=num_segments,
        compactness=compactness,
        enforce_connectivity=False,
        max_num_iter=max_num_iter,
        sigma=sigma,
        convert2lab=convert2lab,
        start_label=1
    )

    # Calculate the MSE
    mse = calculate_segment_mse(image, segments)

    # Print result with synchronized access to print
    with print_lock:
        print(f"{num_segments},{compactness},{
              max_num_iter},{sigma},{convert2lab},{mse}")


# Lists of parameters
num_segments_arr = [1000]
compactness_arr = [1, 10, 100, 1000]
max_num_iter_arr = [10, 20, 50]
sigma_arr = [0.5, 1.0, 2.0]
spacing_arr = [5, 10]
convert2lab_arr = [True, False]

# Create a list to store the parameter combinations
param_combinations = [
    (num_segments, compactness, max_num_iter, sigma, convert2lab)
    for num_segments in num_segments_arr
    for compactness in compactness_arr
    for max_num_iter in max_num_iter_arr
    for sigma in sigma_arr
    for convert2lab in convert2lab_arr
]

# Using ThreadPoolExecutor to run in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Map the function to the parameter combinations
    executor.map(lambda params: process_segmentation(
        *params), param_combinations)

# 3. Visualizar superpíxeles como colores promedio
# Asignar colores promedio a cada superpíxel
# segmented_image = label2rgb(segments, image, kind='avg')

# 4. Mostrar la imagen segmentada
# plt.figure(figsize=(10, 10))
# plt.imshow(segmented_image)
# plt.title(f"Segmentación con SLIC ({num_segments} superpíxeles)")
# plt.axis('off')
# plt.show()
