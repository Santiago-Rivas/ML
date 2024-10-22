import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def get_rgb_values(image_path):
    image = Image.open(image_path)
    rgb_image = image.convert("RGB")  # Convertir la imagen a RGB
    np_image = np.array(rgb_image)    # Convertir la imagen a un array NumPy
    return np_image  # Devolver los valores RGB de todos los píxeles



def plot_histogram(image_array, title):
    r = image_array[:, :, 0].flatten()
    g = image_array[:, :, 1].flatten()
    b = image_array[:, :, 2].flatten()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.hist(r, bins=256, color='red', alpha=0.7)
    plt.title(f'Red Channel - {title}')

    plt.subplot(1, 3, 2)
    plt.hist(g, bins=256, color='green', alpha=0.7)
    plt.title(f'Green Channel - {title}')

    plt.subplot(1, 3, 3)
    plt.hist(b, bins=256, color='blue', alpha=0.7)
    plt.title(f'Blue Channel - {title}')

    plt.tight_layout()
    plt.show()

# Extraer valores de tres imágenes
rgb_image1 = get_rgb_values("./images/train/cielo.jpg")
rgb_image2 = get_rgb_values("./images/train/pasto.jpg")
rgb_image3 = get_rgb_values("./images/train/vaca.jpg")

# Generar histogramas para cada imagen
plot_histogram(rgb_image1, "Imagen 1")
plot_histogram(rgb_image2, "Imagen 2")
plot_histogram(rgb_image3, "Imagen 3")