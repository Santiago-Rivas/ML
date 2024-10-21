import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_rgb_values_from_images(image_folder):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Usar la paleta 'tab20' para colores distintivos
    cmap = plt.get_cmap('tab20')
    num_images = len([f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))])
    colors = [cmap(i % 20) for i in range(num_images)]  # Ajustar si hay más de 20 imágenes

    # Recorrer cada imagen en el folder
    for idx, image_file in enumerate(os.listdir(image_folder)):
        if image_file.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            image_path = os.path.join(image_folder, image_file)
            img = Image.open(image_path)
            img = img.convert('RGB')
            img_data = np.array(img)
            
            # Extraer los valores RGB
            r = img_data[:, :, 0].flatten()
            g = img_data[:, :, 1].flatten()
            b = img_data[:, :, 2].flatten()

            # Plotear los valores RGB en el gráfico 3D
            ax.scatter(r, g, b, color=colors[idx], label=image_file, alpha=0.6)

    # Etiquetas y configuración del gráfico
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.legend(loc='best')
    plt.show()

# Llamar a la función pasando la carpeta de imágenes
plot_rgb_values_from_images('./images/train')
