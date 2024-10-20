import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_image_colors_barchart(image_paths, predominant_colors):
    # Crear una figura para mostrar los gráficos
    plt.figure(figsize=(15, 5))

    # Definir los colores para los canales RGB
    colors = ['Red', 'Green', 'Blue']

    # Lista para almacenar los promedios por imagen
    avg_colors_all_images = []

    # Calcular el promedio de los valores RGB para cada imagen
    for img_path in image_paths:
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        avg_color_per_channel = np.mean(image_rgb, axis=(0, 1))
        avg_colors_all_images.append(avg_color_per_channel)

    # Encontrar el valor máximo global entre todos los canales y todas las imágenes
    max_rgb_value = max([max(avg_colors) for avg_colors in avg_colors_all_images])

    # Crear gráficos de barras para cada imagen
    for i, avg_color_per_channel in enumerate(avg_colors_all_images):
        plt.subplot(1, len(image_paths), i + 1)
        plt.title(f'{predominant_colors[i]}')

        # Crear gráfico de barras para los canales RGB
        plt.bar(colors, avg_color_per_channel, color=['r', 'g', 'b'])

        # Establecer el mismo límite en el eje y para todos los gráficos
        plt.ylim(0, 255)

    # Mostrar los gráficos
    plt.tight_layout()
    plt.show()

# Lista de rutas de las imágenes
image_paths = ['./images/train/cielo.jpg', './images/train/pasto.jpg', './images/train/vaca.jpg']
# Lista de colores predominantes para cada imagen
predominant_colors = ['Cielo', 'Pasto', 'Vaca']

# Llamar a la función
analyze_image_colors_barchart(image_paths, predominant_colors)
