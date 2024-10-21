# Modificar el código para que realice un análisis móvil de las áreas y asigne la etiqueta de la moda a cada píxel

import time
import joblib
import sys
import os
import numpy as np
from PIL import Image
from scipy import stats
from sklearn.svm import SVC
import argparse

from scipy import stats

def classify_image_with_moving_window(image_path, svm_clf, class_colors, square_size=1):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img_data = np.array(img)

    # Dimensiones de la imagen
    height, width, _ = img_data.shape

    # Crear un array vacío para almacenar los vectores de etiquetas por píxel
    pixel_labels = np.empty((height, width), dtype=object)
    for i in range(height):
        for j in range(width):
            pixel_labels[i, j] = []  # Inicializamos con listas vacías para cada píxel

    # Recorrer la imagen en bloques móviles de tamaño square_size x square_size
    for i in range(0, height - square_size + 1):
        print(f"Processing row {i} of {height - square_size + 1}")
        for j in range(0, width - square_size + 1):
            # Extraer el bloque de NxN
            block = img_data[i:i + square_size, j:j + square_size]
            flattened_block = block.flatten()  # Aplanar a [3 * square_size * square_size]

            # Predecir la etiqueta para el bloque
            block_class = svm_clf.predict([flattened_block])[0]
        
            # Agregar la etiqueta predicha a todos los píxeles del bloque
            for x in range(i, i + square_size):
                for y in range(j, j + square_size):
                    pixel_labels[x, y].append(block_class)


    # Crear una nueva imagen clasificada tomando la moda de las etiquetas de cada píxel
    final_img_data = np.zeros_like(img_data)

    '''
    for i in range(height):
        print(f"Processing row {i} of {height - square_size + 1}")
        for j in range(width):
            print(pixel_labels[i, j])
    '''
    for i in range(height):
        print(f"Processing row {i} of {height - square_size + 1}")
        for j in range(width):
            if pixel_labels[i, j]:  # Solo si hay etiquetas para este píxel
                # Calcular la moda (etiqueta más frecuente)
                #print(pixel_labels[i, j])
                pixel_class = stats.mode(pixel_labels[i, j])[0]
                final_img_data[i, j] = class_colors[pixel_class]
            else:
                # Si no hay etiquetas, podemos asignar un valor por defecto o dejarlo sin cambios
                final_img_data[i, j] = [0, 0, 0]  # Negros como valor predeterminado

    classified_img = Image.fromarray(final_img_data.astype('uint8'), 'RGB')
    return classified_img


# Parseo de argumentos del script
parser = argparse.ArgumentParser(description='Load a model and predict an image with moving window.')
parser.add_argument('model_file', type=str, help='Path to the saved model file')
parser.add_argument('image_path', type=str, help='Path to the image to predict')
parser.add_argument('image_out_dir', type=str, help='Path to the image to predict')

args = parser.parse_args()

image_path = args.image_path
image_out_dir = args.image_out_dir

# Cargar el modelo SVM preentrenado
svm_clf = joblib.load(args.model_file)

# Definir los colores para cada clase
classes = {'vaca': 0, 'cielo': 1, 'pasto': 2}
class_colors = {
    0: [255, 0, 0],     # Rojo para 'vaca'
    1: [0, 0, 255],     # Azul para 'cielo'
    2: [0, 255, 0]      # Verde para 'pasto'
}

# Clasificar la imagen con ventanas móviles
classified_image = classify_image_with_moving_window(image_path, svm_clf, class_colors, 40)

# Guardar la imagen clasificada
model_filename = os.path.basename(args.model_file)
model_name, _ = os.path.splitext(model_filename)
output_filename = f"{model_name}_moving_window.png"
classified_image.save(f"{image_out_dir}/{output_filename}")
