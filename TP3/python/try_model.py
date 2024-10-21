import time
import joblib
import sys
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
import multiprocessing
import argparse


def classify_image(image_path, svm_clf, class_colors, square_size=1):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img_data = np.array(img)

    # Dimensiones de la imagen
    height, width, _ = img_data.shape

    # Lista para almacenar los bloques a clasificar
    blocks = []

    # Recorrer la imagen en bloques de tamaño square_size x square_size
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            block = img_data[i:i+square_size, j:j+square_size]
            if block.shape[0] == square_size and block.shape[1] == square_size:
                # Aplanar el bloque a un vector de tamaño 3 * square_size * square_size
                flattened_block = block.flatten()  # Aplana a [3 * square_size * square_size]
                blocks.append(flattened_block)

    # Convertir a numpy array
    blocks = np.array(blocks)

    # Clasificar los bloques usando el modelo SVM
    block_classes = svm_clf.predict(blocks)

    # Crear una nueva imagen clasificada
    colored_img_data = np.zeros_like(img_data)

    block_index = 0
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if block_index < len(blocks):
                pixel_class = block_classes[block_index]
                colored_img_data[i:i+square_size, j:j+square_size] = class_colors[pixel_class]
                block_index += 1

    classified_img = Image.fromarray(colored_img_data.astype('uint8'), 'RGB')
    return classified_img

parser = argparse.ArgumentParser(description='Load a model and predict an image.')
parser.add_argument('model_file', type=str, help='Path to the saved model file')
parser.add_argument('image_path', type=str, help='Path to the image to predict')
parser.add_argument('image_out_dir', type=str, help='Path to the image to predict')
args = parser.parse_args()

image_path = args.image_path
image_out_dir = args.image_out_dir

# Step 2: Load the model from the file
svm_clf = joblib.load(args.model_file)
classes = {'vaca': 0, 'cielo': 1, 'pasto': 2}
class_colors = {
    0: [255, 0, 0],     # Red
    1: [0, 0, 255],     # Blue
    2: [0, 255, 0]      # Green
}

classified_image = classify_image(image_path, svm_clf, class_colors, 10)

model_filename = os.path.basename(args.model_file)  # Get the model filename (e.g., svm_model_i_1_kernel_poly_C_1.00_gamma_scale_cache_500_degree_3.joblib)
model_name, _ = os.path.splitext(model_filename)  # Remove the file extension (.joblib)
output_filename = f"{model_name}.png"
classified_image.save(f"{image_out_dir}/{output_filename}")
