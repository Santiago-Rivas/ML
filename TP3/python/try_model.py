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


def classify_image(image_path, svm_clf, class_colors):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img_data = np.array(img)

    pixels = img_data.reshape(-1, 3)
    pixel_classes = svm_clf.predict(pixels)
    colored_img_data = np.zeros_like(pixels)
    for i, pixel_class in enumerate(pixel_classes):
        colored_img_data[i] = class_colors[pixel_class]
    colored_img_data = colored_img_data.reshape(img_data.shape)
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

classified_image = classify_image(image_path, svm_clf, class_colors)
classified_image.save(f"{image_out_dir}/classified_image.png")
