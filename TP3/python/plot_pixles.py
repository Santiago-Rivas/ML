import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import json
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def load_pixels_as_data(image_dir, classes, reduction_percent=100):
    X = []
    y = []

    for label, class_num in classes.items():
        class_images = [f for f in os.listdir(image_dir) if label in f]

        for image_file in class_images:
            img_path = os.path.join(image_dir, image_file)
            img = Image.open(img_path)
            img = img.convert('RGB')

            img_data = np.array(img)
            pixels = img_data.reshape(-1, 3)
            print(pixels)

            # Calculate number of pixels to keep
            num_pixels = pixels.shape[0]
            num_pixels_to_keep = int(num_pixels * (reduction_percent / 100))
            print(num_pixels)
            print(num_pixels_to_keep)

            # Randomly select a subset of pixels
            selected_indices = np.random.choice(
                num_pixels, num_pixels_to_keep, replace=False)
            X.append(pixels[selected_indices])
            y.append(np.full(selected_indices.shape[0], class_num))

    X = np.vstack(X)
    y = np.hstack(y)

    return X, y


# Function to plot pixels in 3D with RGB values and specific colors for each class
def plot_rgb_pixels(X, y, percentage=2):
    # Calculate the number of points to sample
    num_samples = int(len(X) * (percentage / 100.0))

    # Randomly select indices
    random_indices = np.random.choice(len(X), num_samples, replace=False)

    # Subset X and y based on random indices
    X_subset = X[random_indices]
    y_subset = y[random_indices]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Extract R, G, B values from the subset of X
    R = X_subset[:, 0]
    G = X_subset[:, 1]
    B = X_subset[:, 2]

    # Class color mapping: 0 -> Red, 1 -> Blue, 2 -> Green
    class_colors = {
        0: [255, 0, 0],     # Red
        1: [0, 0, 255],     # Blue
        2: [0, 255, 0]      # Green
    }

    # Convert class_colors to normalized RGB values between 0 and 1
    normalized_class_colors = {k: np.array(
        v) / 255.0 for k, v in class_colors.items()}

    # Create a list of colors corresponding to the labels in y
    colors = np.array([normalized_class_colors[label] for label in y_subset])

    # Scatter plot using R, G, B as coordinates and class-specific colors
    scatter = ax.scatter(R, G, B, c=colors, s=50)

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.title(f'3D RGB Pixel Plot with Class Colors ({percentage}% Sample)')

    plt.show()

# Assuming X is the pixel RGB values and y contains labels (for now, colors)
# X = np.array([[R1, G1, B1], [R2, G2, B2], ...])
# y = array with labels (e.g., 0, 1, 2...)

# Call the plotting function


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_dir>")

        sys.exit(1)

    # CONSTANTES
    classes = {'vaca': 0, 'cielo': 1, 'pasto': 2}
    class_colors = {
        0: [255, 0, 0],     # Red
        1: [0, 0, 255],     # Blue
        2: [0, 255, 0]      # Green
    }

    # ARGUMENTOS
    image_dir = sys.argv[1]

    X, y = load_pixels_as_data(image_dir, classes, 100)
    plot_rgb_pixels(X, y)

    # Step 1: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)

    # Step 2: Standardize the data (optional but helps for SVM performance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 3: Train a linear SVM
    linear_svm = SVC(kernel='linear')
    linear_svm.fit(X_train_scaled, y_train)

    # Step 4: Make predictions
    y_pred = linear_svm.predict(X_test_scaled)

    # Step 5: Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of Linear SVM: {accuracy:.2f}")

    # Check if the accuracy is 100%
    if accuracy == 1.0:
        print("The labels are linearly separable.")
    else:
        print("The labels are not linearly separable.")
