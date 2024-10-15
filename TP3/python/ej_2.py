import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import multiprocessing


def calculate_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    metrics = {}

    for cls in classes:
        # True positives, false positives, false negatives, true negatives
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        tn = np.sum((y_true != cls) & (y_pred != cls))

        # Accuracy
        accuracy = (tp + tn) / (tp + fp + fn + tn)

        # Precision, Recall, F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision +
                                               recall) if (precision + recall) > 0 else 0

        # Store metrics for the class
        metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy
        }

    return metrics


def confusion_matrix(y_true, y_pred, classes):
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


def print_confusion_matrix(matrix, classes):
    print("\nConfusion Matrix:")
    print("    " + " ".join(f"{cls:^7}" for cls in classes))
    for i, row in enumerate(matrix):
        print(f"{classes[i]:<4} " + " ".join(f"{count:^7}" for count in row))


def print_confusion_matrix_to_file(matrix, classes, output_file):
    with open(output_file, 'a') as f:
        f.write("\nConfusion Matrix:\n")
        f.write("    " + " ".join(f"{cls:^7}" for cls in classes) + "\n")
        for i, row in enumerate(matrix):
            f.write(f"{classes[i]:<4} " + " ".join(f"{count:^7}" for count in row) + "\n")


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

            # Calculate number of pixels to keep
            num_pixels = pixels.shape[0]
            num_pixels_to_keep = int(num_pixels * (reduction_percent / 100))
            print(num_pixels)
            print(num_pixels_to_keep)

            # Randomly select a subset of pixels
            selected_indices = np.random.choice(num_pixels, num_pixels_to_keep, replace=False)
            X.append(pixels[selected_indices])
            y.append(np.full(selected_indices.shape[0], class_num))

    X = np.vstack(X)
    y = np.hstack(y)

    return X, y


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


def run_svm(kernel, C, X_train, X_test, y_train, y_test, classes, class_colors, large_image_path, img_out_dir):
    print(f"\nTraining SVM with kernel='{kernel}' and C={C}")

    # Create an SVM classifier with the current kernel and C value
    svm_clf = SVC(kernel=kernel, C=C, cache_size=500)

    # Train the classifier
    svm_clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm_clf.predict(X_test)

    # Calculate and display metrics
    metrics = calculate_metrics(y_test, y_pred)
    # Prepare output string for metrics
    output_str = f"Kernel: {kernel}, C: {C}\n"
    for cls, metric in metrics.items():
        output_str += (f"Class {cls}: Precision={metric['precision']:.4f}, "
                       f"Recall={metric['recall']:.4f}, F1-score={metric['f1_score']:.4f}, "
                       f"Accuracy={metric['accuracy']:.4f}\n")

    # Write metrics to a file
    metrics_output_file = f"{img_out_dir}/metrics-kernel_{kernel}-C_{C}.txt"
    print(f"\nPrinting to File kernel='{kernel}' and C={C}")
    cm = confusion_matrix(y_test, y_pred, list(classes.values()))
    with open(metrics_output_file, 'w') as f:
        f.write(output_str)
    print_confusion_matrix_to_file(cm, list(classes.values()), metrics_output_file)

    print(f"\nClassifing Image kernel='{kernel}' and C={C}")
    # Classify all the pixels of the large image
    classified_image = classify_image(large_image_path, svm_clf, class_colors)

    # Save or display the classified image
    classified_image.save(
        f"{img_out_dir}/classified_image-kernel_{kernel}-C_{C}.png")
    print(f"\nFinished kernel='{kernel}' and C={C}")


if __name__ == '__main__':
    image_dir = 'images/train'
    classes = {'vaca': 0, 'cielo': 1, 'pasto': 2}
    class_colors = {
        0: [255, 0, 0],     # Red
        1: [0, 0, 255],     # Blue
        2: [0, 255, 0]      # Green
    }
    large_image_path = 'images/cow.jpg'
    img_out_dir = "images/out/"

    # Reduce dataset if it is too big
    reduction_percent = 20
    X, y = load_pixels_as_data(image_dir, classes, reduction_percent)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    print(f"Training set shape (pixels, features): {X_train.shape}")
    print(f"Test set shape (pixels, features): {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")

    # Different kernels and C values to experiment with
    # kernels = ['sigmoid']
    kernels = ['linear', 'rbf', 'poly']
    C_values = [0.1, 1, 10]

    # Create a list of arguments for each SVM run
    jobs = [(kernel, C, X_train, X_test, y_train, y_test, classes, class_colors,
             large_image_path, img_out_dir) for kernel in kernels for C in C_values]

    # Use multiprocessing to parallelize the SVM training and evaluation
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(run_svm, jobs)
