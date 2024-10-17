import time
import joblib
import sys
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
import multiprocessing

lock = None


def split_data_into_equal_sets(X, y, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_sets = []
    test_sets = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_sets.append((X_train, y_train))
        test_sets.append((X_test, y_test))
        print("train: ", len(train_index))
        print("test: ", len(test_index))
    return train_sets, test_sets


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
            f.write(f"{classes[i]:<4} " +
                    " ".join(f"{count:^7}" for count in row) + "\n")


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


def run_svm(kernel, C, gamma, train_sets, test_sets, classes, class_colors,
            large_image_path, img_out_dir, metrics_output_file):
    print(f"\nTraining SVM with kernel='{kernel}' and C={C:0.2f} and gamma={gamma:.2f}")
    unix_time = int(time.time())
    new_dir_name = f"{unix_time}_kernel_{kernel}_C_{C:0.2f}_gamma_{gamma:.2f}"
    new_dir_path = os.path.join(img_out_dir, new_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)

    i = 1
    total = len(train_sets)
    for train_set, test_set in zip(train_sets, test_sets):
        print(f"\n{i}/{total} TRAINING kernel='{kernel}' and C={C:0.2f} and gamma={gamma:.2f}")
        x_train, y_train = train_set[0], train_set[1]
        x_test, y_test = test_set[0], test_set[1]

        if gamma != 0:
            svm_clf = SVC(kernel=kernel, C=C, gamma=gamma, cache_size=500)
        else:
            svm_clf = SVC(kernel=kernel, C=C, cache_size=500)

        svm_clf.fit(x_train, y_train)
        y_pred = svm_clf.predict(x_test)

        metrics = calculate_metrics(y_test, y_pred)

        output_str = ""
        for cls, metric in metrics.items():
            output_str += (f"{kernel};{C:0.2f};{gamma:.2f};{i};{cls};{metric['precision']:.4f};"
                           f"{metric['recall']:.4f};{metric['f1_score']:.4f};"
                           f"{metric['accuracy']:.4f}\n")

        with lock:
            print(f"\n{i}/{total} Printing to File kernel='{kernel}' and C={C:0.2f} and gamma={gamma:.2f}")
            with open(metrics_output_file, 'a') as f:
                f.write(output_str)

        print(f"\n{i}/{total} Printing CM to File kernel='{kernel}' and C={C:0.2f} and gamma={gamma:.2f}")

        cm = confusion_matrix(y_test, y_pred, list(classes.values()))
        cm_output_file = os.path.join(new_dir_path, f"cm_i-{i}_{kernel}-C_{C:0.2f}-gamma_{gamma:.2f}.txt")
        print_confusion_matrix_to_file(cm, list(classes.values()), cm_output_file)

        print(f"\n{i}/{total} Saving model kernel='{kernel}' and C={C:0.2f} and gamma={gamma:.2f}")

        model_output_file = os.path.join(
            new_dir_path, f"svm_model-i_{i}-kernel_{kernel}-C_{C:0.2f}_gamma_{gamma:.2f}.joblib")
        joblib.dump(svm_clf, model_output_file)

        print(f"\n{i}/{total} SVM model saved at {model_output_file}")

        # print(f"\nClassifing Image kernel='{kernel}' and C={C}")
        # classified_image = classify_image(large_image_path, svm_clf, class_colors)
        # classified_image.save(
        #     f"{img_out_dir}/classified_image-kernel_{kernel}-C_{C}.png")
        print(f"\nFinished kernel='{kernel}' and C={C} and gamma={gamma:.2f}")

        i += 1


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python script.py <image_dir> <large_image_path> <img_out_dir>")
        sys.exit(1)

    image_dir = sys.argv[1]
    large_image_path = sys.argv[2]
    img_out_dir = sys.argv[3]

    classes = {'vaca': 0, 'cielo': 1, 'pasto': 2}
    class_colors = {
        0: [255, 0, 0],     # Red
        1: [0, 0, 255],     # Blue
        2: [0, 255, 0]      # Green
    }

    reduction_percent = 100
    X, y = load_pixels_as_data(image_dir, classes, reduction_percent)

    n_splits = 5
    train_sets, test_sets = split_data_into_equal_sets(X, y, n_splits)

    # Different kernels and C values to experiment with
    # kernels = ['sigmoid']
    kernels = ['poly', 'rbf']
    C_values = [10]
    # kernels = ['linear', 'poly', 'rbf']
    # C_values = [i * 0.1 for i in range(1, 11)]
    gamma_values = [0]
    print(C_values)

    metrics_output_file = os.path.join(img_out_dir, "metrics.csv")
    output_str = "kernel;c_value;gamma;iteration;class;precision;recall;f1;accuracy\n"

    # Check if the file exists
    if not os.path.exists(metrics_output_file):
        # If it doesn't exist, write the header to the file
        with open(metrics_output_file, 'w') as f:
            f.write(output_str)

    lock = multiprocessing.Lock()  # Create a multiprocessing lock

    # Create a list of arguments for each SVM run
    jobs = [(kernel, C, gamma, train_sets, test_sets, classes, class_colors,
             large_image_path, img_out_dir, metrics_output_file) for kernel in kernels for C in C_values for gamma in gamma_values]

    # Use multiprocessing to parallelize the SVM training and evaluation
    with multiprocessing.Pool(processes=3) as pool:
        pool.starmap(run_svm, jobs)
