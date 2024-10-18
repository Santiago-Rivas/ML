import time
import joblib
import sys
import json
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
import multiprocessing

lock = None
lock_final = None

class linear_svc:
    def __init__(self, kernel, c, cache_size):
        self.kernel = kernel
        self.c = c
        self.cache_size = cache_size
    
    def dir_name_string(self):
        return f"_kernel_{self.kernel}_C_{self.c:0.2f}_cache_{self.cache_size}"
    
    def train(self, x_train, y_train):
        self.svm_clf = SVC(kernel=self.kernel, C=self.c, cache_size=self.cache_size)
        self.svm_clf.fit(x_train, y_train)

    def predict(self, x_test):
        return self.svm_clf.predict(x_test)
    
    def properties(self):
        return f"kernel='{self.kernel}' and C={self.c:0.2f}"
    
    def csv_properties(self):
        return f"{self.kernel};{self.c:0.2f};0;0"
    
    def get_model(self):
        return self.svm_clf

class poly_svc:
    def __init__(self, kernel, c, gamma, degree, cache_size):
        self.kernel = kernel
        self.c = c
        self.gamma = gamma
        self.degree = degree
        self.cache_size = cache_size
    
    def dir_name_string(self):
        return f"_kernel_{self.kernel}_C_{self.c:0.2f}_gamma_{self.gamma}_cache_{self.cache_size}_degree_{self.degree}"
    
    def train(self, x_train, y_train):
        self.svm_clf = SVC(kernel=self.kernel, C=self.c, gamma=self.gamma, degree=self.degree, cache_size=self.cache_size)
        self.svm_clf.fit(x_train, y_train)

    def predict(self, x_test):
        return self.svm_clf.predict(x_test)

    def properties(self):
        return f"kernel='{self.kernel}' and C={self.c:0.2f} and gamma={self.gamma} and degree={self.degree}"
    
    def csv_properties(self):
        return f"{self.kernel};{self.c:0.2f};{self.gamma};{self.degree}"
    
    def get_model(self):
        return self.svm_clf

class sigmoid_svc:
    def __init__(self, kernel, c, gamma, cache_size):
        self.kernel = kernel
        self.c = c
        self.gamma = gamma
        self.cache_size = cache_size
    
    def dir_name_string(self):
        return f"_kernel_{self.kernel}_C_{self.c:0.2f}_cache_{self.cache_size}_gamma_{self.gamma}"
    
    def train(self, x_train, y_train):
        self.svm_clf = SVC(kernel=self.kernel, C=self.c, gamma=self.gamma ,cache_size=self.cache_size)
        self.svm_clf.fit(x_train, y_train)

    def predict(self, x_test):
        return self.svm_clf.predict(x_test)
    
    def properties(self):
        return f"kernel='{self.kernel}' and C={self.c:0.2f} and gamma={self.gamma}"
    
    def csv_properties(self):
        return f"{self.kernel};{self.c:0.2f};{self.gamma};0"
    
    def get_model(self):
        return self.svm_clf

class rbf_svc:
    def __init__(self, kernel, c, gamma, cache_size):
        self.kernel = kernel
        self.c = c
        self.gamma = gamma
        self.cache_size = cache_size
    
    def dir_name_string(self):
        return f"_kernel_{self.kernel}_C_{self.c:0.2f}_cache_{self.cache_size}_gamma_{self.gamma}"
    
    def train(self, x_train, y_train):
        self.svm_clf = SVC(kernel=self.kernel, C=self.c, gamma=self.gamma ,cache_size=self.cache_size)
        self.svm_clf.fit(x_train, y_train)

    def predict(self, x_test):
        return self.svm_clf.predict(x_test)
    
    def properties(self):
        return f"kernel='{self.kernel}' and C={self.c:0.2f} and gamma={self.gamma}"
    
    def csv_properties(self):
        return f"{self.kernel};{self.c:0.2f};{self.gamma};0"
    
    def get_model(self):
        return self.svm_clf
    

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


def run_svm(svc_model, train_sets, test_sets, classes, class_colors,
            large_image_path, img_out_dir, metrics_output_file):
    print(f"\nTraining SVM with {svc_model.properties()}")
    unix_time = int(time.time())
    new_dir_name = f"{unix_time}{svc_model.dir_name_string()}"
    new_dir_path = os.path.join(img_out_dir, new_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)

    i = 1
    total = len(train_sets)
    for train_set, test_set in zip(train_sets, test_sets):
        print(f"\n{i}/{total} TRAINING SVM with {svc_model.properties()}")
        x_train, y_train = train_set[0], train_set[1]
        x_test, y_test = test_set[0], test_set[1]

        svc_model.train(x_train, y_train)
        y_pred = svc_model.predict(x_test)

        metrics = calculate_metrics(y_test, y_pred)

        output_str = ""
        for cls, metric in metrics.items():
            output_str += (f"{svc_model.csv_properties()};{i};{cls};"
                           f"{metric['precision']:.4f};{metric['recall']:.4f};{metric['f1_score']:.4f};{metric['accuracy']:.4f}\n")

        with lock:
            print(f"\n{i}/{total} Printing to File {svc_model.properties()}")
            with open(metrics_output_file, 'a') as f:
                f.write(output_str)

        print(f"\n{i}/{total} Printing CM to File {svc_model.properties()}")

        cm = confusion_matrix(y_test, y_pred, list(classes.values()))
        cm_output_file = os.path.join(new_dir_path, f"cm_i_{i}_{svc_model.dir_name_string()}.txt")
        print_confusion_matrix_to_file(cm, list(classes.values()), cm_output_file)

        print(f"\n{i}/{total} Saving model {svc_model.properties()}")

        model_output_file = os.path.join(
            new_dir_path, f"svm_model_i_{i}{svc_model.dir_name_string()}.joblib")
        joblib.dump(svc_model.get_model() , model_output_file)

        print(f"\n{i}/{total} SVM model saved at {model_output_file}")

        # print(f"\nClassifing Image kernel='{kernel}' and C={C}")
        # classified_image = classify_image(large_image_path, svm_clf, class_colors)
        # classified_image.save(
        #     f"{img_out_dir}/classified_image-kernel_{kernel}-C_{C}.png")
        print(f"\nFinished {svc_model.properties()}")

        i += 1

def calculate_final_metrics(svc_model, x_train, y_train, x_test, y_test, classes, img_out_dir, metrics_output_file):
    print(f"\nTraining SVM with {svc_model.properties()}")
    unix_time = int(time.time())
    new_dir_name = f"final_{unix_time}{svc_model.dir_name_string()}"
    new_dir_path = os.path.join(img_out_dir, new_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)

    print(f"\nTRAINING SVM with {svc_model.properties()}")

    svc_model.train(x_train, y_train)
    y_pred = svc_model.predict(x_test)

    metrics = calculate_metrics(y_test, y_pred)

    output_str = ""
    for cls, metric in metrics.items():
        output_str += (f"{svc_model.csv_properties()};{cls};"
                        f"{metric['precision']:.4f};{metric['recall']:.4f};{metric['f1_score']:.4f};{metric['accuracy']:.4f}\n")

    with lock:
        print(f"\nPrinting to File {svc_model.properties()}")
        with open(metrics_output_file, 'a') as f:
            f.write(output_str)

    print(f"\nPrinting CM to File {svc_model.properties()}")

    cm = confusion_matrix(y_test, y_pred, list(classes.values()))
    cm_output_file = os.path.join(new_dir_path, f"cm_{svc_model.dir_name_string()}.txt")
    print_confusion_matrix_to_file(cm, list(classes.values()), cm_output_file)

    print(f"\nSaving model {svc_model.properties()}")

    model_output_file = os.path.join(
        new_dir_path, f"svm_model_{svc_model.dir_name_string()}.joblib")
    joblib.dump(svc_model.get_model() , model_output_file)

    print(f"\nSVM model saved at {model_output_file}")

    print(f"\nFinished {svc_model.properties()}")


# PUEDE SER MAS EFICIENTE
def create_models(kernels, c, gamma, cache_size, degree):
    models = []
    for kernel in kernels:
        match kernel:
            case 'linear':
                for c_value in c:
                    for cache_size_value in cache_size:
                        models.append(linear_svc(kernel, c_value, cache_size_value))
            case 'poly':
                for c_value in c:
                    for cache_size_value in cache_size:
                        for gamma_value in gamma:
                            for degree_value in degree:
                                models.append(poly_svc(kernel, c_value, gamma_value, degree_value, cache_size_value))
            case 'rbf':
                for c_value in c:
                    for cache_size_value in cache_size:
                        for gamma_value in gamma:
                            models.append(rbf_svc(kernel, c_value, gamma_value, cache_size_value))
            case 'sigmoid':
                for c_value in c:
                    for cache_size_value in cache_size:
                        for gamma_value in gamma:
                            models.append(sigmoid_svc(kernel, c_value, gamma_value, cache_size_value)) # degree es irrelevante
            case _:
                raise ValueError(f"Kernel desconocido: {kernel}")
    return models


def split_data(X, y, test_size=0.2, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python script.py <image_dir> <large_image_path> <img_out_dir>")
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
    large_image_path = sys.argv[2]
    img_out_dir = sys.argv[3]
    config_json_path = sys.argv[4]

    # CONFIGS
    with open(config_json_path, 'r') as f:
        config = json.load(f)

    kernels = config.get('kernels', [])
    c_values = config.get('c', [])
    gamma_values = config.get('gamma', [])
    cache_size_values = config.get('cache_size', [])
    degree_values = config.get('degree', [])
    reduction_percent = config.get('reduction_percent')
    n_splits = config.get('n_splits')


    # AHORA SI EMPIEZO
    X, y = load_pixels_as_data(image_dir, classes, reduction_percent)

    # Esto es para evaluar el C final que elegimos
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)

    # Esto es para elegir el mejor C con los datos de entrenamiento con validacion cruzada
    train_sets, test_sets = split_data_into_equal_sets(X_train, y_train, n_splits)

    svc_models = create_models(kernels=kernels, c=c_values, gamma=gamma_values, cache_size=cache_size_values,  degree=degree_values)


    # ARCHIVO DE SALIDA DE RESULTADOS PARA ELEGIR C
    metrics_output_file = os.path.join(img_out_dir, "metrics.csv")
    if not os.path.exists(metrics_output_file):
        with open(metrics_output_file, 'w') as f:
            f.write("kernel;c_value;gamma;degree;iteration;class;precision;recall;f1;accuracy\n")
    lock = multiprocessing.Lock()


    # LISTA DE ARGUMENTOS PARA EL THREAD
    jobs = [(svc_model, train_sets, test_sets, classes, class_colors, large_image_path, img_out_dir, metrics_output_file) 
            for svc_model in svc_models]

    with multiprocessing.Pool(processes=3) as pool:
        pool.starmap(run_svm, jobs)
    
    print("FINISHED DEFINITION BEST MODEL")


    # ARCHIVO DE SALIDA DE RESULTADOS PARA VER COMO LE FUE AL MEJOR C (Lo hago con todos para no definir aca "el mejor")
    metrics_output_file = os.path.join(img_out_dir, "metrics_final.csv")
    if not os.path.exists(metrics_output_file):
        with open(metrics_output_file, 'w') as f:
            f.write("kernel;c_value;gamma;degree;class;precision;recall;f1;accuracy\n")

    jobs = [(svc_model, X_train, y_train, X_test, y_test ,classes, img_out_dir, metrics_output_file) 
            for svc_model in svc_models]

    with multiprocessing.Pool(processes=3) as pool:
        pool.starmap(calculate_final_metrics, jobs)
    
    print("FINISHED DEFINITION METRICS OF BEST MODEL")
