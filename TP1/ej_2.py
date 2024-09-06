import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import snowballstemmer
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# Calculate evaluation metrics
def compute_metrics(confusion_matrix):
    metrics = {}
    for category, values in confusion_matrix.items():
        TP = values['TP']
        FP = values['FP']
        TN = values['TN']
        FN = values['FN']

        accuracy = (TP + TN) / (TP + FP + TN +
                                FN) if (TP + FP + TN + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision +
                                               recall) if (precision + recall) > 0 else 0

        metrics[category] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score
        }

    return metrics


def compute_confusion_matrix(y_true, y_pred, categories):
    cm = {category: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
          for category in categories}

    for true, pred in zip(y_true, y_pred):
        for category in categories:
            if true == category and pred == category:
                cm[category]['TP'] += 1  # True Positive
            elif true != category and pred == category:
                cm[category]['FP'] += 1  # False Positive
            elif true == category and pred != category:
                cm[category]['FN'] += 1  # False Negative
            elif true != category and pred != category:
                cm[category]['TN'] += 1  # True Negative

    return cm

class Tokenizer:
    def __init__(self, filter, sanitizer):
        self.filter = filter
        self.sanitizer = sanitizer

    def apply(self, text):
        return [self.sanitizer(word) for word in text.split() if self.filter(word)]


class NaiveBayesClassifier:
    def __init__(self, tokenizer):
        self.vocab = set()
        self.tokenizer = tokenizer
        self.class_priors = {}
        self.word_counts = {}
        self.class_word_counts = {}

    def fit(self, X, y):
        for i in range(len(X)):
            label = y.iloc[i]
            #words = self._tokenize(X.iloc[i])
            words = self.tokenizer.apply(X.iloc[i])
            self.vocab.update(words)
            if label not in self.class_word_counts:
                self.class_word_counts[label] = {}
                self.class_word_counts[label]['__total__'] = 0
            self.class_word_counts[label]['__total__'] += len(words)

            for word in words:
                if word not in self.class_word_counts[label]:
                    self.class_word_counts[label][word] = 0
                self.class_word_counts[label][word] += 1

        # Calculate class priors
        total_documents = len(y)
        self.class_priors = {
            label: count['__total__'] / total_documents for label, count in self.class_word_counts.items()}

    def predict(self, X):
        predictions = []
        for text in X:
            posteriors = self._calculate_posteriors(text)
            predictions.append(max(posteriors, key=posteriors.get))
        return predictions

    def classify(self, X, y, umbral, category):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for index in range(len(X)):
            posteriors = self._calculate_posteriors(X.iloc[index])
            cat_prob = posteriors[category]  # Obtiene el valor máximo asociado a esa clave
            total_sum = sum(posteriors.values())
            if cat_prob/total_sum > umbral:  # Compara si el valor máximo es mayor a 'x'
                if y.iloc[index] == category:
                    TP+=1
                else:
                    FP+=1
            else:
                if y.iloc[index] == category:
                    FN+=1
                else:
                    TN+=1
        return TP, FP, FN, TN

    def _tokenize(self, text):
        return text.lower().split()

    def _calculate_posteriors(self, text):
        #words = self._tokenize(text)
        words = self.tokenizer.apply(text)
        posteriors = {}

        for label in self.class_priors:
            prior = self.class_priors[label]
            likelihood = 1.0

            # Laplace Correction
            for word in words:
                word_count = self.class_word_counts[label].get(word, 0)
                total_words = self.class_word_counts[label]['__total__']
                likelihood *= (word_count + 1) / \
                    (total_words + len(self.vocab))

            posteriors[label] = prior * likelihood

        return posteriors

def read_input(path='data/Noticias_argentinas'):
    # Define the file paths
    pkl_file = path + '.pkl'
    excel_file = path + '.xlsx'

    # Check if the .pkl file exists
    if os.path.exists(pkl_file):
        # If .pkl file exists, load the DataFrame from it
        df = pd.read_pickle(pkl_file)
        print("Loaded DataFrame from pickle file.")
    else:
        # If .pkl file does not exist, read the Excel file and save it to .pkl
        df = pd.read_excel(excel_file)
        df.to_pickle(pkl_file)
        print("Loaded DataFrame from Excel file and saved it to pickle.")

    # Esto me parece que no sirve
    # total_index = df[df['Internacional'] == "Total"].index
    # if not total_index.empty:
    #     stats = df.iloc[:total_index[0] + 1, 5:]
    # else:
    #     stats = df.iloc[:, 5:]
    # print(stats)

    df = df.iloc[:, :4]
    # Display the DataFrame
    # print(df)
    return df


def no_category_filter(df):
    df.loc[df["categoria"] == "Destacadas", "categoria"] = "Noticias destacadas"
    with_cat = df[df["categoria"].notna()]
    df["categoria"] = df["categoria"].fillna("Sin categoría")
    #with_cat = df[df["categoria"].notna() & (df["categoria"] != "Destacadas")]
    # print(no_cat)
    # print(with_cat)
    return df, with_cat


def train_test_split(df, test_size=0.3, random_state=None, stratify_column='categoria'):
    if random_state:
        np.random.seed(random_state)
    
    # Inicializamos listas vacías para entrenamiento y prueba
    train_set = pd.DataFrame(columns=df.columns)
    test_set = pd.DataFrame(columns=df.columns)

    # Estratificar los datos según la columna
    for category in df[stratify_column].unique():
        category_subset = df[df[stratify_column] == category]
        category_subset = category_subset.sample(frac=1)  # Mezclar los datos de esa categoría

        # Dividimos según el tamaño del conjunto de prueba
        split_idx = int(len(category_subset) * (1 - test_size))
        train_subset = category_subset[:split_idx]
        test_subset = category_subset[split_idx:]

        # Eliminar filas completamente vacías (all-NA) antes de concatenar
        train_subset = train_subset.dropna(how='all', axis=0)
        test_subset = test_subset.dropna(how='all', axis=0)

        # Solo concatenar si los subconjuntos no están vacíos
        if not train_subset.empty:
            train_set = pd.concat([train_set, train_subset], ignore_index=True)
        if not test_subset.empty:
            test_set = pd.concat([test_set, test_subset], ignore_index=True)

    # Mezclamos nuevamente ambos conjuntos
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    test_set = test_set.sample(frac=1).reset_index(drop=True)

    return train_set, test_set


def split_train_test(df):
    RANDOM_STATE = 42

    train_set, test_set = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify_column='categoria')

    train_len = len(train_set)
    test_len = len(test_set)

    # print("Training set size:", train_len)
    # print("Testing set size:", test_len)
    # print("Validation set size:", val_len)
    #
    # print("\nTraining set:")
    # print(train_set.head())
    #
    # print("\nTesting set:")
    # print(test_set.head())
    #
    # print("\nValidation set:")
    # print(val_set.head())
    #
    # # Count occurrences of each category in each set
    # print("\nCategory distribution in the Training set:")
    # print(train_set['categoria'].value_counts() / train_len)
    #
    # print("\nCategory distribution in the Testing set:")
    # print(test_set['categoria'].value_counts() / test_len)
    #
    # print("\nCategory distribution in the Validation set:")
    # print(val_set['categoria'].value_counts() / val_len)

    return train_set, test_set

def extract_categories(df):
    categories = df['categoria'].unique()
    # print(categories)
    return categories

def split_x_y(df):
    x = df['titular']
    y = df['categoria']
    return x, y


def values_matrix(y_test, y_pred, categories):
    # Confusion matrix
    confusion_matrix = compute_confusion_matrix(y_test, y_pred, categories)
    print("Confusion Matrix:")
    for category, values in confusion_matrix.items():
        print(f"{category}: {values}")


    # Calculate metrics
    metrics = compute_metrics(confusion_matrix)
    print("Evaluation Metrics:")
    for category, values in metrics.items():
        print(f"{category}: {values}")


def plot_confusion_matrix(title, true_positive, false_negative, false_positive, true_negative):
    # Crear la matriz de confusión
    confusion_matrix = np.array([[true_positive, false_negative],
                                 [false_positive, true_negative]])
    
    # Crear el heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Positive', 'Predicted Negative'], 
                yticklabels=['Actual Positive', 'Actual Negative'])
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

def macroaverage_values_matrix(y_test, y_pred, categories):
    # Crear una matriz vacía con ceros para almacenar las frecuencias
    matrix_size = len(categories)
    confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Crear un mapeo entre las categorías y los índices de la matriz
    category_to_index = {category: i for i, category in enumerate(categories)}
    # Llenar la matriz de confusión
    for true_label, pred_label in zip(y_test, y_pred):
        true_index = category_to_index[true_label]
        pred_index = category_to_index[pred_label]
        confusion_matrix[true_index, pred_index] += 1
    # Inicializamos listas para almacenar métricas por clase
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    sum_TP = 0
    sum_FP = 0
    sum_FN = 0
    sum_TN = 0
    # Calcular las métricas por clase
    for i, category in enumerate(categories):
        TP = confusion_matrix[i, i]  # Verdaderos positivos
        FP = sum(confusion_matrix[:, i]) - TP  # Falsos positivos
        FN = sum(confusion_matrix[i, :]) - TP  # Falsos negativos
        TN = np.sum(confusion_matrix) - (TP + FP + FN)  # Verdaderos negativos
        
        #plot_confusion_matrix(category, TP, FN, FP, TN)

        # Precisión para la clase actual
        if TP + FP > 0:
            precision = TP / (TP + FP)
            #print(category + " Precision: " + str(precision))
        else:
            precision = 0
        precision_per_class.append(precision)
        
        # Recall (Sensibilidad) para la clase actual
        if TP + FN > 0:
            recall = TP / (TP + FN)
            #print(category + " recall: " + str(recall))
        else:
            recall = 0
        recall_per_class.append(recall)
        
        # F1 Score para la clase actual
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            #print(category + " f1: " + str(f1))
        else:
            f1 = 0
        f1_per_class.append(f1)

        sum_TP += TP
        sum_FP += FP
        sum_FN += FN
        sum_TN += TN
    
    #plot_confusion_matrix("Total", sum_TP, sum_FN, sum_FP, sum_TN)

    # Cálculo de macro-averages
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)

    # Accuracy (Exactitud) global
    total_correct = np.trace(confusion_matrix)  # Suma de la diagonal principal
    total_predictions = np.sum(confusion_matrix)
    accuracy = total_correct / total_predictions

    # Mostrar resultados
    print(f"Precisión (Macro Average): {macro_precision:.2f}")
    print(f"Exactitud (Accuracy): {accuracy:.2f}")
    print(f"F1 Score (Macro Average): {macro_f1:.2f}")


def remove_short_words(word):
    return len(word) > 3

def remove_non_alpha(word):
    return word.isalpha()

def complex_filter(word):
    return remove_short_words(word) and remove_non_alpha(word)

def to_lower(word):
    return word.lower()

def remove_punctuation(word):
    return word.translate(str.maketrans('', '', string.punctuation))

stemmer = snowballstemmer.stemmer('spanish')

def stemming_es(palabra):
    return stemmer.stemWord(palabra)

def complex_sanitize(word):
    return to_lower(remove_punctuation(word))

def show_matrix(y_test, y_pred, categories):
    # Crear una matriz vacía con ceros para almacenar las frecuencias
    matrix_size = len(categories)
    confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Crear un mapeo entre las categorías y los índices de la matriz
    category_to_index = {category: i for i, category in enumerate(categories)}

    # Llenar la matriz de confusión
    for true_label, pred_label in zip(y_test, y_pred):
        true_index = category_to_index[true_label]
        pred_index = category_to_index[pred_label]
        confusion_matrix[true_index, pred_index] += 1

    # Convertir a porcentajes por fila (dividiendo cada fila por la suma de la fila)
    confusion_matrix_percentage = confusion_matrix.astype(float) / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100

    # Visualizar la matriz de confusión
    sns.heatmap(confusion_matrix_percentage, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=categories, yticklabels=categories, vmax=100)

    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title('Confusion Matrix as Percentages', fontsize=16)
    plt.xticks(rotation=45, ha='right')  # Rotar etiquetas de las columnas
    plt.yticks(rotation=0)  # Asegurar que las etiquetas de las filas estén horizontales
    plt.show()

    # Paso 2: Crear una matriz vacía con ceros para almacenar las frecuencias
    matrix_size = len(categories)
    confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Paso 3: Crear un mapeo entre las categorías y los índices de la matriz
    category_to_index = {category: i for i, category in enumerate(categories)}

    # Paso 4: Llenar la matriz de confusión
    for true_label, pred_label in zip(y_test, y_pred):
        true_index = category_to_index[true_label]
        pred_index = category_to_index[pred_label]
        confusion_matrix[true_index, pred_index] += 1

    # Paso 5: Visualizar la matriz de confusión
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)

    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xticks(rotation=45, ha='right')  # Rotar etiquetas de las columnas
    plt.yticks(rotation=0)  # Asegurar que las etiquetas de las filas estén horizontales
    plt.show()


def roc(x_test, y_test, categories, nb_classifier):
    thresholds = np.linspace(0.0, 1.0, 11)

    # Crear el gráfico FP vs TP en porcentajes
    plt.figure(figsize=(8, 6))

    for category in categories:
        # Listas para guardar los porcentajes de TP y FP
        TP_percentages = []
        FP_percentages = []

        # Probar el clasificador para cada umbral
        for threshold in thresholds:
            TP, FP, FN, TN = nb_classifier.classify(x_test, y_test, threshold, category)
            TP_percentage = TP / (TP + FN)
            FP_percentage = FP / (FP + TN)
            
            TP_percentages.append(TP_percentage)
            FP_percentages.append(FP_percentage)

        plt.plot(FP_percentages, TP_percentages, marker='o', linestyle='-', label=category)
    
    plt.xlabel('Taza de Falsos Positivos')
    plt.ylabel('Taza de Verdaderos Positivos')
    plt.xscale('log')
    plt.title('Taza de FP vs Taza de TP para diferentes umbrales')
    plt.legend(title='Categoría', loc='best')
    plt.grid(False)
    plt.show()


def main():
    df = read_input()
    df_no_cat, df_categories = no_category_filter(df)
    train_set, test_set = split_train_test(df_categories)
    x_train, y_train = split_x_y(train_set)
    x_test, y_test = split_x_y(test_set)

    tokenizer = Tokenizer(complex_filter, complex_sanitize)
    nb_classifier = NaiveBayesClassifier(tokenizer)
    nb_classifier.fit(x_train, y_train)

    categories = extract_categories(train_set)
    y_pred = nb_classifier.predict(x_test)

    show_matrix(y_test, y_pred, categories)
    macroaverage_values_matrix(y_test, y_pred, categories)

    #values_matrix(y_test, y_pred, categories)

    # OJO, DEMORA MUCHO
    #roc(x_test, y_test, categories, nb_classifier)


if __name__ == '__main__':
    main()

