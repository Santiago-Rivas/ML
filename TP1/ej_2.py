import pandas as pd
from sklearn.model_selection import train_test_split
import os


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


class NaiveBayesClassifier:
    def __init__(self):
        self.vocab = set()
        self.class_priors = {}
        self.word_counts = {}
        self.class_word_counts = {}

    def fit(self, X, y):
        for i in range(len(X)):
            label = y.iloc[i]
            words = self._tokenize(X.iloc[i])
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

    def _tokenize(self, text):
        return text.lower().split()

    def _calculate_posteriors(self, text):
        words = self._tokenize(text)
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


# Define the file paths
pkl_file = 'data/Noticias_argentinas.pkl'
excel_file = 'data/Noticias_argentinas.xlsx'

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


no_cat = df[df["categoria"].isna()]
with_cat = df[df["categoria"].notna()]

# print(no_cat)
# print(with_cat)

RANDOM_STATE = 42

train_set, temp_set = train_test_split(
    with_cat, test_size=0.3, random_state=RANDOM_STATE, stratify=with_cat['categoria'])

test_set, val_set = train_test_split(
    temp_set, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_set['categoria'])

train_len = len(train_set)
test_len = len(test_set)
val_len = len(val_set)

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

categories = with_cat['categoria'].unique()
# print(categories)


x_train = train_set['titular']
y_train = train_set['categoria']

x_test = train_set['titular']
y_test = train_set['categoria']


# Train the Naive Bayes classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(x_train, y_train)

# Make predictions
y_pred = nb_classifier.predict(x_test)

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
