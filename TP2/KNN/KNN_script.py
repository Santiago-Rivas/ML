# %% [markdown]
# ### Imports

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %% [markdown]
# # 1. Deep Analysis of Variables and Correlation with Star Rating

# %% [markdown]
# ## 1.1 Load and Explore the Dataset

# %%
# Load the dataset
df = pd.read_csv('reviews_sentiment.csv', sep=';')

# Display the first few rows
df.head()

# %% [markdown]
# ## 1.2 Handle Missing Values (Imputation)

# %%
# Check for missing values
df.isnull().sum()

# %% [markdown]
# We'll impute them using the mode

# %%
# Impute missing values in 'titleSentiment' with the mode
mode_title_sentiment = df['titleSentiment'].mode()[0]
df['titleSentiment'] = df['titleSentiment'].fillna(mode_title_sentiment)

# %%
# Verify that there are no more missing values
df['titleSentiment'].isnull().sum()

# %% [markdown]
# Updating the enconding of titleSentiment and textSentiment

# %%
# Map 'negative' to 0 and 'positive' to 1
sentiment_mapping = {'negative': 0, 'positive': 1}

df['titleSentiment_encoded'] = df['titleSentiment'].map(sentiment_mapping)
df['textSentiment_encoded'] = df['textSentiment'].map(sentiment_mapping)

# %%
# Check the unique values after encoding
print('Unique values in titleSentiment_encoded:', df['titleSentiment_encoded'].unique())
print('Unique values in textSentiment_encoded:', df['textSentiment_encoded'].unique())

# %% [markdown]
# ## 1.3 Descriptive Statistics

# %%
# Numerical variables
df.describe()

# %% [markdown]
# ### Distribution of Star Raiting
#
# - Visualizes how many reviews fall into each star rating.
# - Helps identify if the dataset is balanced or skewed.

# %%
# Plot the distribution of 'Star Rating'
sns.countplot(x='Star Rating', data=df)
plt.title('Distribution of Star Ratings')
plt.savefig('output/Distribution_of_Star_Ratings.png')

# %% [markdown]
# ### Statistics for wordcount and sentimentValue

# %%
# Histograms for 'wordcount' and 'sentimentValue'
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(df['wordcount'], kde=True, ax=ax[0])
ax[0].set_title('Distribution of Word Count')

sns.histplot(df['sentimentValue'], kde=True, ax=ax[1])
ax[1].set_title('Distribution of Sentiment Value')

plt.savefig("output/Histograms_wordcount_sentimentValue.png")

# %% [markdown]
# ### Statistics for titleSentiment and textSentiment

# %%
# Plot distribution of titleSentiment_encoded
plt.figure(figsize=(6, 4))
sns.countplot(x='titleSentiment_encoded', data=df)
plt.title('Distribution of Title Sentiment (Encoded)')
plt.xlabel('Title Sentiment (0: Negative, 1: Positive)')
plt.savefig("output/titleSentiment_encoded.png")

# Plot distribution of textSentiment_encoded
plt.figure(figsize=(6, 4))
sns.countplot(x='textSentiment_encoded', data=df)
plt.title('Distribution of Text Sentiment (Encoded)')
plt.xlabel('Text Sentiment (0: Negative, 1: Positive)')
plt.savefig("output/distribution_textSentiment_encoded.png")

# %% [markdown]
# ## 1.4 Correlation Analysis
#
# - Spearman correlation is useful for ordinal data or non-linear relationships.
# - It assesses how well the relationship between two variables can be described by a monotonic function.

# %% [markdown]
# Compute Pearson Correlation Coefficient

# %%
# Select relevant variables
corr_vars = ['wordcount', 'sentimentValue', 'titleSentiment_encoded', 'textSentiment_encoded', 'Star Rating']

# Compute correlation matrix
corr_matrix = df[corr_vars].corr(method='pearson')

# Display the correlation matrix
corr_matrix

# %% [markdown]
# Visualize Correlation Matrix

# %%
# Heatmap of correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig("output/Heatmap_correlation_matrix.png")

# %% [markdown]
# ## 1.5 Information Gain with Shannon Entropy
#
# - Variables with higher information gain are more significant in predicting Star Rating.
# - This helps in feature selection for the model.

# %%
from scipy.stats import entropy

# Calculate the entropy of 'Star Rating'
star_rating_counts = df['Star Rating'].value_counts(normalize=True)
star_rating_entropy = entropy(star_rating_counts, base=2)
print(f"Entropy of 'Star Rating': {star_rating_entropy:.4f}")

# %% [markdown]
# Compute Information Gain for Each Variable:

# %%
def information_gain(data, split_attribute, target_attribute='Star Rating'):
    # Calculate the entropy before the split
    total_entropy = entropy(data[target_attribute].value_counts(normalize=True), base=2)

    # Calculate the values and counts for the split attribute
    vals, counts = np.unique(data[split_attribute], return_counts=True)

    # Calculate the weighted entropy after the split
    weighted_entropy = 0
    for i in range(len(vals)):
        subset = data[data[split_attribute] == vals[i]]
        subset_entropy = entropy(subset[target_attribute].value_counts(normalize=True), base=2)
        weighted_entropy += (counts[i] / np.sum(counts)) * subset_entropy

    # Information gain is the difference in entropy
    info_gain = total_entropy - weighted_entropy
    return info_gain

# Compute information gain for each variable
variables = ['wordcount', 'sentimentValue', 'titleSentiment_encoded', 'textSentiment_encoded']
info_gains = {}
for var in variables:
    ig = information_gain(df, var)
    info_gains[var] = ig
    print(f"Information Gain for {var}: {ig:.4f}")

# %% [markdown]
# Plot Information Gain:

# %%
# Bar chart of information gain
plt.figure(figsize=(8, 6))
sns.barplot(x=list(info_gains.keys()), y=list(info_gains.values()), palette='viridis')
plt.title('Information Gain of Variables')
plt.ylabel('Information Gain')
plt.savefig("output/bar_chart_info_gain.png")


# %% [markdown]
# ## 1.6 Principal Component Analysis (PCA)

# %% [markdown]
# Standardize the Data:

# %%

# Select numerical features including encoded sentiments
features = ['wordcount', 'sentimentValue', 'titleSentiment_encoded', 'textSentiment_encoded']

# Standardize features
x = df[features].values
x = StandardScaler().fit_transform(x)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)

# Create a DataFrame with principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Star Rating'] = df['Star Rating']

# Explained variance
print(f"Explained variance by component: {pca.explained_variance_ratio_}")

# %%
# Scree plot
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, 'o-')
plt.title('Updated Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.savefig("output/scree_plot.png")


# %%
# Biplot of the first two principal components
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Star Rating', data=pca_df, palette='bright')
plt.title('Updated PCA Biplot')
plt.savefig("output/biplot_two_principle_components.png")


# %% [markdown]
# # 2. Data Standardization and Normalization

# %%
from sklearn.model_selection import train_test_split

# Features and target variable
features = ['sentimentValue', 'titleSentiment_encoded', 'textSentiment_encoded', 'wordcount'] # TODO: Modificar esto e ir probando diferentes combinaciones
X = df[features]
Y = df['Star Rating']


# %% [markdown]
# Standardize features to have a mean of 0 and a standard deviation of 1.

# %%
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit on training data and transform both training and testing data
X_std = scaler.fit_transform(X)

# %% [markdown]
# Normalize features to a range between 0 and 1.

# %%
from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
min_max_scaler = MinMaxScaler()

# Fit on training data and transform both training and testing data
X_norm = min_max_scaler.fit_transform(X)

# %%
# Create a DataFrame for plotting
X_std_df = pd.DataFrame(X_std, columns=features)
X_norm_df = pd.DataFrame(X_norm, columns=features)

# %%
# Plot standardized features
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()
for i, feature in enumerate(features):
    sns.histplot(X_std_df[feature], kde=True, ax=axs[i])
    axs[i].set_title(f'Standardized {feature}')
plt.tight_layout()
plt.savefig("output/std_features.png")

# Plot normalized features
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()
for i, feature in enumerate(features):
    sns.histplot(X_norm_df[feature], kde=True, ax=axs[i])
    axs[i].set_title(f'Normalized {feature}')
plt.tight_layout()
plt.savefig("output/norm_features.png")

# %% [markdown]
# # 3. Implementing K-NN and Weighted K-NN from Scratch

# %% [markdown]
# ## 3.1 K-Nearest Neighbors (K-NN)

# %%
from collections import Counter

# %%
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# %% [markdown]
# Define K-NN Function:

# %%
def knn_predict(X_train, y_train, X_test_instance, k):
    # Compute distances between X_test_instance and all X_train instances
    distances = np.linalg.norm(X_train - X_test_instance, axis=1)
    # Get the indices of the k nearest neighbors
    k_indices = distances.argsort()[:k]
    # Get the labels of the k nearest neighbors
    k_nearest_labels = y_train[k_indices]
    # Majority vote
    most_common = Counter(k_nearest_labels).most_common(1)
    predicted_class = most_common[0][0]
    # Calculate class scores (proportion of votes)
    class_counts = Counter(k_nearest_labels)
    total_votes = sum(class_counts.values())
    class_scores = {cls: count / total_votes for cls, count in class_counts.items()}
    return predicted_class, class_scores

# %% [markdown]
# Define Weighted K-NN Function:

# %%
def weighted_knn_predict(X_train, y_train, X_test_instance, k):
    # Compute distances
    distances = np.linalg.norm(X_train - X_test_instance, axis=1)
    # Get the indices of the k nearest neighbors
    k_indices = distances.argsort()[:k]
    # Get labels and distances of the k nearest neighbors
    k_nearest_labels = y_train[k_indices]
    k_nearest_distances = distances[k_indices]
    # Compute weights (inverse of distance)
    weights = 1 / (k_nearest_distances + 1e-5)  # Avoid division by zero
    # Calculate weighted votes
    class_weights = {}
    for label, weight in zip(k_nearest_labels, weights):
        class_weights[label] = class_weights.get(label, 0) + weight
    # Select the class with the highest total weight
    predicted_class = max(class_weights.items(), key=lambda x: x[1])[0]
    # Normalize weights to get class scores
    total_weight = sum(class_weights.values())
    class_scores = {cls: weight / total_weight for cls, weight in class_weights.items()}
    return predicted_class, class_scores

# %% [markdown]
# Define Cross-Validation Function

# %%
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize

def cross_validate_knn(X, y, k_values, num_folds=10, weighted=False, random_state=42):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    classes = np.unique(y)
    n_classes = len(classes)
    avg_accuracies = {k: [] for k in k_values}
    fold = 1

    for train_index, test_index in kf.split(X):
        print(f"Processing fold {fold}/{num_folds}")
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        for k in k_values:
            preds = []
            for X_test_instance in X_test_fold:
                if weighted:
                    pred, _ = weighted_knn_predict(X_train_fold, y_train_fold, X_test_instance, k)
                else:
                    pred, _ = knn_predict(X_train_fold, y_train_fold, X_test_instance, k)
                preds.append(pred)
            acc = accuracy_score(y_test_fold, preds)
            avg_accuracies[k].append(acc)
        fold += 1

    # Compute average accuracies for each k
    for k in avg_accuracies:
        avg_accuracies[k] = np.mean(avg_accuracies[k])

    # Determine the best k
    best_k = max(avg_accuracies, key=avg_accuracies.get)
    print(f"Best k: {best_k} with accuracy: {avg_accuracies[best_k]:.4f}")

    return best_k, avg_accuracies

# %%
def cross_validate_knn_best_k(X, y, best_k, num_folds=10, weighted=False):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    classes = np.unique(y)
    n_classes = len(classes)
    all_y_true = []
    all_y_scores = []
    all_preds = []
    confusion_mat = np.zeros((n_classes, n_classes))
    fold = 1

    for train_index, test_index in kf.split(X):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        preds = []
        scores = []
        for X_test_instance in X_test_fold:
            if weighted:
                pred, class_scores = weighted_knn_predict(X_train_fold, y_train_fold, X_test_instance, best_k)
            else:
                pred, class_scores = knn_predict(X_train_fold, y_train_fold, X_test_instance, best_k)
            preds.append(pred)
            # Create a score vector aligned with the classes
            score_vector = [class_scores.get(cls, 0) for cls in classes]
            scores.append(score_vector)

        # Update confusion matrix
        cm = confusion_matrix(y_test_fold, preds, labels=classes)
        confusion_mat += cm

        # Collect true labels and scores
        all_y_true.extend(y_test_fold)
        all_y_scores.extend(scores)
        all_preds.extend(preds)

        fold += 1

    # Convert lists to arrays
    all_y_true = np.array(all_y_true)
    all_y_scores = np.array(all_y_scores)
    all_preds = np.array(all_preds)

    # Binarize the true labels
    all_y_true_binarized = label_binarize(all_y_true, classes=classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_y_true_binarized[:, i], all_y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return confusion_mat, fpr, tpr, roc_auc, classes

# %% [markdown]
# ## 3.2 Evaluate the Model

# %% [markdown]
# Run Cross-Validation for K-NN and Weighted K-NN:

# %%
# Convert DataFrames to numpy arrays
X_scaled = X_std_df.values  # Use the scaled data (standardized or normalized)
y = Y.values

# k_values = range(1, 21)  # Testing k from 1 to 20
# 
# best_k_knn, avg_accuracies_knn = cross_validate_knn(X_scaled, y, k_values, num_folds=10, weighted=False, random_state=42)
# best_k_weighted_knn, avg_accuracies_weighted_knn = cross_validate_knn(X_scaled, y, k_values, num_folds=10, weighted=True, random_state=42)
# 
# 
# # %% [markdown]
# # Plotting the Results:
# 
# # %%
# # Plot the average accuracies for different k values
# plt.figure(figsize=(10, 6))
# plt.plot(k_values, [avg_accuracies_knn[k] for k in k_values], label='K-NN Accuracy', marker='o')
# plt.plot(k_values, [avg_accuracies_weighted_knn[k] for k in k_values], label='Weighted K-NN Accuracy', marker='s')
# plt.title('Average Accuracy vs. K Value (Cross-Validation)')
# plt.xlabel('Number of Neighbors (k)')
# plt.ylabel('Average Accuracy')
# plt.legend()
# plt.grid(True)
# plt.savefig("output/avg_acc_different_k.png")

random_states = range(1, 1000, 100)         # Generate Random states
k_values = range(1, 21)                     # Testing k from 1 to 20

# Initialize lists to accumulate results across random states
avg_accuracies_knn_all_states = {k: [] for k in k_values}
avg_accuracies_weighted_knn_all_states = {k: [] for k in k_values}

# Iterate over random states
for random_state in random_states:
    best_k_knn, avg_accuracies_knn = cross_validate_knn(X_scaled, y, k_values, num_folds=10, weighted=False, random_state=random_state)
    best_k_weighted_knn, avg_accuracies_weighted_knn = cross_validate_knn(X_scaled, y, k_values, num_folds=10, weighted=True, random_state=random_state)

    # Accumulate accuracies for each k value across random states
    for k in k_values:
        avg_accuracies_knn_all_states[k].append(avg_accuracies_knn[k])
        avg_accuracies_weighted_knn_all_states[k].append(avg_accuracies_weighted_knn[k])

# Calculate the average accuracy and standard deviation across random states for each k
avg_accuracies_knn_mean = {k: np.mean(avg_accuracies_knn_all_states[k]) for k in k_values}
avg_accuracies_knn_std = {k: np.std(avg_accuracies_knn_all_states[k]) for k in k_values}
avg_accuracies_weighted_knn_mean = {k: np.mean(avg_accuracies_weighted_knn_all_states[k]) for k in k_values}
avg_accuracies_weighted_knn_std = {k: np.std(avg_accuracies_weighted_knn_all_states[k]) for k in k_values}

# Plot the average accuracies with error bars for different k values
plt.figure(figsize=(10, 6))

# Plot for unweighted K-NN
plt.errorbar(
    k_values, 
    [avg_accuracies_knn_mean[k] for k in k_values], 
    yerr=[avg_accuracies_knn_std[k] for k in k_values], 
    label='K-NN Accuracy', 
    marker='o', 
    capsize=5
)

# Plot for weighted K-NN
plt.errorbar(
    k_values, 
    [avg_accuracies_weighted_knn_mean[k] for k in k_values], 
    yerr=[avg_accuracies_weighted_knn_std[k] for k in k_values], 
    label='Weighted K-NN Accuracy', 
    marker='s', 
    capsize=5
)

plt.title('Average Accuracy vs. K Value (Cross-Validation across Random States)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Average Accuracy')
plt.legend()
plt.grid(True)
plt.xticks(ticks=k_values)  # Ensure x-axis shows integer ticks only

# Save the plot
plt.savefig("output/avg_acc_different_k_with_error_bars.png")

# %% [markdown]
# ## 4. Confusion Matrix & ROC Curves for the best K

# %%
# For K-NN
confusion_mat_knn, fpr_knn, tpr_knn, roc_auc_knn, classes = cross_validate_knn_best_k(X_scaled, y, best_k_knn, num_folds=10, weighted=False)

# For Weighted K-NN
confusion_mat_weighted_knn, fpr_weighted_knn, tpr_weighted_knn, roc_auc_weighted_knn, classes = cross_validate_knn_best_k(X_scaled, y, best_k_weighted_knn, num_folds=10, weighted=True)


# %% [markdown]
# Plot Confusion Matrix for Best K

# %%
def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("output/confusion_mat_k.png")

plot_confusion_matrix(confusion_mat_knn, classes, f"Confusion Matrix for K-NN k={best_k_knn})")
plot_confusion_matrix(confusion_mat_weighted_knn, classes, f"Confusion Matrix for Weighted K-NN k={best_k_weighted_knn})")



# %% [markdown]
# Plot ROC Curves

# %%
def plot_roc_curves(fpr_dict, tpr_dict, roc_auc_dict, classes, title):
    plt.figure(figsize=(8, 6))
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green']
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=2,
                 label='Class {0} (AUC = {1:0.2f})'.format(classes[i], roc_auc_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(f"output/ROC_{title}.png")

# Plot ROC Curves for K-NN
plot_roc_curves(fpr_knn, tpr_knn, roc_auc_knn, classes, f'ROC Curves for K-NN (k={best_k_knn})')

# Plot ROC Curves for Weighted K-NN
plot_roc_curves(fpr_weighted_knn, tpr_weighted_knn, roc_auc_weighted_knn, classes, f'ROC Curves for Weighted K-NN (k={best_k_weighted_knn})')


# %% [markdown]
# ## 5. Decision Boundary Visualization

# %%
from matplotlib.colors import ListedColormap


def plot_decision_boundaries_2d(X_train, y_train, k, weighted=False, feature_names=None):
    # Define the color maps for the plot
    cmap_light = ListedColormap(['#FFAAAA', '#8BD992', '#AAAAFF', '#F5E292', '#8B49A2'])  # Colors for the background
    cmap_bold = ['red', 'green', 'blue', 'orange', 'purple']  # Colors for the actual points

    # Create a mesh grid for plotting the decision boundaries
    h = 0.01  # Step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the class for each point in the mesh grid
    Z = []
    for xx1, yy1 in zip(xx.ravel(), yy.ravel()):
        X_test_instance = np.array([xx1, yy1])
        if weighted:
            pred, _ = weighted_knn_predict(X_train, y_train, X_test_instance, k)
        else:
            pred, _ = knn_predict(X_train, y_train, X_test_instance, k)
        Z.append(pred)

    Z = np.array(Z).reshape(xx.shape)

    # Plot the decision boundaries and training points
    plt.figure(figsize=(10, 8))

    # Use pcolormesh to plot the decision boundaries with proper handling of the background color
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto', alpha=0.8)

    # Plot the training points with the actual class labels
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette=cmap_bold, edgecolor='k', s=20, zorder=2)

    # Label axes
    plt.xlabel(feature_names[0] if feature_names else 'Feature 1')
    plt.ylabel(feature_names[1] if feature_names else 'Feature 2')
    plt.title(f'Decision Boundaries (k={k}, Weighted={weighted})')
    plt.legend(title="Class")
    plt.savefig(f"output/Decision_Boundaries_(k={k}_Weighted={weighted}'.png")


# %%
# Select two features
X_2d = X_scaled[:, [0, 3]]  # sentimentValue and wordcount

# Plot decision boundaries for K-NN
plot_decision_boundaries_2d(X_2d, y, best_k_knn, weighted=False, feature_names=['Sentiment Value', 'Wordcount'])

# Plot decision boundaries for Weighted K-NN
plot_decision_boundaries_2d(X_2d, y, best_k_weighted_knn, weighted=True, feature_names=['Sentiment Value', 'Wordcount'])



