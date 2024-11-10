import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results CSV file
results_df = pd.read_csv('output/hc_best_clusters.csv')

# Set up the visualization style
sns.set(style="whitegrid")


metrics = ['best_num_clusters', 'best_silhouette_score', 'best_davies_bouldin_score', 'best_calinski_harabasz_score',]

for m in metrics:
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=results_df,
        x='feature_pair',
        y=m,
        hue='linkage_method'
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{m} by Feature Pair and Linkage Method")
    plt.xlabel("Feature Pair")
    plt.ylabel(f"{m}")
    plt.legend(title="Linkage Method")
    plt.tight_layout()
    plt.show()

exit()

# 2. Silhouette Score distribution by linkage method
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=results_df,
    x='linkage_method',
    y='best_silhouette_score'
)
plt.title("Silhouette Score Distribution by Linkage Method")
plt.xlabel("Linkage Method")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.show()

# 3. Davies-Bouldin Score distribution by linkage method
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=results_df,
    x='linkage_method',
    y='best_davies_bouldin_score'
)
plt.title("Davies-Bouldin Score Distribution by Linkage Method")
plt.xlabel("Linkage Method")
plt.ylabel("Davies-Bouldin Score (Lower is Better)")
plt.tight_layout()
plt.show()

# 4. Calinski-Harabasz Score distribution by linkage method
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=results_df,
    x='linkage_method',
    y='best_calinski_harabasz_score'
)
plt.title("Calinski-Harabasz Score Distribution by Linkage Method")
plt.xlabel("Linkage Method")
plt.ylabel("Calinski-Harabasz Score (Higher is Better)")
plt.tight_layout()
plt.show()
