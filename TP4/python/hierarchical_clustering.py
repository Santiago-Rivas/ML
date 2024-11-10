from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.animation as animation
import plotly.figure_factory as ff
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram, fcluster


def animate_cluster_merging(Z, X):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Inicialmente, cada punto es un cluster
    clusters = np.arange(X.shape[0])

    # Reducir a 2D si es necesario
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
    else:
        X_vis = X

    frames = []
    cluster_history = [clusters.copy()]

    # Generar la historia de clusters
    for i in range(Z.shape[0]):
        cluster1, cluster2 = int(Z[i, 0]), int(Z[i, 1])
        clusters[clusters == cluster2] = cluster1
        cluster_history.append(clusters.copy())

    def update(frame):
        ax.clear()
        sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1],
                        hue=cluster_history[frame], palette='tab10', ax=ax, legend=False)
        ax.set_title(f'Fusión de Clusters - Iteración {frame}')

    ani = animation.FuncAnimation(
        fig, update, frames=len(cluster_history), blit=False)
    ani.save('hc_outputs/cluster_merging.gif', writer='pillow')
    plt.close()


def animate_dendrogram_threshold(Z, X):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    thresholds = np.linspace(np.max(Z[:, 2]), 0, num=20)

    # Reducir a 2D si es necesario
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
    else:
        X_vis = X

    def update(frame):
        thresh = thresholds[frame]
        ax1.clear()
        dendrogram(Z, color_threshold=thresh, ax=ax1, no_labels=True)
        ax1.axhline(y=thresh, c='k', linestyle='--')
        ax1.set_title(f'Dendrograma (Threshold = {thresh:.2f})')

        clusters = fcluster(Z, thresh, criterion='distance')
        ax2.clear()
        sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1],
                        hue=clusters, palette='tab10', ax=ax2, legend=False)
        ax2.set_title('Clusters en 2D')

    ani = animation.FuncAnimation(
        fig, update, frames=len(thresholds), blit=False)
    ani.save('hc_outputs/dendrogram_threshold.gif', writer='pillow')
    plt.close()


def plot_dendrogram(Z):
    plt.figure(figsize=(10, 7))
    dendrogram(Z.astype(float))
    plt.title('Dendrograma del Clustering Jerárquico')
    plt.xlabel('Índices de las Muestras')
    plt.ylabel('Distancia')
    plt.show()


def animate_distance_matrix(distance_matrices):
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        im = ax.imshow(distance_matrices[frame], cmap='viridis')
        ax.set_title(f'Matriz de Distancias - Iteración {frame}')
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(distance_matrices), blit=True)
    ani.save('hc_outputs/distance_matrix_evolution.gif', writer='pillow')
    plt.close()


def interactive_dendrogram(Z):
    fig = ff.create_dendrogram(X_scaled_sample, linkagefun=lambda x: Z)
    fig.update_layout(width=800, height=500)
    fig.show()


def hierarchical_clustering(X, linkage='single'):
    n_samples = X.shape[0]
    clusters = [[i] for i in range(n_samples)]
    cluster_labels = [i for i in range(n_samples)]
    distances = np.sqrt(
        ((X[:, np.newaxis] - X[np.newaxis, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    Z = []
    distance_matrices = [distances.copy()]

    for step in range(n_samples - 1):
        # Encontrar el par con la distancia mínima
        idx = np.argmin(distances)
        i, j = np.unravel_index(idx, distances.shape)
        dist = distances[i, j]

        # Registrar la unión
        c1_label = cluster_labels[i]
        c2_label = cluster_labels[j]
        Z.append([c1_label, c2_label, dist, len(
            clusters[i]) + len(clusters[j])])

        # Fusionar clusters
        new_cluster = clusters[i] + clusters[j]
        new_label = n_samples + step
        clusters.append(new_cluster)
        cluster_labels.append(new_label)

        # Eliminar los clusters antiguos
        # Eliminar en orden inverso para evitar el desplazamiento de índices
        for index in sorted([i, j], reverse=True):
            del clusters[index]
            del cluster_labels[index]

        # Actualizar la matriz de distancias
        distances = np.delete(distances, [i, j], axis=0)
        distances = np.delete(distances, [i, j], axis=1)

        # Calcular distancias entre el nuevo cluster y los clusters restantes
        new_dist_row = []
        for k in range(len(clusters) - 1):  # Excluir el nuevo cluster mismo
            if linkage == 'single':
                d = min([np.linalg.norm(X[p1] - X[p2])
                        for p1 in new_cluster for p2 in clusters[k]])
            elif linkage == 'complete':
                d = max([np.linalg.norm(X[p1] - X[p2])
                        for p1 in new_cluster for p2 in clusters[k]])
            elif linkage == 'average':
                dists = [np.linalg.norm(X[p1] - X[p2])
                         for p1 in new_cluster for p2 in clusters[k]]
                d = sum(dists) / len(dists)
            elif linkage == 'centroid':
                centroid_new = np.mean(X[new_cluster], axis=0)
                centroid_k = np.mean(X[clusters[k]], axis=0)
                d = np.linalg.norm(centroid_new - centroid_k)
            new_dist_row.append(d)
        # Distancia del nuevo cluster a sí mismo es infinito
        new_dist_row.append(np.inf)
        new_dist_row = np.array(new_dist_row)

        # Añadir la nueva fila y columna a la matriz de distancias
        distances = np.vstack([distances, new_dist_row[:-1]])
        new_dist_col = np.append(new_dist_row[:-1], np.inf).reshape(-1, 1)
        distances = np.hstack([distances, new_dist_col])

        # Añadir la matriz de distancias actualizada a la lista
        distance_matrices.append(distances.copy())

    return np.array(Z), distance_matrices


plt.rcParams['figure.dpi'] = 300
sns.set(style="whitegrid")

print("Reading CSV")
df = pd.read_csv('movie_data.csv', sep=';').dropna()
filtered_df = df[df['genres'].isin(['Action', 'Drama', 'Comedy'])]
print(filtered_df)

df_sample = df
# df_sample = filtered_df

print("Creating Sample")
df_sample = df_sample.sample(frac=0.1, random_state=42)
print(df_sample)

numerical_features_full = ['budget', 'popularity',
                           'revenue', 'runtime', 'vote_average', 'vote_count']


print("Numerical Features")
numerical_features_sample = ['runtime', 'revenue']
X_sample = df_sample[numerical_features_sample].values

numerical_features_genre = numerical_features_sample
numerical_features_genre.append('genres')

X_samle_with_genre = df_sample[numerical_features_genre]
X_samle_with_genre_values = df_sample[numerical_features_genre].values
scaler = StandardScaler()

# plt.figure(figsize=(10, 7))
# sns.scatterplot(data=X_samle_with_genre, x=numerical_features_sample[0], y=numerical_features_sample[1], hue='genres', palette='Set1', s=100)
# plt.legend(title='Genres')
# plt.show()

# print("Scaling")
# X_scaled_sample = scaler.fit_transform(X_sample)
# print("Clustering")
# Z, distance_matrices = hierarchical_clustering(
#     X_scaled_sample, linkage='single')


# clusters_classified = {}
# for c in Z:
#     print(c)
#     i = c[0]
#     j = c[1]
#     if i < len(X_samle_with_genre_values):
#         i_gen = X_samle_with_genre_values[int(i)][2]
#     elif i in clusters_classified:
#         i_gen = clusters_classified[i]
#     else:
#         i_gen = None
#     if j < len(X_samle_with_genre_values):
#         j_gen = X_samle_with_genre_values[int(i)][2]
#     elif i in clusters_classified:
#         j_gen = clusters_classified[j]
#     else:
#         j_gen = None
#
#     if i_gen is None and j_gen is not None:
#         clusters_classified[i] = j_gen
#     elif i_gen is not None and j_gen is None:
#         clusters_classified[j] = i_gen
#     elif i_gen is not None and j_gen is not None:
#         if i_gen != j_gen:
#             print("end")
#             break
#     else:
#         print("else")
#         continue
#
# print(clusters_classified)

# print("Animate Distance")
# animate_distance_matrix(distance_matrices)
# print("Plot Dendrogram")
# plot_dendrogram(Z)
# print("Animate Dendrogram Threshold")
# animate_dendrogram_threshold(Z, X_scaled_sample)
# print("Animate Cluster Merging")
# animate_cluster_merging(Z, X_scaled_sample)

df_sample = df
random_states = [10, 20, 30, 40, 50]
linkage_methods = ['single', 'complete', 'average', 'centroid']


results = []

for rs in random_states:
    for i in range(2, len(numerical_features_full) + 1):
        for feature_pair in combinations(numerical_features_full, i):
            for linkage_method in linkage_methods:
                df_sample = df.sample(frac=0.1, random_state=rs)
                f = list(feature_pair)
                f.append('genres')
                values = df_sample[f].values
                X_pair = values[:, :-1]
                Y = values[:, -1]
                # X_pair = list(feature_pair)
                X_scaled_pair = scaler.fit_transform(X_pair)

                print("Started")
                Z_pair, _ = hierarchical_clustering(
                    X_scaled_pair, linkage=linkage_method)
                print("Finished")

                # Generar y guardar el dendrograma
                # plt.figure(figsize=(10, 7))
                # dendrogram(Z_pair.astype(float))
                # plt.title(f'Dendrograma para {feature_pair[0]} y {feature_pair[1]}')
                # plt.xlabel('Índices de las Muestras')
                # plt.ylabel('Distancia')
                # plt.savefig(
                #     f'hc_outputs/dendrogram_{feature_pair[0]}_{feature_pair[1]}.png')
                # plt.close()

                best_num_clusters = 2
                best_silhouette_score = -1
                best_davies_bouldin_score = float('inf')
                best_calinski_harabasz_score = -1

                # Try different numbers of clusters
                for num_clusters in range(2, 10):
                    clusters = fcluster(
                        Z_pair, num_clusters, criterion='maxclust')

                    # Calculate metrics
                    try:
                        silhouette = silhouette_score(X_scaled_pair, clusters)
                        davies_bouldin = davies_bouldin_score(
                            X_scaled_pair, clusters)
                        calinski_harabasz = calinski_harabasz_score(
                            X_scaled_pair, clusters)

                        # Update best scores based on silhouette score
                        if silhouette > best_silhouette_score:
                            best_silhouette_score = silhouette
                            best_num_clusters = num_clusters
                            best_davies_bouldin_score = davies_bouldin
                            best_calinski_harabasz_score = calinski_harabasz
                    except Exception as e:
                        # Code that runs if any exception occurs
                        print(f"An error occurred: {e}")

                print(f"Optimal clusters: {best_num_clusters}, Silhouette: {best_silhouette_score}, "
                      f"Davies-Bouldin: {best_davies_bouldin_score}, Calinski-Harabasz: {best_calinski_harabasz_score}")

                # Append result to list
                results.append({
                    'random_state':rs,
                    'i': i,
                    'feature_pair': feature_pair,
                    'linkage_method': linkage_method,
                    'best_num_clusters': best_num_clusters,
                    'best_silhouette_score': best_silhouette_score,
                    'best_davies_bouldin_score': best_davies_bouldin_score,
                    'best_calinski_harabasz_score': best_calinski_harabasz_score
                })

results_df = pd.DataFrame(results)
results_df.to_csv('hc_best_clusters.csv', index=False)
