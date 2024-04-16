from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
import numpy as np

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data  # Características (features)

class KMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Inicialização aleatória dos centróides
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Passo 1: Atribuição dos pontos aos clusters
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Passo 2: Atualização dos centróides
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Verifica convergência
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        self.labels_ = labels

# Executando o algoritmo K-Means com 3 e 5 clusters
for n_clusters in [3, 5]:
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    
    # Calculando o silhouette score
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print(f"Para k = {n_clusters}, o Silhouette Score é {silhouette_avg}")
