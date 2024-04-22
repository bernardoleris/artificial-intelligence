import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Passo 1: Especifique K, o número de agrupamentos, e os centros iniciais Ck(k=1,...,K)
def initialize_centers(data, k):
    # Inicialize os centros aleatoriamente
    random_indices = np.random.choice(data.shape[0], k, replace=False)
    centers = data[random_indices]
    return centers

# Passo 2: Atualize os conjuntos Sk(k=1,...K)
def assign_clusters(data, centers):
    clusters = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        # Calcule a distância do ponto aos centros e atribua ao cluster mais próximo
        distances = np.linalg.norm(data[i] - centers, axis=1)
        cluster_index = np.argmin(distances)
        clusters[i] = cluster_index
    return clusters

# Passo 3: Atualize os centros Ck(k=1,...K)
def update_centers(data, clusters, centers, k):
    new_centers = np.zeros((k, data.shape[1]))
    for i in range(k):
        # Calcule a média dos pontos em cada cluster para obter os novos centros
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centers[i] = np.mean(cluster_points, axis=0)
        else:
            # Se um cluster estiver vazio, mantenha o centro anterior
            new_centers[i] = centers[i]
    return new_centers

# Passo 4: Verifique se os novos centros coincidem com os anteriores
def centers_converged(centers, new_centers):
    return np.array_equal(centers, new_centers)

# Função principal do KMeans
def kmeans(data, k, max_iterations=100):
    # Passo 1: Inicialização dos centros
    centers = initialize_centers(data, k)
    
    for _ in range(max_iterations):
        # Passo 2: Atribuição dos clusters
        clusters = assign_clusters(data, centers)
        
        # Passo 3: Atualização dos centros
        new_centers = update_centers(data, clusters, centers, k)
        
        # Passo 4: Verificação de convergência
        if centers_converged(centers, new_centers):
            break
        
        centers = new_centers
    
    # Calcular o Silhouette Score
    silhouette_avg = silhouette_score(data, clusters)
    
    return silhouette_avg, clusters, centers

# Carregar a base de dados Iris
iris = load_iris()
data = iris.data

# Especificar os números de agrupamentos (K)
ks = [3, 5]

best_k = None
best_silhouette = -1

# Executar o algoritmo KMeans para cada valor de K
for k in ks:
    silhouette_avg, _, _ = kmeans(data, k)
    print("Silhouette Score para k =", k, ":", silhouette_avg)
    
    if silhouette_avg > best_silhouette:
        best_silhouette = silhouette_avg
        best_k = k

print("Melhor valor de K:", best_k)

# Aplicar PCA com 1 componente
pca_1 = PCA(n_components=1)
X_pca_1 = pca_1.fit_transform(data)

# Criar o modelo KMeans com o melhor k para 1 componente
best_silhouette_1, clusters_1, centers_1 = kmeans(X_pca_1, best_k)

# Plotar clusters e centróides para 1 componente
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_1, np.zeros_like(X_pca_1), c=clusters_1, cmap='viridis')
plt.scatter(centers_1, np.zeros_like(centers_1), marker='x', color='red', s=200)
plt.title(f'Clusters e Centróides (k={best_k}, 1 componente)')
plt.xlabel('Componente Principal 1')
plt.yticks([])
plt.show()

# Aplicar PCA com 2 componentes
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(data)

# Criar o modelo KMeans com o melhor k para 2 componentes
best_silhouette_2, clusters_2, centers_2 = kmeans(X_pca_2, best_k)

# Plotar clusters e centróides para 2 componentes
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=clusters_2, cmap='viridis')
plt.scatter(centers_2[:, 0], centers_2[:, 1], marker='x', color='red', s=200)
plt.title(f'Clusters e Centróides (k={best_k}, 2 componentes)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()
