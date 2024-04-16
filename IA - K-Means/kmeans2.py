import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

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
    
    return silhouette_avg

# Carregar a base de dados Iris
iris = load_iris()
data = iris.data

# Especificar os números de agrupamentos (K)
ks = [3, 5]

# Executar o algoritmo KMeans para cada valor de K
for k in ks:
    silhouette_avg = kmeans(data, k)
    print("Silhouette Score para k =", k, ":", silhouette_avg)
