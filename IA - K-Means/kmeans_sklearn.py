import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from memory_profiler import memory_usage

start_time = time.time()  # Iniciar contagem de tempo
mem_usage_before = memory_usage()  # Uso de memória antes da execução

# Carregar a base de dados iris
iris = load_iris()
X = iris.data

# Definir os valores de k
k_values = [3, 5]

# Lista para armazenar os valores do Silhouette Score
silhouette_scores = []

print("Algoritmo já implementado:")
# Iterar sobre os valores de k
for k in k_values:
    # Criar o modelo KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    
    # Treinar o modelo
    kmeans.fit(X)
    
    # Fazer previsões de cluster
    labels = kmeans.labels_
    
    # Calcular o Silhouette Score
    silhouette_avg = silhouette_score(X, labels)
    
    # Armazenar o Silhouette Score
    silhouette_scores.append(silhouette_avg)
    
    # Imprimir o resultado
    print(f"Para k = {k}, o Silhouette Score é {silhouette_avg}")

# Encontrar o melhor valor de k
best_k_index = np.argmax(silhouette_scores)
best_k = k_values[best_k_index]

print("Melhor valor de k:", best_k)

# Medir o tempo de execução
end_time = time.time()
execution_time = end_time - start_time
print("Tempo de execução:", execution_time, "segundos")

# Medir o uso de memória após a execução
mem_usage_after = memory_usage()
mem_diff = mem_usage_after[0] - mem_usage_before[0]
print("Uso de memória:", mem_diff, "MB")

# Aplicar PCA com 1 componente
pca_1 = PCA(n_components=1)
X_pca_1 = pca_1.fit_transform(X)

# Criar o modelo KMeans com o melhor k para 1 componente
kmeans_best_1 = KMeans(n_clusters=best_k, random_state=42)
kmeans_best_1.fit(X_pca_1)

# Plotar clusters e centróides para 1 componente
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_1, np.zeros_like(X_pca_1), c=kmeans_best_1.labels_, cmap='viridis')
plt.scatter(kmeans_best_1.cluster_centers_, np.zeros_like(kmeans_best_1.cluster_centers_), marker='x', color='red', s=200)
plt.title(f'Clusters e Centróides (k={best_k}, 1 componente)')
plt.xlabel('Componente Principal 1')
plt.yticks([])
plt.show()

# Aplicar PCA com 2 componentes
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X)

# Criar o modelo KMeans com o melhor k para 2 componentes
kmeans_best_2 = KMeans(n_clusters=best_k, random_state=42)
kmeans_best_2.fit(X_pca_2)

# Plotar clusters e centróides para 2 componentes
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=kmeans_best_2.labels_, cmap='viridis')
plt.scatter(kmeans_best_2.cluster_centers_[:, 0], kmeans_best_2.cluster_centers_[:, 1], marker='x', color='red', s=200)
plt.title(f'Clusters e Centróides (k={best_k}, 2 componentes)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()
