import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Carregar a base de dados iris
iris = load_iris()
X = iris.data

# Definir o número de clusters (k=3 e k=5)
k_values = [3, 5]

# Iterar sobre os valores de k
for k in k_values:
    # Criar o modelo KMeans
    kmeans = KMeans(n_clusters=k)
    
    # Treinar o modelo
    kmeans.fit(X)
    
    # Fazer previsões de cluster
    labels = kmeans.labels_
    
    # Calcular o Silhouette Score
    silhouette_avg = silhouette_score(X, labels)
    
    # Imprimir o resultado
    print(f"Para k = {k}, o Silhouette Score é {silhouette_avg}")
