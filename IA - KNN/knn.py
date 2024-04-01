import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

# Carregar os dados da base de dados Iris
iris_data = pd.read_csv('iris.csv')

# Pré-processamento dos dados
X = iris_data.drop('Species', axis=1).values  # Recursos (convertendo para array NumPy)
y = iris_data['Species'].values  # Rótulos (convertendo para array NumPy)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir função para calcular a distância Euclidiana
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Definir função para o algoritmo K-Nearest Neighbors
def knn_predict(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        distance = euclidean_distance(X_train[i], x_test)
        distances.append((distance, y_train[i]))
    distances = sorted(distances)[:k]
    targets = [item[1] for item in distances]
    return max(set(targets), key=targets.count)

# Definir função para avaliar a taxa de reconhecimento para um valor de k específico
def calculate_accuracy(X_train, y_train, X_test, y_test, k):
    predictions = []
    for i in range(len(X_test)):
        predictions.append(knn_predict(X_train, y_train, X_test[i], k))
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Definir os valores de k a serem testados
k_values = [1, 3, 5, 7]

# Armazenar os resultados de desempenho para cada valor de k
results = {}

# Calcular e armazenar a taxa de reconhecimento para cada valor de k
for k in k_values:
    accuracy = calculate_accuracy(X_train, y_train, X_test, y_test, k)
    results[k] = accuracy
    print("Accuracy for k =", k, ":", accuracy)     

# Escolher o melhor valor de k baseado na maior taxa de reconhecimento
best_k = max(results, key=results.get)
print("Best k:", best_k)
print("Accuracy for best k:", results[best_k])

# Plotar a matriz de confusão e métricas de avaliação para o melhor valor de k
y_pred = []
for i in range(len(X_test)):
    y_pred.append(knn_predict(X_train, y_train, X_test[i], best_k))

# Plotar a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=iris_data['Species'].unique(), yticklabels=iris_data['Species'].unique())
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Calcular e mostrar métricas de avaliação para o melhor valor de k
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
