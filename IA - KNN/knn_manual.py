from sklearn.datasets import load_iris
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import time

# Começa a contar o tempo
start_time = time.time()

# Carrega o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Define a função de distância euclidiana
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Define a função do algoritmo KNN
def knn(train_data, test_instance, k):
    distances = []
    for train_instance, label in train_data:
        distance = euclidean_distance(test_instance, train_instance)
        distances.append((distance, label))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    labels = [neighbor[1] for neighbor in neighbors]
    most_common = Counter(labels).most_common(1)
    return most_common[0][0]

# Define a função para calcular a acurácia
def calculate_accuracy(train_data, test_data, k):
    correct_predictions = 0
    total_predictions = len(test_data)
    for test_instance, true_label in test_data:
        predicted_label = knn(train_data, test_instance, k)
        if predicted_label == true_label:
            correct_predictions += 1
    accuracy = correct_predictions / total_predictions
    return accuracy

# Divisão dos dados em conjunto de treinamento e teste
np.random.seed(42)
data = list(zip(X, y))
np.random.shuffle(data)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Valores de k a serem testados
k_values = [1, 3, 5, 7]
best_accuracy = 0
best = 0

for k in k_values:
    print(f"\n--- KNN with k = {k} ---")
    
    accuracy = calculate_accuracy(train_data, test_data, k)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best = k
    
    print(f"Accuracy for k = {k}: {accuracy}")
    
    predictions = []
    true_labels = []
    for test_instance, true_label in test_data:
        predicted_label = knn(train_data, test_instance, k)  
        predictions.append(predicted_label)
        true_labels.append(true_label)
        
    conf_matrix = confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    accuracy = accuracy_score(true_labels, predictions)

    print(f'Average Precision: {precision}')
    print(f'Average Recall: {recall}')

# Termina de contar o tempo
end_time = time.time()

# Calcula o tempo total de execução
execution_time = end_time - start_time

print("Tempo de execução total:", execution_time, "segundos")
