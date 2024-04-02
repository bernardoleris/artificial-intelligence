import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

# Leitura dos dados
iris_data = pd.read_csv('iris.csv')
iris_data.columns = ['id', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Função para classificação utilizando k-NN
def knn(train_data, test_instance, k):
    distances = []
    for index, row in train_data.iterrows():
        train_instance = row.iloc[:-1]
        label = row.iloc[-1]
        distance = euclidean_distance(test_instance, train_instance)
        distances.append((distance, label))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    labels = [neighbor[1] for neighbor in neighbors]
    most_common = Counter(labels).most_common(1)
    return most_common[0][0]

# Função para calcular a taxa de reconhecimento para diferentes valores de k
def calculate_accuracy(train_data, test_data, k):
    correct_predictions = 0
    total_predictions = len(test_data)
    for index, row in test_data.iterrows():
        test_instance = row.iloc[:-1]
        true_label = row.iloc[-1]
        predicted_label = knn(train_data, test_instance, k)
        if predicted_label == true_label:
            correct_predictions += 1
    accuracy = correct_predictions / total_predictions
    return accuracy

# Dividindo os dados em conjunto de treino e teste
np.random.seed(42)
iris_data = iris_data.sample(frac=1).reset_index(drop=True)  # Embaralhando os dados
train_size = int(0.8* len(iris_data))
train_data = iris_data[:train_size]
test_data = iris_data[train_size:]

# Testando o classificador para diferentes valores de k
k_values = [1, 3, 5, 7]
best_accuracy = 0
best = 0
for k in k_values:
    accuracy = calculate_accuracy(train_data, test_data, k)
    if(accuracy > best_accuracy):
        best_accuracy = accuracy
        best = k
    print(f'Accuracy for k = {k}: {accuracy}')

# Matriz de confusão e métricas de avaliação
predictions = []
true_labels = []
for index, row in test_data.iterrows():
    test_instance = row.iloc[:-1]
    true_label = row.iloc[-1]
    predicted_label = knn(train_data, test_instance, best)  
    predictions.append(predicted_label)
    true_labels.append(true_label)

conf_matrix = confusion_matrix(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
accuracy = accuracy_score(true_labels, predictions)

print(best)
print('\nConfusion Matrix:')
print(conf_matrix)
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')
