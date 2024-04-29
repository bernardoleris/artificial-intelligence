import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import psutil

# Função para calcular o uso de memória
def memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024  # em megabytes
    return f"Uso de memória: {mem:.2f} MB"

# Função para plotar a matriz de confusão
def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2. else "black")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Carregando os conjuntos de dados Iris e Wine
iris = datasets.load_iris()
wine = datasets.load_wine()

# Dividindo os conjuntos de dados em features (X) e labels (y)
X_iris, y_iris = iris.data, iris.target
X_wine, y_wine = wine.data, wine.target

# Dividindo os conjuntos de dados em conjuntos de treinamento e teste (80% para treinamento, 20% para teste)
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

# Padronizando os conjuntos de dados
scaler = StandardScaler()
X_train_iris_std = scaler.fit_transform(X_train_iris)
X_test_iris_std = scaler.transform(X_test_iris)

X_train_wine_std = scaler.fit_transform(X_train_wine)
X_test_wine_std = scaler.transform(X_test_wine)

# Começa a contar o tempo
start_time = time.time()

# Criando e treinando o Perceptron para o conjunto de dados Iris
perceptron_iris = Perceptron(max_iter=100, eta0=0.1, random_state=42)
perceptron_iris.fit(X_train_iris_std, y_train_iris)

# Criando e treinando o Perceptron para o conjunto de dados Wine
perceptron_wine = Perceptron(max_iter=100, eta0=0.1, random_state=42)
perceptron_wine.fit(X_train_wine_std, y_train_wine)

# Prevendo os rótulos para o conjunto de dados de teste
y_pred_iris = perceptron_iris.predict(X_test_iris_std)
y_pred_wine = perceptron_wine.predict(X_test_wine_std)

# Calculando as métricas de avaliação para o conjunto de dados Iris
print("Métricas de Avaliação para o conjunto de dados Iris:\n", classification_report(y_test_iris, y_pred_iris, target_names=iris.target_names))

# Calculando as métricas de avaliação para o conjunto de dados Wine
print("Métricas de Avaliação para o conjunto de dados Wine:\n", classification_report(y_test_wine, y_pred_wine, target_names=wine.target_names))

# Termina de contar o tempo
end_time = time.time()

# Calcula o tempo total de execução
execution_time = end_time - start_time

# Obtém o uso de memória
memory = memory_usage()

print("\nTempo de execução total:", execution_time, "segundos")
print(memory)

# Plotando a matriz de confusão para o conjunto de dados Iris
plot_confusion_matrix(y_test_iris, y_pred_iris, iris.target_names, title="Matriz de Confusão - Iris")

# Plotando a matriz de confusão para o conjunto de dados Wine
plot_confusion_matrix(y_test_wine, y_pred_wine, wine.target_names, title="Matriz de Confusão - Wine")
