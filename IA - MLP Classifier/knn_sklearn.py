import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import psutil

# Função para imprimir métricas de avaliação
def print_evaluation_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    conf_matrix = confusion_matrix(y_true, y_pred)
    
    report = classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(report)

# Função para calcular o uso de memória
def memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 * 1024)  # em megabytes
    return f"Uso de memória: {mem:.2f} MB"

# Começa a contar o tempo
start_time = time.time()

# Carregando os conjuntos de dados Iris e Wine
iris = load_iris()
wine = load_wine()

datasets = [(iris, "Iris"), (wine, "Wine")]

k_values = [1, 3, 5, 7]  
random_seed = 42  
test_size = 0.20  

for dataset, name in datasets:
    X = dataset.data
    y = dataset.target

    print(f"\n--- {name} Dataset ---")

    best_accuracy = 0
    best_k = None

    for k_value in k_values:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        model = KNeighborsClassifier(n_neighbors=k_value)

        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k_value

    print(f"\nMetrics for the best k ({best_k}):")

    X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_train_best, y_train_best)
    print_evaluation_metrics(y_test_best, best_model.predict(X_test_best))

# Termina de contar o tempo
end_time = time.time()

# Calcula o tempo total de execução
execution_time = end_time - start_time

# Obtém o uso de memória
memory = memory_usage()

print("\nTempo de execução total:", execution_time, "segundos")
print(memory)
