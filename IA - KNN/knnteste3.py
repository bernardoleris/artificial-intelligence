import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import numpy as np
import time

# Começa a contar o tempo
start_time = time.time()

# Carregamento dos dados
df = pd.read_csv("iris.csv")

# Plot scatter plot
sns.scatterplot(x=df['SepalLengthCm'], y=df['SepalWidthCm'], hue=df['Species'])

# Dividindo os dados em features (X) e target (y)
X = df.drop('Species', axis=1)
y = df['Species']

# Definindo a semente aleatória
np.random.seed(42)

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Escalonando as features usando StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Valores de k a serem testados
k_values = [1, 3, 5, 7]

for k in k_values:
    print(f"\n--- KNN with k = {k} ---")
    
    # Inicializando e treinando o classificador KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Prevendo os dados de teste
    y_pred = knn.predict(X_test)

    # Calculando a acurácia do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy for k =", k, ":", accuracy)

    # Matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Precision Score
    precision = precision_score(y_test, y_pred, average='weighted')
    print("Average Precision:", precision)

    # Recall Score
    recall = recall_score(y_test, y_pred, average='weighted')
    print("Average Recall:", recall)

# Termina de contar o tempo
end_time = time.time()

# Calcula o tempo total de execução
execution_time = end_time - start_time

print("Tempo de execução total:", execution_time, "segundos")
