import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar os dados do arquivo CSV
iris_data = pd.read_csv('iris.csv')

# Dividir os dados em recursos (X) e rótulos (y)
X = iris_data.drop('Species', axis=1)
y = iris_data['Species']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir os valores de k a serem testados
k_values = [1, 3, 5, 7]

# Testar o modelo com diferentes valores de k
for k in k_values:
    # Inicializar e ajustar o modelo KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = knn.predict(X_test)

    # Calcular a acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy for k =", k, ":", accuracy)

