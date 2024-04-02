from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Carregamento do dataset Iris
iris = load_iris()

# Separando features e target
X = iris.data
y = iris.target

# Definição dos parâmetros
k_value = 1  # Defina o valor de K desejado
random_seed = 42  # Defina a semente aleatória (random seed)
test_size = 0.20  # Defina a porcentagem de dados para teste

# Dividindo os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

# Instanciando o modelo KNN com o valor de K definido
model = KNeighborsClassifier(n_neighbors=k_value)

# Treinando o modelo utilizando o conjunto de treino
model.fit(X_train, y_train)

# Validando o modelo utilizando o conjunto de teste
accuracy = str(round(model.score(X_test, y_test) * 100, 2)) + "%"

# Imprimindo o resultado
print("A acurácia do modelo KNN com K =", k_value, "foi", accuracy)
