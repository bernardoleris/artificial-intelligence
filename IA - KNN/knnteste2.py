from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

iris = load_iris()

X = iris.data
y = iris.target

k_values = [1, 3, 5, 7]  
random_seed = 42  
test_size = 0.20  

for k_value in k_values:
    print(f"\n--- KNN with k = {k_value} ---")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    model = KNeighborsClassifier(n_neighbors=k_value)

    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy for k = {k_value}: {accuracy}")

    conf_matrix = confusion_matrix(y_test, model.predict(X_test))
    print("\nConfusion Matrix:")
    print(conf_matrix)

    precision = precision_score(y_test, model.predict(X_test), average='weighted')
    print(f"Average Precision: {precision}")

    recall = recall_score(y_test, model.predict(X_test), average='weighted')
    print(f"Average Recall: {recall}")

