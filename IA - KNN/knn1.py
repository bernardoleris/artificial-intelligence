import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def euclidean_distance(vec1, vec2):
    distance = 0
    for i in range(len(vec1)):
        distance += ((vec1[i] - vec2[i]) ** 2)
    euclidean_distance = np.sqrt(distance)
    return euclidean_distance

def classify(X_train, X_test, Y_train, k):
    classifications = []
    for i in range(len(X_test)):
        distances = []
        for j in range(len(X_train)):
            dist = euclidean_distance(X_test[i], X_train[j])
            dist_label = [dist, Y_train[j]]
            distances.append(dist_label)
        distances.sort()
        nearest = distances[:k]
        classifications.append(vote(nearest))
    return classifications

def vote(nearest_list):
    results = []
    for i in range(len(nearest_list)):
        results.append(nearest_list[i][1])
    classification = max(set(results), key=results.count)
    return classification

def accuracy(yhat, y):
    correct = sum(1 for pred, true in zip(yhat, y) if pred == true)
    total = len(y)
    acc = correct / total
    return acc

def precision_recall(yhat, y, class_name):
    tp = sum((yhat[i] == class_name) and (y[i] == class_name) for i in range(len(yhat)))
    fp = sum((yhat[i] == class_name) and (y[i] != class_name) for i in range(len(yhat)))
    fn = sum((yhat[i] != class_name) and (y[i] == class_name) for i in range(len(yhat)))
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    return precision, recall

def average_precision_recall(predictions, Y_test):
    classes = set(Y_test)
    precisions = []
    recalls = []
    for class_name in classes:
        precision, recall = precision_recall(predictions, Y_test, class_name)
        precisions.append(precision)
        recalls.append(recall)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    return avg_precision, avg_recall

iris_data = pd.read_csv('iris.csv')
iris_data.columns = ['id', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

X = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
Y = iris_data['class'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k_values = [1, 3, 5, 7]
for k in k_values:
    predictions = classify(X_train, X_test, Y_train, k)

    acc = accuracy(predictions, Y_test)
    print(f'Accuracy for k = {k}: {acc}')

    avg_precision, avg_recall = average_precision_recall(predictions, Y_test)
    print(f'Average Precision: {avg_precision}')
    print(f'Average Recall: {avg_recall}')

    conf_matrix = pd.crosstab(pd.Series(Y_test, name='Actual'), pd.Series(predictions, name='Predicted'))
    print('\nConfusion Matrix:')
    print(conf_matrix)
    print('\n')
