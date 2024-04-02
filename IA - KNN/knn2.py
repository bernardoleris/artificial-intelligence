import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the Data
df = pd.read_csv("iris.csv")
df.head()

# Standardize the Variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Species', axis=1))
scaled_features = scaler.transform(df.drop('Species', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['Species'],
                                                    test_size=0.30, random_state=42)

## Using KNN
# Remember that we are trying to come up with a model to predict the species
# We'll start with k=1.

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

# Predicting and evaluations 
# Let's evaluate our knn model.

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))

error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# Here we can see that that after around K>3 the error rate tends to hover around 0.05-0.03

# Let's retrain the model with that and check the classification report!

# Retrain the model with K=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('WITH K=3')
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))
