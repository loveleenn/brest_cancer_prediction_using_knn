import numpy as np
import pandas as pd
data = pd.read_csv("/content/data-2.csv")
print(data)

data = data.drop('id', axis=1)
data= data.drop('Unnamed: 32', axis= 1)
print(data)

data.isnull().sum()

data.shape

data.info(10)

from sklearn. model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,1:], data.iloc[:,0], test_size=0.4, random_state=2)

data.shape

X_train

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)

for i in range (1,16):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  print(i, accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt

neighbors = list(range(1, 16))
accuracy_scores = [0.92, 0.93, 0.94, 0.93, 0.95, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83]

plt.plot(neighbors, accuracy_scores)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy Score')
plt.title('Accuracy vs. Number of Neighbors')
plt.show()
