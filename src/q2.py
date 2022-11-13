import pandas as pd
import numpy as np

# Obtendo dados
df = pd.read_csv("data/class02.csv")
x = df.drop('target', axis = 1).to_numpy()
y = df['target'].to_numpy()

# Validação K-fold
from sklearn.model_selection import KFold
kf = KFold(5)

# Modelo KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(10)

acuracia_treino, acuracia_validacao = 0, 0
for train, test in kf.split(x):
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]

    knn.fit(x_train, y_train)

    acuracia_treino += knn.score(x_train, y_train)
    acuracia_validacao += knn.score(x_test, y_test)

acuracia_treino /= kf.get_n_splits(x)
acuracia_validacao /= kf.get_n_splits(x)

print('Acurácia na base de treino:', acuracia_treino)
print('Acurácia na base de validação:', acuracia_validacao)