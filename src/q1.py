import pandas as pd
import numpy as np

# Obtendo dados
df = pd.read_csv("data/class01.csv")
x = df.drop('target', axis = 1).to_numpy()
y = df['target'].to_numpy()

# Validação hold-out
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 350, shuffle = False)

# Modelo Naive Bayes Gaussiano
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

nb.fit(x_train, y_train)

acuracia_treino = nb.score(x_train, y_train)
acuracia_validacao = nb.score(x_test, y_test)

print('Acurácia na base de treino:', acuracia_treino)
print('Acurácia na base de validação:', acuracia_validacao)