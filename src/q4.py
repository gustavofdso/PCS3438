import pandas as pd
import numpy as np

# Obtendo dados
df = pd.read_csv("data/reg02.csv")
x = df.drop('target', axis = 1).to_numpy()
y = df['target'].to_numpy()

# Validação K-fold
from sklearn.model_selection import KFold
kf = KFold(5)

# Modelo árvore de regressão
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(criterion = 'absolute_error')

mae_treino, mae_validacao = 0, 0
from sklearn.metrics import mean_absolute_error
for train, test in kf.split(x):
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]

    tree.fit(x_train, y_train)

    mae_treino += mean_absolute_error(y_train, tree.predict(x_train))
    mae_validacao += mean_absolute_error(y_test, tree.predict(x_test))

mae_treino /= kf.get_n_splits(x)
mae_validacao /= kf.get_n_splits(x)

print('MAE na base de treino:', mae_treino)
print('MAE na base de validação:', mae_validacao)