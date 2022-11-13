import pandas as pd
import numpy as np

# Obtendo dados
df = pd.read_csv("data/reg01.csv")
x = df.drop('target', axis = 1).to_numpy()
y = df['target'].to_numpy()

# Validação Leave-One-Out
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

# Modelo LASSO
from sklearn.linear_model import Lasso
l = Lasso(1)

rmse_treino, rmse_validacao = 0, 0
from sklearn.metrics import mean_squared_error
for train, test in loo.split(x):
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]

    l.fit(x_train, y_train)

    rmse_treino += np.sqrt(mean_squared_error(y_train, l.predict(x_train)))
    rmse_validacao += np.sqrt(mean_squared_error(y_test, l.predict(x_test)))

rmse_treino /= loo.get_n_splits(x)
rmse_validacao /= loo.get_n_splits(x)

print('RMSE na base de treino:', rmse_treino)
print('RMSE na base de validação:', rmse_validacao)