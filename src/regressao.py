# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
np.random.seed(28)

data = [ (i, np.random.normal(i, 2)) for i in np.random.choice( list(range(1,10)), 15, replace=True ) ]

data = pd.DataFrame(data, columns = ["Cerveja", "Nota"])
data['Nota'] = data['Nota'].abs()
data["Nota"] = data["Nota"].apply(lambda x: x if x < 10 else 10)

# Fazer os modelos na aula!!!!

plt.plot( data["Cerveja"], data["Nota"], "o" )
plt.grid(True)
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.title("Relação Cerveja e Nota")

# %%

# CRIANDO UM MODELO DE REGRESSÃO LINEAR

from sklearn import linear_model

# Esse é o nome modelo de regressão linear
model = linear_model.LinearRegression(fit_intercept=True)

# %%

# Vamos ajustar o modelo

X = data[ ["Cerveja"] ] # Isso retorn um DF
y = data["Nota"] # Isso retorna uma Series
model.fit(X, y) # Ajuste dos parâmetros beta0 e beta1

# %%
b0, b1 = model.intercept_, model.coef_[0]

print(f" Beta0 = {b0} | Beta1 = {b1}")

# %%

values = [[i] for i in range(0,11)] # Novo X
y_pred = model.predict(values) # Predição de y ou y chapeu

plt.plot( data["Cerveja"], data["Nota"], "o" ) # y verdadeiro
plt.plot( values, y_pred, "-" ) # y estimado
plt.grid(True)
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.title("Relação Cerveja e Nota")
plt.legend(["Observações", "Modelo"])