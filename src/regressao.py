# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz 


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
y_pred_reg = model.predict(values) # Predição de y ou y chapeu

plt.plot( data["Cerveja"], data["Nota"], "o" ) # y verdadeiro
plt.plot( values, y_pred_reg, "-" ) # y estimado
plt.grid(True)
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.title("Relação Cerveja e Nota")
plt.legend(["Observações", "Modelo"])

# %%

# Ajustando árvore de decisão

# Importa modelos de árvore
from sklearn import tree

# Define um objeto de árvore de decisão
tree_reg_full = tree.DecisionTreeRegressor(criterion="mse")

tree_reg_2 = tree.DecisionTreeRegressor(criterion="mse",
                                        max_depth=2 )

X = data[ ["Cerveja"] ] # Isso retorn um DF
y = data["Nota"] # Isso retorna uma Series
tree_reg_full.fit(X,y) # Ajusta a árvore
tree_reg_2.fit(X,y) # Ajusta a árvore

values = [[i] for i in range(0,11)] # Novo X
y_pred_full = tree_reg_full.predict(values) # Predição de y ou y chapeu
y_pred_2 = tree_reg_2.predict(values) # Predição de y ou y chapeu

plt.plot( data["Cerveja"], data["Nota"], "o" ) # y verdadeiro
plt.plot( values, y_pred_full, "-" ) # y estimado
plt.plot( values, y_pred_2, "-" ) # y estimado
plt.grid(True)
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.title("Relação Cerveja e Nota")
plt.legend(["Observações", "Árvore Full", "Árvore max_depth=2"])

# %% 

## Métricas de ajuste de modelos de Regressão

from sklearn import metrics


pred_y_reg = model.predict( data[["Cerveja"]] )
pred_y_arvore_full = tree_reg_full.predict( data[["Cerveja"]] )
pred_y_arvore_2 = tree_reg_2.predict( data[["Cerveja"]] )

mse_regressao = metrics.mean_squared_error(y, pred_y_reg)
mse_arvore_full = metrics.mean_squared_error(y, pred_y_arvore_full)
mse_arvore_2 = metrics.mean_squared_error(y, pred_y_arvore_2)

print("MSE Regressão Linear:", mse_regressao)
print("MSE Árvore Full:", mse_arvore_full)
print("MSE Árvore max_depth = 2:", mse_arvore_2)

# %%

mae_regressao = metrics.mean_absolute_error(y, pred_y_reg)
mae_arvore_full = metrics.mean_absolute_error(y, pred_y_arvore_full)
mae_arvore_2 = metrics.mean_absolute_error(y, pred_y_arvore_2)

print("MAE Regressão Linear:", mae_regressao)
print("MAE Árvore Full:", mae_arvore_full)
print("MAE Árvore max_depth = 2:", mae_arvore_2)

# %%
r2_regressao = metrics.r2_score(y, pred_y_reg)
r2_arvore_full = metrics.r2_score(y, pred_y_arvore_full)
r2_arvore_2 = metrics.r2_score(y, pred_y_arvore_2)

print("R2 Regressão Linear:", r2_regressao)
print("R2 Árvore Full:", r2_arvore_full)
print("R2 Árvore max_depth = 2:", r2_arvore_2)

# %%

# Plot da árvore de Decisão
classes = data["Nota"].unique()
classes.sort()

dot_data = tree.export_graphviz(tree_reg_full, out_file=None, 
                     feature_names=["Cerveja"],  
                     class_names=classes,  
                     filled=True, rounded=True,  
                     special_characters=True)

graph = graphviz.Source(dot_data)  
graph 

# %%