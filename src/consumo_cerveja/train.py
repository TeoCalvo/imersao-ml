# %%
from numpy import object0
import pandas as pd

# %%

df = pd.read_csv("../../data/Consumo_cerveja.csv",
                 decimal=",")

df.head()

# %%

target = "Consumo de cerveja (litros)"
features = df.columns.tolist()[1:-1]

y = df[target]
X = df[features]

# %%

print("y:", y)
print("X:", X)

# %%

from sklearn import tree

# Ajuste do modelo preditivo
model = tree.DecisionTreeRegressor(max_depth=5)
model.fit(X, y)

# %%
y_pred = model.predict(X)

# %%
from sklearn import metrics

print("MAE:", metrics.mean_absolute_error(y, y_pred))
print("MSE:", metrics.mean_squared_error(y, y_pred))
print("R2:", metrics.r2_score(y, y_pred))

# %%

from sklearn import model_selection

# Dividir base de treino e de teste
X_train, X_test, y_train, y_test = model_selection.train_test_split( X,
                                                                     y,
                                                                     random_state=42,
                                                                     test_size=0.2 )

# Ajuste do modelo preditivo
model = tree.DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)

# %%
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("MAE TREINO:", metrics.mean_absolute_error(y_train, y_pred_train))
print("MSE TREINO:", metrics.mean_squared_error(y_train, y_pred_train))
print("R2 TREINO:", metrics.r2_score(y_train, y_pred_train))

print("\n\nMAE TEST:", metrics.mean_absolute_error(y_test, y_pred_test))
print("MSE TEST:", metrics.mean_squared_error(y_test, y_pred_test))
print("R2 TEST:", metrics.r2_score(y_test, y_pred_test))


# %%

df_train = df.iloc[:-30].copy()
df_oot = df.iloc[-30:].copy()

X_train, X_test, y_train, y_test =  model_selection.train_test_split(df_train[features],
                                                                     df_train[target],
                                                                     random_state = 42,
                                                                     test_size = 0.1 )

# Ajuste do modelo preditivo
model = tree.DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)

# %%
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_oot = model.predict(df_oot[features])

print("MAE TREINO:", metrics.mean_absolute_error(y_train, y_pred_train))
print("MSE TREINO:", metrics.mean_squared_error(y_train, y_pred_train))
print("R2 TREINO:", metrics.r2_score(y_train, y_pred_train))

print("\nMAE TEST:", metrics.mean_absolute_error(y_test, y_pred_test))
print("MSE TEST:", metrics.mean_squared_error(y_test, y_pred_test))
print("R2 TEST:", metrics.r2_score(y_test, y_pred_test))

print("\nMAE OOT:", metrics.mean_absolute_error(df_oot[target], y_pred_oot))
print("MSE OOT:", metrics.mean_squared_error(df_oot[target], y_pred_oot))
print("R2 OOT:", metrics.r2_score(df_oot[target], y_pred_oot))

# %%

import numpy as np

df_train["particao"] = np.random.choice( [1,2,3,4],
                                         df_train.shape[0] )

metrica_treino = []
metrica_test = []

for i in df_train["particao"].unique():

    X_test = df_train[ df_train["particao"] == i ][features]
    X_train = df_train[ df_train["particao"] != i ][features]

    y_test = df_train[ df_train["particao"] == i ][target]
    y_train = df_train[ df_train["particao"] != i ][target]

    # Ajuste do modelo preditivo
    model = tree.DecisionTreeRegressor(max_depth=5)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    values = df_train["particao"].unique().tolist()
    values.remove(i)
    print(f"\nTreino: {values}, Teste: [{i}] ")
    
    metrica_treino.append( metrics.mean_absolute_error(y_train, y_pred_train) )
    metrica_test.append( metrics.mean_absolute_error(y_test, y_pred_test) )
    
    print("MAE TREINO:", metrics.mean_absolute_error(y_train, y_pred_train))
    print("MAE TEST:", metrics.mean_absolute_error(y_test, y_pred_test))

print("\nAVG MAE Treino:", np.mean(metrica_treino) )
print("AVG MAE Treino:", np.mean(metrica_test) )
print("STD MAE Treino:", np.std(metrica_treino) )
print("STD MAE Treino:", np.std(metrica_test) )

# %%

model = tree.DecisionTreeRegressor(max_depth=4)

cv_result = model_selection.cross_validate( model,
                                            df_train[features],
                                            df_train[target],
                                            cv=4,
                                            return_train_score=True,
                                            scoring='neg_mean_absolute_error' )

cv_result = pd.DataFrame( cv_result )

cv_result.mean().abs()

# %%

# FLUXO DEFINITIVO

df_oot = df.iloc[:-30]
df_train = df.iloc[-30:]

target = "Consumo de cerveja (litros)"
features = df.columns.tolist()[1:-1]

model = tree.DecisionTreeRegressor(max_depth=4)