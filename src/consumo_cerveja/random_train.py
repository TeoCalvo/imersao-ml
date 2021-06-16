# %%

from joblib.parallel import DEFAULT_THREAD_BACKEND
import pandas as pd
from scipy.sparse.construct import random
from sklearn import tree
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics

# %%

df = pd.read_csv("../../data/Consumo_cerveja.csv",
                 decimal=",")

df.head()

# %%

df["Data"] = pd.to_datetime( df["Data"] )

df_oot = df[ df['Data'] >= "2015-12-01" ].copy()
df_treino = df[ df['Data'] < "2015-12-01" ].copy()

features = df.columns.tolist()[1:-1]
target = df.columns.tolist()[-1]

# %%

X_train, X_test, y_train, y_test = model_selection.train_test_split(df_treino[features],
                                                                    df_treino[target],
                                                                    random_state=42,
                                                                    test_size=0.1)

tree_reg = tree.DecisionTreeRegressor()

random_params = {
    "criterion": ['mae', 'mse'],
    "max_depth": list(range(1,100)),
    "min_samples_split": list(range(1,150))
}

random_search = model_selection.RandomizedSearchCV( tree_reg,
                                                    random_params,
                                                    n_iter = 10000, # Sorteio aleatório de 50 combinações
                                                    scoring='neg_mean_absolute_error',
                                                    n_jobs=-1,
                                                    cv=3,
                                                    random_state=42 )

random_search.fit( X_train, y_train )
# %%

random_search.best_params_

# %%

pred_test = random_search.predict( X_test )
print( metrics.mean_absolute_error(y_test, pred_test) )

# %%

model = tree.DecisionTreeRegressor( **random_search.best_params_ )
model.fit( df_treino[features], df_treino[target] )

# %%

pred_oot = random_search.predict( df_oot[features] )
print( metrics.mean_absolute_error(df_oot[target], pred_oot) )
