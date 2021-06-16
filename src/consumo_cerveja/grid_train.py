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

grid_params = {
    "criterion": ['mae', 'mse'],
    "max_depth": [2,3,4],
    "min_samples_split": [1,2,3,4,5,6,7,]
}

grid_model = model_selection.GridSearchCV( tree_reg,
                                           grid_params,
                                           cv=3,
                                           n_jobs=-1,
                                           scoring='neg_mean_absolute_error' )

grid_model.fit(X_train, y_train)

# %%

cv_results = pd.DataFrame(grid_model.cv_results_)
cv_results

# %%

grid_model.best_params_

# %%
pred_test = grid_model.predict(X_test)
print(metrics.mean_absolute_error(y_test, pred_test))

# %%
model = tree.DecisionTreeRegressor( **grid_model.best_params_ )
model.fit(df_treino[features], df_treino[target])

pred_oot = model.predict( df_oot[features] )
print(metrics.mean_absolute_error(df_oot[target], pred_oot))

