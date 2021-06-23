# %%

import feature_engine
import pandas as pd
from pandas.core.algorithms import mode
from scipy.sparse.construct import random
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.encoding import OneHotEncoder

from sklearn import pipeline
from sklearn import tree
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics

# %%

class MLImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables: list, estimator: BaseEstimator, only_complete=True):
        self.variables = variables  # variáveis para realizar a imputação
        self.estimator = estimator  # Tipo de algoritmo para realizar a imputação
        self.only_complete = only_complete  # User apenas as variáveis que temos imformação completa para a modelagem
        self.estimators = {}

    def get_params(self, deep):
        return {       
                "variables": self.variables,
                "estimator": self.estimator,
                "only_complete": self.only_complete, }

    def fit(self, X, y):
        features = X.count()[X.count() == X.shape[0]].index.tolist()
        
        for i in self.variables:
            if i in features:
                features.remove(i)

        for v in self.variables:
            self.fit_one(X=X, target=v, features=features)

        return self

    def transform(self, X):
        for v in self.variables:
            X = self.transform_one(X, v)
        return X

    def fit_one(self, X: pd.DataFrame, target: str, features=None):

        X_complete = X[~X[target].isna()].copy()

        model = self.estimator()
        model.fit(X_complete[features], X_complete[target])

        self.estimators[target] = {"features": features, "model": model}

    def transform_one(self, X, target):

        X_full = X[~X[target].isna()].copy()

        if X_full.shape[0] == X.shape[0]:
            return X

        X_na = X[X[target].isna()].copy()

        model = self.estimators[target]["model"]
        features = self.estimators[target]["features"]

        X_na[target] = model.predict(X_na[features])

        X = pd.concat([X_full, X_na]).sort_index()

        return X


# %%

df = pd.read_csv("../../data/titanic/train.csv")

target = "Survived"
cat_features = ["Sex", "Embarked", "Pclass"]
num_features = ["Age","SibSp", "Parch", "Fare"]

df[cat_features] = df[cat_features].astype(str)

features = cat_features + num_features
# %%

X_train, X_test, y_train, y_test = model_selection.train_test_split( df[features],
                                                                     df[target],
                                                                     test_size=0.2,
                                                                     random_state=42 )


# %%
onehot_1 = OneHotEncoder(variables=["Sex", "Pclass"])

cat_imputer = MLImputer( variables=["Embarked"],
                         estimator=tree.DecisionTreeClassifier )

onehot_2 = OneHotEncoder(variables=["Embarked"])

num_imputer = MLImputer( variables=["Age","Fare"],
                         estimator=tree.DecisionTreeRegressor )

clf = ensemble.RandomForestClassifier(n_jobs=1,random_state=42)

params = {"n_estimators":[100,200,300],
          "max_depth":[2,5,7,10],
          "min_samples_leaf": [5,10,15] }

grid_search = model_selection.GridSearchCV(clf, params, cv=4)

model = pipeline.Pipeline( [ ("onehot-1", onehot_1),
                             ("cat_imputer", cat_imputer),
                             ("onehot-2", onehot_2),
                             ("num_imputer", num_imputer),
                             ("random_forest", grid_search)] )
# %%

model.fit(X_train, y_train)
# %%


# %%
onehot_1 = OneHotEncoder(variables=["Sex", "Pclass"])

cat_imputer = MLImputer( variables=["Embarked"],
                         estimator=tree.DecisionTreeClassifier )

onehot_2 = OneHotEncoder(variables=["Embarked"])

num_imputer = MLImputer( variables=["Age","Fare"],
                         estimator=tree.DecisionTreeRegressor )

clf = ensemble.RandomForestClassifier(n_jobs=1,random_state=42)

model = pipeline.Pipeline( [ ("onehot-1", onehot_1),
                             ("cat_imputer", cat_imputer),
                             ("onehot-2", onehot_2),
                             ("num_imputer", num_imputer),
                             ("random_forest", clf)] )


params = {"random_forest__n_estimators":[100,200,300],
          "random_forest__max_depth":[2,5,7,10],
          "random_forest__min_samples_leaf": [5,10,15],
          "cat_imputer__estimator": [tree.DecisionTreeClassifier, ensemble.RandomForestClassifier],
          "num_imputer__estimator": [tree.DecisionTreeRegressor, ensemble.RandomForestRegressor],
           }

# %%
grid_search = model_selection.GridSearchCV(model, params, cv=4,n_jobs=-1)
grid_search.fit( X_train, y_train )


# %%
random_search = model_selection.RandomizedSearchCV(model, params, n_iter=100, cv=4, random_state=42)
grid_search.fit( X_train, y_train)
# %%
