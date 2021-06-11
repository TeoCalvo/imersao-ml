# %%
import os
import pandas as pd

from sklearn import tree
from sklearn import metrics
from sklearn import linear_model

from feature_engine.encoding import OneHotEncoder

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(".")) # Define o endereço do script
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Define o endereço do script
BASE_DIR = os.path.dirname( SRC_DIR ) # Define o endereço do projeto
DATA_DIR = os.path.join( BASE_DIR, "data" ) # Define o endereço das bases de dados
TITANIC = os.path.join(DATA_DIR, "titanic") # Define o endereço do titanic

# %%
data_file = os.path.join(TITANIC, "train.csv")
df = pd.read_csv( data_file ) # Pandas importando uma base de dados

# %%

def cabin_split(x):

    try:
        return x.split(" ")
    except AttributeError:
        return []

def flag_letter_cabin(x, flag="A"):
    count = 0
    for i in x:
        count += 1 if flag in i else 0
    return count

df["CabinSplit"] = df["Cabin"].apply(cabin_split)

df["CabinA"] = df["CabinSplit"].apply(flag_letter_cabin, flag="A")
df["CabinB"] = df["CabinSplit"].apply(flag_letter_cabin, flag="B")
df["CabinC"] = df["CabinSplit"].apply(flag_letter_cabin, flag="C")
df["CabinD"] = df["CabinSplit"].apply(flag_letter_cabin, flag="D")

# %%

features = ["Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "CabinA",
            "CabinB",
            "CabinC",
            "CabinD",
            "Embarked"]

target = ["Survived"]

df[features].describe

# %%

# Decribe variáveis numéricas
cat_features = df[features].dtypes[df[features].dtypes == 'object'].index.tolist()
cat_features += ["Pclass"]

num_features = list(set(features) - set(cat_features))

df[cat_features] = df[cat_features].astype(str)
df[cat_features].describe()

# %%

onehot = OneHotEncoder(variables=cat_features, drop_last=False)
onehot.fit( df[features] )

X = onehot.transform( df[features] )

# %%

X_age = X[~X["Age"].isna()].copy()

imput_age_model = tree.DecisionTreeRegressor(max_depth=4)

features_age = X.columns.tolist()
features_age.remove("Age")

imput_age_model.fit( X_age[features_age], X_age["Age"] )

X["AgePredict"] = imput_age_model.predict(X[features_age])

def imput_age(row):
    if np.isnan(row["Age"]):
        return row["AgePredict"]
    else:
        return row["Age"]

X["Age"] = X[["Age", "AgePredict"]].apply( imput_age, axis=1 )

# %%

clf_rl = linear_model.LogisticRegression(penalty="none")
clf_rl.fit(X, df[target])

pred_rl = clf_rl.predict(X)

# %%

clf_tree = tree.DecisionTreeClassifier(max_depth=4)
clf_tree.fit(X, df[target])

pred_tree = clf_tree.predict(X)

# %%

print(metrics.accuracy_score(df[target], pred_rl))
print(metrics.accuracy_score(df[target], pred_tree))

# %%

model_dict = {
    "features": features,
    "cat_features": cat_features,
    "num_features": num_features,
    "onehot": onehot,
    "model_age": imput_age_model,
    "features_age":features_age,
    "model": clf_tree,
}

model_serie = pd.Series(model_dict)

model_serie.to_pickle("model.pkl")