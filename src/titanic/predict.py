# %%
import os
import pandas as pd
from pandas.core.algorithms import mode

from sklearn import tree
from sklearn import metrics
from sklearn import linear_model
from feature_engine.encoding import OneHotEncoder

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath("."))# Define o endereço do script
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Define o endereço do script
BASE_DIR = os.path.dirname( SRC_DIR ) # Define o endereço do projeto
DATA_DIR = os.path.join( BASE_DIR, "data" ) # Define o endereço das bases de dados
TITANIC = os.path.join(DATA_DIR, "titanic") # Define o endereço do titanic

# %%

# Importação dos dados
data_file = os.path.join(TITANIC, "test.csv")
df = pd.read_csv(data_file)

# %%
# Importação do modelo

model = pd.read_pickle("model.pkl")
model

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

X = model["onehot"].transform( df[model["features"]] )
X = X.fillna(0)

# %%

X["AgePredict"] = model["model_age"].predict(X[ model["features_age"] ])

def imput_age(row):
    if np.isnan(row["Age"]):
        return row["AgePredict"]
    else:
        return row["Age"]

X["Age"] = X[["Age", "AgePredict"]].apply( imput_age, axis=1 )

# %%
df["Survived"] = model["model"].predict(X)
df[["PassengerId", "Survived"]].to_csv("predictions.csv", index=False)