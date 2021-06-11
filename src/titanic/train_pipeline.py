# %%

import os
from feature_engine.encoding import OneHotEncoder, one_hot
from feature_engine.imputation import CategoricalImputer, ArbitraryNumberImputer

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import tree
from sklearn import pipeline

SRC_DIR = os.path.dirname(os.path.abspath(".")) # Define o endereço do script
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Define o endereço do script
BASE_DIR = os.path.dirname( SRC_DIR ) # Define o endereço do projeto
DATA_DIR = os.path.join( BASE_DIR, "data" ) # Define o endereço das bases de dados
TITANIC = os.path.join(DATA_DIR, "titanic") # Define o endereço do titanic

# %%

class CabinTransform(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        print("Transformação de cabines iniciado")

    def get_label(self, x, label):
        try:
            return label in x
        except:
            return False

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.copy()
        for l in "ABCD":
            X[f"Cabin{l}"] = X["Cabin"].apply(self.get_label, label=l)

        del X["Cabin"]
        X["Pclass"] = X["Pclass"].astype('str')
        return X

# %%

class AgeImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        X = X.copy()
        X = X[~X["Age"].isna()].copy()
        age = X["Age"].copy()
        del X["Age"]

        self.features = X.count()[X.count() == X.shape[0]].index.tolist()

        self.tree = tree.DecisionTreeRegressor(max_depth=4)
        self.tree.fit( X[self.features], age )

        return self

    def transform(self, X):
        X = X.copy()
        X["Age"] = self.tree.predict( X[self.features] )
        return X

# %%

if __name__ == "__main__":

    data_file = os.path.join(TITANIC, "train.csv")
    df = pd.read_csv( data_file ) # Pandas importando uma base de dados
    df.head()

    features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Cabin","Embarked"]

    cat_features = df[features].dtypes[ df[features].dtypes == "object" ].index.tolist()
    cat_features += ["Pclass"]
    num_features = list(set( features ) - set( cat_features))

    # %%

    cabin_transform = CabinTransform()
    
    new_cat = cat_features.copy()
    new_cat.remove("Cabin")
    missing_cat = CategoricalImputer(variables=new_cat)
    onehot_transform = OneHotEncoder(variables=new_cat)
    num_imputer = ArbitraryNumberImputer(arbitrary_number=-999, variables=num_features)
    age_imputer = AgeImputer()
    model = tree.DecisionTreeClassifier(max_depth=5)


    # %%
    model_pipeline = pipeline.Pipeline( [ ("cabin", cabin_transform),
                                        ("missing_cat", missing_cat),
                                        ("onehot", onehot_transform),
                                        ("num_imputer", num_imputer),
                                        ("age_imputer", age_imputer),
                                        ("model", model) ] )

    # %%

    model_pipeline.fit( df[features], df["Survived"] )

    # %%

    y_pred = model_pipeline.predict( df[features] )


    # %%

    model = pd.Series( {"features": features,
                        "cat_features": cat_features,
                        "num_features": num_features,
                        "model": model_pipeline } )

    model.to_pickle( os.path.join(TITANIC,"model_pipeline.pkl"))
