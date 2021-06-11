# %%
import os
import pandas as pd
from train_pipeline import CabinTransform, AgeImputer

SRC_DIR = os.path.dirname(os.path.abspath(".")) # Define o endereço do script
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Define o endereço do script
BASE_DIR = os.path.dirname( SRC_DIR ) # Define o endereço do projeto
DATA_DIR = os.path.join( BASE_DIR, "data" ) # Define o endereço das bases de dados
TITANIC = os.path.join(DATA_DIR, "titanic") # Define o endereço do titanic

# %%
model = pd.read_pickle("model_pipeline.pkl")

# %%
data = pd.read_csv(os.path.join(TITANIC,"test.csv"))

# %%

predict = model["model"].predict(data[ model["features"] ])