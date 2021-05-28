# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz 


# %%
np.random.seed(28) # Definindo seed

data = [ (i, np.random.normal(i, 2)) for i in np.random.choice( list(range(1,10)), 15, replace=True ) ]

data = pd.DataFrame(data, columns = ["Cerveja", "Nota"])
data['Nota'] = data['Nota'].abs()
data["Nota"] = data["Nota"].apply(lambda x: x if x < 10 else 10)

data["Aprovado"] = (data["Nota"] >= 6).astype(int)

data

# %%

from sklearn import linear_model

model_rl = linear_model.LogisticRegression(penalty="none")
model_rl.fit( data[["Cerveja"]], data["Aprovado"] )

# %%

print("Beta0:", model_rl.intercept_)
print("Beta1:", model_rl.coef_)

# %%

pred_rl = model_rl.predict_proba(data[["Cerveja"]])

plt.plot( "Cerveja", "Aprovado", "o" ,data=data)
plt.plot( data["Cerveja"], pred_rl, "o" )
plt.grid(True)
plt.legend(["Oberservações", "Prob(y=0)", "Prob(y=1)"])

# %%

cervejas = [ [i] for i in np.linspace(0,10,100)]
pred_n = model_rl.predict_proba( cervejas )

plt.plot( "Cerveja", "Aprovado", "o" ,data=data)
plt.plot( cervejas, pred_n, "-" )
plt.grid(True)
plt.legend(["Oberservações", "Prob(y=0)", "Prob(y=1)"])


# %%

from sklearn import tree

clf_tree_gini = tree.DecisionTreeClassifier(criterion="gini")