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

data[ "Proba_Reg_Log" ] = pred_rl[:,1]
data

# %%

cervejas = [ [i] for i in np.linspace(0,10,100)]
pred_n = model_rl.predict_proba( cervejas )

plt.plot( "Cerveja", "Aprovado", "o" ,data=data)
plt.plot( cervejas, pred_n, "-" )
plt.grid(True)
plt.legend(["Oberservações", "Prob(y=0)", "Prob(y=1)"])


# %%

from sklearn import tree
import graphviz 

clf_tree_gini = tree.DecisionTreeClassifier(criterion="gini")
clf_tree_gini.fit( data[["Cerveja"]], data["Aprovado"] )

# %%

dot_data = tree.export_graphviz(clf_tree_gini, out_file=None, 
                     feature_names=["Cerveja"],  
                     class_names=["0", "1"],  
                     filled=True, rounded=True,  
                     special_characters=True)

graph = graphviz.Source(dot_data)  
graph 

# %%


clf_tree_entropy = tree.DecisionTreeClassifier(criterion="entropy")
clf_tree_entropy.fit( data[["Cerveja"]], data["Aprovado"] )

# %%

dot_data = tree.export_graphviz(clf_tree_entropy, out_file=None, 
                     feature_names=["Cerveja"],  
                     class_names=["0", "1"],  
                     filled=True, rounded=True,  
                     special_characters=True)

graph = graphviz.Source(dot_data)  
graph 


# %%

from sklearn import naive_bayes

clf_nb = naive_bayes.GaussianNB()
clf_nb.fit( data[["Cerveja"]], data["Aprovado"] )

prob_nb = clf_nb.predict_proba( data[["Cerveja"]] )

prob_nb

# %%

pred_rl = model_rl.predict(data[["Cerveja"]])
pred_tree_gini = clf_tree_gini.predict(data[["Cerveja"]])
pred_tree_entropy = clf_tree_entropy.predict(data[["Cerveja"]])
pred_nb = clf_nb.predict(data[["Cerveja"]])

from sklearn import metrics