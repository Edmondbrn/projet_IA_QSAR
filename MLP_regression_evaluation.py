# %% [markdown]
# ### Evaluation des 10 meilleurs modèles issus de l'optimisation des hyperparamètres

# %% [markdown]
# #### Importation des modules

# %%
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr,loguniform
import qsar_utils as qu
import pickle
import os
import re
import ast

# # %% [markdown]
# # #### Chargement des données

# # %%
# df_norm = pd.read_csv("data/qsar_fish_toxicity_norm.csv")
# LC50 = df_norm["LC50"]
# data = df_norm.drop(columns=["LC50"])
# best_models = pd.read_csv("nestedCV_MLP/best_models.csv")
# best_models = best_models[["nb_neurones","alpha","activation","hidden_layer_sizes","learning_rate_init"]]
# # conversion de la colonne avec les couches en tuple
# best_models["hidden_layer_sizes"] = best_models["hidden_layer_sizes"].apply(lambda x: ast.literal_eval(x))

# # %% [markdown]
# # #### Entrainement des 10 meilleurs modèles 5 fois en cross validation

# # %%
# df_final = pd.DataFrame(columns=["nb_neurones","alpha","activation","hidden_layer_sizes","learning_rate_init","no_AD","strict","soft"])
# n_model_tot = len(best_models)
# cpt =  0
# for model in best_models.iterrows():
#     cpt += 1
#     param = model[1].to_dict()
#     scores = {"no_AD": [], "strict": [], "soft": []}
#     mean_score = {"no_AD": 0, "strict": 0, "soft": 0}
#     kf = KFold(n_splits=5, shuffle=True, random_state=None)
#     for train_index, test_index in kf.split(data):
#         knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
#         model = MLPRegressor(
#             activation=param["activation"],
#             alpha=param["alpha"],
#             hidden_layer_sizes=param["hidden_layer_sizes"],
#             learning_rate_init=param["learning_rate_init"],
#             random_state=42,
#             max_iter = 100000
#         )
#         X_train, X_test = data.iloc[train_index], data.iloc[test_index]
#         y_train, y_test = LC50.iloc[train_index], LC50.iloc[test_index]
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         score_noAD,_,_ = qu.apply_AD_and_score(X_train, X_test, y_test, y_pred, knn, threshold=None)
#         score_strict,_,_ = qu.apply_AD_and_score(X_train, X_test, y_test, y_pred, knn, threshold=0.13)
#         score_soft,_,_ = qu.apply_AD_and_score(X_train, X_test, y_test, y_pred, knn, threshold=0.2)
#         scores["no_AD"].append(score_noAD)
#         scores["strict"].append(score_strict)
#         scores["soft"].append(score_soft)

#     row_final = pd.DataFrame({
#         "nb_neurones": [param["nb_neurones"]],
#         "alpha": [param["alpha"]],
#         "activation": [param["activation"]],
#         "hidden_layer_sizes": [param["hidden_layer_sizes"]],
#         "learning_rate_init": [param["learning_rate_init"]],
#         "no_AD": [np.mean(scores["no_AD"])],
#         "strict": [np.mean(scores["strict"])],
#         "soft": [np.mean(scores["soft"])]
#     })
#     df_final = pd.concat([df_final, row_final])
#     df_final.to_csv("nestedCV_MLP/Evalution_finale_temp_light.csv", index=False)
#     print(f"{cpt}/{n_model_tot}", end = "\r")
# print(f'{cpt}/{n_model_tot}')

# df_final.to_csv("nestedCV_MLP/Evalution_finale_light.csv", index=False)




def LOO_LC50_MLP(df : pd.DataFrame, df_LC50 : pd.DataFrame) -> tuple[list[float], float]:
    """
    Cette fonction permet d'effectuer un Leave one Out sur des données pandas et de l'appliquer à un modèle MLP pour
    prédire la valeur de la LC50 d'une molécule
    df : tableau pandas avec les données
    df_LC50 : tableau pandas avec les valeurs de la LC50
    model : modèle MLP
    """
    print("Début du LOO du MLP")
    loo = LeaveOneOut() # initialisation du LeaveOneOut
    predicted_LC50 = list()
    model = MLPRegressor(hidden_layer_sizes=(56, 56, 56, 56, 56, 55, 55, 55, 55), 
                         max_iter=100000, 
                         alpha=0.005044, 
                         solver='adam', 
                         activation='relu', 
                         learning_rate_init=0.000028)
    tot = len(df)
    cpt = 0
    for train_index, test_index in loo.split(df): # parcours des index sélectionnés par le leave one out
        cpt += 1
        model.fit(df.iloc[train_index], df_LC50.iloc[train_index])
        predict = model.predict(df.iloc[test_index])
        predicted_LC50.append(predict[0])
        print(f"{cpt}/{tot}, predicted LC50: {predict[0]}")
    
    corr_coef, _ = pearsonr(df_LC50, predicted_LC50) # calcul du coefficients de corrélation
    slope = model.coefs_
    intercept = model.intercepts_
    return predicted_LC50, corr_coef, slope, intercept

df_norm = pd.read_csv("data/qsar_fish_toxicity_norm.csv")
df_LC50 = df_norm["LC50"]
df_norm = df_norm.drop(columns=["LC50"])

predicted_LC50, corr_coef, slope, intercept = LOO_LC50_MLP(df_norm, df_LC50)


print(f"Le coefficient de corrélation est de {corr_coef}")
print(f"Le slope est de {slope}")
print(f"L'intercept est de {intercept}")
print(f"Les valeurs prédites sont {predicted_LC50}")


