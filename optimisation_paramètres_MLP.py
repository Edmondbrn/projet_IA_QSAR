import pandas as pd
import qsar_utils as qu
from scipy.stats import loguniform
from sklearn.model_selection import train_test_split

# Création des données de test et d'entrainement
# df = pd.read_csv("/home/be203133/projet_IA_QSAR/data/qsar_fish_toxicity_norm.csv")
# df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
# df_train.to_csv("/home/be203133/projet_IA_QSAR/data/qsar_fish_toxicity_norm_train.csv", index=False)
# df_test.to_csv("/home/be203133/projet_IA_QSAR/data/qsar_fish_toxicity_norm_test.csv", index=False)





import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr,loguniform
import qsar_utils as qu
import pickle
import ast



df_norm = pd.read_csv("data/qsar_fish_toxicity_norm.csv")
LC50 = df_norm["LC50"]
data = df_norm.drop(columns=["LC50"])
best_models = pd.read_csv("nested_MLP_2/best_models.csv")
best_models = best_models[["nb_neurones","alpha","activation","hidden_layer_sizes","learning_rate_init"]]
# conversion de la colonne avec les couches en tuple
best_models["hidden_layer_sizes"] = best_models["hidden_layer_sizes"].apply(lambda x: ast.literal_eval(x))



df_final = pd.DataFrame(columns=["nb_neurones","alpha","activation","hidden_layer_sizes","learning_rate_init","no_AD","strict","soft"])
n_model_tot = len(best_models)
cpt =  0
for model in best_models.iterrows(): # parcours des modèles
    cpt += 1
    param = model[1].to_dict() # conversion des lignes en dictionnaires pour la praticité
    scores = {"no_AD": [], "strict": [], "soft": []} # initialisation des scores
    mean_score = {"no_AD": 0, "strict": 0, "soft": 0}
    for i in range(5): # on boucle 5 fois pour la cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42+i) # on split les données en 5, 5 fois différemment
        for train_index, test_index in kf.split(data):
            knn = NearestNeighbors(n_neighbors=6, metric='euclidean') # initialisation du knn
            model = MLPRegressor( # initialisation du modèle
                activation=param["activation"],
                alpha=param["alpha"],
                hidden_layer_sizes=param["hidden_layer_sizes"],
                learning_rate_init=param["learning_rate_init"],
                random_state=42,
                max_iter = 100000
            )
            X_train, X_test = data.iloc[train_index], data.iloc[test_index] # on récupère les données
            y_train, y_test = LC50.iloc[train_index], LC50.iloc[test_index]
            model.fit(X_train, y_train) # entrainement du modèle
            y_pred = model.predict(X_test) # prédictions
            # Calculd es score et stockage
            score_noAD,_,_ = qu.apply_AD_and_score(X_train, X_test, y_test, y_pred, knn, threshold=None)
            score_strict,_,_ = qu.apply_AD_and_score(X_train, X_test, y_test, y_pred, knn, threshold=0.13)
            score_soft,_,_ = qu.apply_AD_and_score(X_train, X_test, y_test, y_pred, knn, threshold=0.2)
            scores["no_AD"].append(score_noAD)
            scores["strict"].append(score_strict)
            scores["soft"].append(score_soft)
    # calculd es moyenens des scores pour chaque modèle
    row_final = pd.DataFrame({
        "nb_neurones": [param["nb_neurones"]],
        "alpha": [param["alpha"]],
        "activation": [param["activation"]],
        "hidden_layer_sizes": [param["hidden_layer_sizes"]],
        "learning_rate_init": [param["learning_rate_init"]],
        "no_AD": [np.mean(scores["no_AD"])],
        "no_AD_sd": [np.std(scores["no_AD"])],
        "strict": [np.mean(scores["strict"])],
        "strict_sd": [np.std(scores["strict"])],
        "soft": [np.mean(scores["soft"])],
        "soft_sd": [np.std(scores["soft"])]
    })
    df_final = pd.concat([df_final, row_final])
    df_final.to_csv("nested_MLP_2/Evalution_finale_temp.csv", index=False)
    print(f"{cpt}/{n_model_tot}", end = "\r")
print(f'{cpt}/{n_model_tot}')

df_final.to_csv("nested_MLP_2/Evalution_finale.csv", index=False)


# df_norm = pd.read_csv("/home/be203133/projet_IA_QSAR/data/qsar_fish_toxicity_norm_train.csv")
# LC50 = df_norm["LC50"]
# df_norm = df_norm.drop(columns=["LC50"])

# total_neurones = nb_total_neuron=[10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
# couches = range(1, 11)
# qu.generate_configurations(total_neurones, couches, "configurations.txt")

# for i, T in enumerate(total_neurones):
#     list_of_tuples = qu.load_neural_configuration(f"configurations_{T}.txt")
#     # définit les ensembles de paramètres à tester
#     param_distributions = {
#         'hidden_layer_sizes': list_of_tuples,
#         'activation': ['relu', 'tanh'],
#         'alpha': loguniform(1e-5, 1e-1),
#         'learning_rate_init': loguniform(1e-5, 1e-1)
#     }
#     qu.run_nested_cv_MLP(df_norm, LC50, param_distributions, f"MLP_{T}", threshold_strict=0.13, threshold_soft=0.2)