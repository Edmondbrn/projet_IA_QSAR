import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
import qsar_utils_rfr as qu  # Module utilisateur pour des fonctions QSAR spécifiques
import numpy as np

# Chargement des données d'entraînement et de test
df_train = pd.read_csv("data/qsar_fish_toxicity_norm_train.csv")
df_test = pd.read_csv("data/qsar_fish_toxicity_norm_test.csv")

# Extraction des cibles (LC50) et des features
LC50_train = df_train["LC50"]
LC50_test = df_test["LC50"]

data_train = df_train.drop(columns=["LC50"])
data_test = df_test.drop(columns=["LC50"])

# Initialisation du DataFrame pour stocker les résultats finaux
df_final = pd.DataFrame(columns=[
    "max_depth", "min_samples_leaf", "min_samples_split", "n_estimators", 
    "no_AD", "no_AD_sd", "strict", "strict_sd", "soft", "soft_sd"
])

# Chargement des paramètres optimaux des modèles
best_models = pd.read_csv("RFR_models/nestedCV_rf_final_results0.csv")
best_models = best_models[["best_params"]]  # Ne conserver que la colonne des paramètres optimaux
n_model_tot = len(best_models)  # Nombre total de modèles à évaluer

# Boucle sur chaque modèle avec ses paramètres optimaux
for k, param in best_models.iterrows():
    # Conversion de la chaîne de caractères contenant les paramètres en dictionnaire
    best_param_dict = eval(param["best_params"])

    # Initialisation des scores pour les différentes stratégies de Domain of Applicability (AD)
    scores = {"no_AD": [], "strict": [], "soft": []}

    # Cross-validation répétée 5 fois
    for i in range(5):
        # Création d'un validateur KFold avec 5 splits
        kf = KFold(n_splits=5, shuffle=True, random_state=42 + i)

        # Boucle sur les splits de la validation croisée
        for train_index, test_index in kf.split(data_train):
            # Initialisation du modèle KNN pour définir le Domain of Applicability
            knn = NearestNeighbors(n_neighbors=6, metric='euclidean')

            # Initialisation du modèle Random Forest avec les paramètres optimaux
            model = RandomForestRegressor(**best_param_dict)

            # Séparation des données d'entraînement et de test
            X_train, X_test = data_train.iloc[train_index], data_train.iloc[test_index]
            y_train, y_test = LC50_train.iloc[train_index], LC50_train.iloc[test_index]

            # Entraînement du modèle Random Forest
            model.fit(X_train, y_train)

            # Prédictions sur les données de test
            y_pred = model.predict(X_test)

            # Évaluation des scores pour différentes stratégies d'AD
            score_noAD, _, _ = qu.apply_AD_and_score(X_train, X_test, y_test, y_pred, knn, threshold=None)
            score_strict, _, _ = qu.apply_AD_and_score(X_train, X_test, y_test, y_pred, knn, threshold=0.13)
            score_soft, _, _ = qu.apply_AD_and_score(X_train, X_test, y_test, y_pred, knn, threshold=0.2)

            # Stockage des scores pour cette itération
            scores["no_AD"].append(score_noAD)
            scores["strict"].append(score_strict)
            scores["soft"].append(score_soft)

    # Calcul des moyennes et écarts-types des scores pour chaque stratégie
    row_final = pd.DataFrame({
        'max_depth': [best_param_dict["max_depth"]], 
        'min_samples_leaf': [best_param_dict["min_samples_leaf"]], 
        'min_samples_split': [best_param_dict["min_samples_split"]],
        'n_estimators': [best_param_dict["n_estimators"]],
        "no_AD": [np.mean(scores["no_AD"])], "no_AD_sd": [np.std(scores["no_AD"])],
        "strict": [np.mean(scores["strict"])], "strict_sd": [np.std(scores["strict"])],
        "soft": [np.mean(scores["soft"])], "soft_sd": [np.std(scores["soft"])]
    })

    # Ajout des résultats au DataFrame final
    df_final = pd.concat([df_final, row_final], ignore_index=True)

    # Sauvegarde intermédiaire des résultats
    df_final.to_csv("RFR_models/valution_finale_temp.csv", index=False)

    # Affichage de la progression
    print(f"{k + 1}/{n_model_tot}", end="\r")

# Sauvegarde finale des résultats
print("\nÉvaluation terminée. Résultats sauvegardés.")
