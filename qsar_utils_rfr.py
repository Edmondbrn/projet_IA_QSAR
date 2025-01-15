import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import qsar_utils as qu
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr,loguniform
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, r2_score




def plot_feature_importances(model: RandomForestRegressor, feature_names: list):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Importance des caractéristiques")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

def tune_random_forest(X: pd.DataFrame, y: pd.Series, param_distributions: dict, cv: int = 5):
    rf = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(
        rf, param_distributions=param_distributions,
        scoring='r2', cv=cv, n_iter=50, random_state=42, n_jobs=-1
    )
    search.fit(X, y)
    print(f"Meilleurs paramètres : {search.best_params_}")
    print(f"Score R² moyen : {search.best_score_}")
    return search.best_estimator_

def scatter_QSAR(x_data : list[float], 
                 y_data : list[float], 
                 corr_coef : float,
                 dot_color : str = "skyblue", 
                 line_color : str = "black",
                 line_style : str = "dashed",
                 model_type : str = "linéaire") -> None:
    """
    Fonction qui génère un graphique de régression pour un modèle défini
    x_data : valeur pour les axes des x
    y_data : valeur pour les axes des y
    corr_coef : coefficient de corrélation entre les 2 axes
    dot_color : couleur pour les points
    line_color : couleur pour la droite de régression
    line_style : définit si la ligne doit être pleine ou discontinue
    model_type : nom du modèle pour l'insérer dans le titre
    """
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 10))
    sns.regplot(x = x_data, y = y_data, color = dot_color, line_kws={"color" : line_color, "linestyle" : line_style})
    plt.xlabel("LC50 prédite", fontsize = 14)
    plt.ylabel("LC50 expérimentale", fontsize = 14)
    plt.title(f"Droite de régression depuis un modèle {model_type} entre \nla LC50 prédite et l'expérimentale", fontsize = 15)
    plt.tick_params(labelsize = 12)
    plt.text(0.05, 0.95, f"r = {corr_coef:.2f}", color = "red", fontsize = 14, ha = "left", va = "center", transform = plt.gca().transAxes ) # transform sert à indiquer que les coordonnées x et y sont relatives et non absolues
    plt.show()

def compute_score(model : RandomForestRegressor, 
            df : pd.DataFrame, 
            knn : NearestNeighbors,
            thresh : float,
            train_idx : np.ndarray,
            test_idx : np.ndarray,
            sse_global : float,
            sst_global : float,
            nb_out_AD : int,
            nb_total : int,
            predict_var : str = "LC50") -> None:
    """
    Cette fonction est le chef d'orchestre du calcul de la performance du modèle QSAR
    Elle créera les données d'entraînement et de test depuis les folds sélectionnés, 
    Elle entraînera le modèle sur les données d'entraînement et prédira les données de test
    Elle calculera les distances entre les données de test et d'entraînement pour déterminer si une molécule est dans l'AD ou non
    Elle mettra à jour les scores de SSE et SST pour le calcul de Q²
    model : modèle de régression
    df : dataframe contenant les données
    knn : modèle k-NN
    thresh : seuil de distance pour déterminer si une molécule est dans l'AD
    train_idx : index des données d'entraînement
    test_idx : index des données de test
    sse_global : somme des erreurs quadratiques pour tous les folds
    sst_global : somme des carrés totaux pour tous les folds
    nb_out_AD : nombre de molécules hors AD
    nb_total : nombre total de molécules
    predict_var : variable à prédire (LC50 par défaut)
    """
    
    # création des variables pour les données de test et d'entrainement
    X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = X_train[predict_var], X_test[predict_var]
    # suppression de la variable de toxicité
    X_train = X_train.drop(columns=[predict_var])
    X_test = X_test.drop(columns=[predict_var])

    # Entraînement du MLP sur ce fold
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mean_dist = get_mean_dist(X_train, X_test, knn)
    
    # on vérifie ceux qui sont "dans l'AD"
    # -> ici, dans AD si mean_dist < thresh
    in_AD_mask = (mean_dist < thresh)
    
    # Comptabiliser les molécules hors AD par rapport à l'ensemble des folds
    nb_out_AD += np.sum(~in_AD_mask)  # ceux hors AD
    nb_total  += len(in_AD_mask)
    
    # Pour calculer le Q², on ne garde que ceux dans l'AD
    y_test_in  = y_test[in_AD_mask]
    y_pred_in = y_pred[in_AD_mask]
    sse_global, sst_global = compute_sse_sst(y_test_in, y_pred_in, sse_global, sst_global)
    return sse_global, sst_global, nb_out_AD, nb_total

def get_mean_dist(X_train : pd.DataFrame, X_test : pd.DataFrame, knn : NearestNeighbors) -> np.ndarray:
    """
    Cette fonction permet de calculer la distance moyenne entre les k plus proches voisins d'un point
    Le modèle knn doit déjà avoir été établit au préalable
    X_train : données d'entraînement
    X_test : données de test
    knn : modèle k-NN
    Out : distance moyenne entre les k plus proches voisins des données de test
    """ 
    # Définition de l'AD sur la base du train set
    knn.fit(X_train)  # On entraîne le k-NN sur le train set
    # On calcule les distances du test set à leurs k plus proches voisins du train
    dist, _ = knn.kneighbors(X_test)  # dist.shape = (len(test_idx), k)
    # moyenne des distances aux k plus proches voisins
    mean_dist = dist.mean(axis=1)
    return mean_dist

def compute_sse_sst(y_test_in : np.ndarray, 
               y_pred_in : np.ndarray, 
               sse_global : float, 
               sst_global : float) -> tuple[float, float]:
    
    """
    Cette fonction permet de calculer et de mettre à jour les sse et sst pour tous les folds dans un modèle de corss validation
    y_test_in : valeurs de toxicité expérimentale filtrées en fonction de l'AD
    y_pred_in : valeurs de toxicité prédite filtrées en fonction de l'AD
    sse_global : somme des erreurs quadratiques pour tous les folds
    sst_global : somme des carrés totaux pour tous les folds
    Out : sse_global, sst_global
    """
    # si au moins une molécule dans l'AD          
    if len(y_test_in) > 0:
        # SSE
        sse_fold = np.sum((y_pred_in - y_test_in)**2)
        # SST
        sst_fold = np.sum((y_test_in - np.mean(y_test_in))**2)
    else:
        # s'il n'y a aucune molécule dans l'AD (très strict), on peut ignorer ou mettre 0
        sse_fold = 0.0
        sst_fold = 0.0
    # ajout des scores globaux commun aux folds
    sse_global += sse_fold
    sst_global += sst_fold
    return sse_global, sst_global

def compute_Q2(q2_per_threshold : list, pct_out_of_AD : list, sse_global : float, sst_global : float, nb_out_AD : int, nb_total : int) -> None:
    """
    FOnction qui prend en argument les sse et sst globaux pour calculer le Q²
    q2_per_threshold : liste des valeurs de Q²
    pct_out_of_AD : liste des % de molécules hors AD
    sse_global : somme des erreurs quadratiques pour tous les folds
    sst_global : somme des carrés totaux pour tous les folds
    nb_out_AD : nombre de molécules hors AD
    nb_total : nombre total de molécules
    Out : q2_per_threshold, pct_out_of_AD
    """   
    # Calcul du Q²
    if sst_global == 0:
        q2_current = 0.0
    else:
        q2_current = 1 - sse_global / sst_global
    pct_out = nb_out_AD / nb_total
    q2_per_threshold.append(q2_current)
    pct_out_of_AD.append(pct_out)
    return q2_per_threshold, pct_out_of_AD

def filter_AD(model : RandomForestRegressor, 
              df : pd.DataFrame, 
              knn : NearestNeighbors,
              cv : KFold, 
              thresholds : np.ndarray) -> tuple[list[float], list[float]]:
    """
    Fonction qui va calculer les distances entre les 6 voisins les plus proches des données de tests en cross-validation
    et les données d'entraînement pour déterminer si une molécule est dans l'AD ou non
    Si une molécule n'y est pas, son score de toxicité prédite ne sera pas compatbilisée pour le calcul de Q² (performance du modèle)
    """
    q2_per_threshold = [] # liste pour stocker les valeurs de Q2
    pct_out_of_AD = [] # liste pour stocker les valeurs de % de molécules hors AD
    # parcours des seuils de distance
    for thresh in thresholds:
        # Pour accumuler SSE et SST sur TOUTES les molécules dans l'AD
        sse_global = 0.0
        sst_global = 0.0
        # Pour compter combien de molécules "hors AD" sur l'ensemble des folds
        nb_out_AD = 0
        nb_total = 0
        print(f"Seuil : {thresh:.3f}", end = "\r")
        # On boucle sur les folds
        for train_idx, test_idx in cv.split(df):
            sse_global, sst_global, nb_out_AD, nb_total = compute_score(model, df, knn, thresh, train_idx, test_idx, sse_global, sst_global, nb_out_AD, nb_total) 
        q2_per_threshold, pct_out_of_AD = compute_Q2(q2_per_threshold, pct_out_of_AD, sse_global, sst_global, nb_out_AD, nb_total)
    print(f"Seuil : {thresh:.3f}")
    return q2_per_threshold, pct_out_of_AD

def apply_AD_and_score(X_train : pd.DataFrame, 
                       X_test : pd.DataFrame, 
                       y_test : pd.DataFrame, 
                       y_pred_test :  pd.DataFrame, 
                       knn : NearestNeighbors, 
                       threshold=None):
    """
    Applique le domaine d'applicabilité (AD) : 
    - threshold=None => pas d'AD
    - threshold=valeur => on filtre les points hors AD
    """
    if threshold is None: # On applique pas d'AD pour le score (aucun variable filtrée)
        return r2_score(y_test, y_pred_test), len(y_test), 0
    else:
        y_test_in, y_pred_in, in_AD_mask = apply_AD(X_train, X_test, y_test, y_pred_test, knn, threshold)
        nb_out = np.sum(~in_AD_mask) # somme des molécules hors AD
        if len(y_test_in) == 0:
            return np.nan, 0, nb_out
        return r2_score(y_test_in, y_pred_in), len(y_test_in), nb_out
    


def AD_graph(thresholds : np.ndarray, q2_per_threshold : np.ndarray, pct_out_of_AD : np.ndarray,
             strict_threshold : float = 0, soft_threshold : float = 0, title : str = ""):
    """
    Cette fonction permet de visualiser les espaces optimaux pour le domaine d'applicabilité du modèle QSAR
    """
    sns.set_style("whitegrid")
    sns.lineplot(x=thresholds, y=q2_per_threshold, label="Q²", color = "black" , linestyle = ":")
    sns.lineplot(x=thresholds, y=pct_out_of_AD, label="% molécules hors AD", color = "black", linestyle = "-")
    plt.axvline(x=strict_threshold, color='r', linestyle='--', label="Seuil strict")
    plt.axvline(x=soft_threshold, color='b', linestyle='--', label="Seuil souple")
    plt.xlabel("Seuil de distance")
    plt.ylabel("Q² / % molécules hors AD")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.title(title)
    plt.show()

def apply_AD(X_train : pd.DataFrame, X_test : pd.DataFrame, y_test : pd.DataFrame, y_pred_test :  pd.DataFrame, knn : NearestNeighbors, threshold : float=None):
    """
    Fonction qui filtre les données en fonctions du seuil de domaine d'applicabilité (AD)
    """
    knn.fit(X_train) # on calculce les distances entre les points de test et d'entrainement
    dist, _ = knn.kneighbors(X_test)
    mean_dist = dist.mean(axis=1)
    in_AD_mask = mean_dist < threshold
    y_test_in = y_test[in_AD_mask]
    y_pred_in = y_pred_test[in_AD_mask]
    return y_test_in, y_pred_in, in_AD_mask

# Fonction pour la validation croisée imbriquée
def run_nested_cv_rf(data, target, param_distributions, n_splits=5, n_iter=100, k=6, random_state=42, threshold_strict=0.13, threshold_soft=0.20, csv_temp="nestedCV_rf_temp.csv"):
    """
    Effectue une validation croisée imbriquée pour un RandomForestRegressor avec évaluation AD.
    """
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    inner_cv = KFold(n_splits=n_splits - 1, shuffle=True, random_state=random_state)
    
    # Définir le modèle KNN pour l'AD
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    
    results_per_fold = []
    fold_id = 0
    total = outer_cv.get_n_splits(data)
    
    for train_idx, test_idx in outer_cv.split(data):
        fold_id += 1
        X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
        
        rf = RandomForestRegressor(random_state=random_state)
        
        # RandomizedSearchCV pour trouver les meilleurs hyperparamètres sur l'ensemble interne
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=inner_cv,
            random_state=random_state,
            scoring='r2',
            n_jobs=-1
        )
        
        # Entraînement du RandomForest avec recherche d'hyperparamètres
        search.fit(X_train, y_train)
        
        # Meilleur modèle et meilleur score sur l'ensemble interne
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        # Évaluation du modèle sur l'ensemble de test (externe)
        y_pred = best_model.predict(X_test)
        
        # Application de l'AD et évaluation de la performance avec la fonction evaluate_MLP
        dict_perf = evaluate_RF(X_train, X_test, y_test, y_pred, knn, threshold_strict, threshold_soft)

        fold_result = {
            'fold': fold_id,
            'best_params': best_params,
            'best_score_innerCV': best_score,
            'r2_no_AD': dict_perf["no_AD"][0],
            'r2_strict': dict_perf["strict"][0],
            'r2_soft': dict_perf["soft"][0],
            'nb_test': len(test_idx),
            'nb_in_noAD': dict_perf["no_AD"][1],
            'nb_in_strict': dict_perf["strict"][1],
            'nb_in_soft': dict_perf["soft"][1],
            'nb_out_noAD': dict_perf["no_AD"][2],
            'nb_out_strict': dict_perf["strict"][2],
            'nb_out_soft': dict_perf["soft"][2],
        }
        results_per_fold.append(fold_result)
        pd.DataFrame(results_per_fold).to_csv(csv_temp, index=False)
        print(f"Fold {fold_id}/{total} terminé")

    # Sauvegarde des résultats finaux
    df_results = pd.DataFrame(results_per_fold)
    df_results.to_csv("nestedCV_rf_final_results.csv", index=False)
    print(df_results)

    return df_results

def evaluate_RF(X_train_outer : pd.DataFrame, 
                X_test_outer : pd.DataFrame, 
                y_test_outer : pd.DataFrame, 
                y_pred_outer : pd.DataFrame, 
                knn : NearestNeighbors, 
                threshold_strict : float, 
                threshold_soft : float,
                threshold_none : float = None) -> dict[str : list[float]]:
    """
    Cette fonction permet d'obtenir les performances du modèle RandomForest selon 3 niveaux d'AD.
    X_train_outer : données d'entraînement
    X_test_outer : données de test
    y_test_outer : valeurs de toxicité expérimentale
    y_pred_outer : valeurs de toxicité prédite
    knn : modèle k-NN
    threshold_strict : seuil strict pour l'AD
    threshold_soft : seuil souple pour l'AD
    threshold_none : pas d'AD
    """
    # Appliquer l'AD et calculer le score R² sans AD
    r2_noAD, nb_in_noAD, nb_out_noAD = apply_AD_and_score(
        X_train_outer, X_test_outer, y_test_outer,
        y_pred_outer, knn, threshold_none
    )
    
    # Appliquer l'AD avec le seuil strict
    r2_strict, nb_in_strict, nb_out_strict = apply_AD_and_score(
        X_train_outer, X_test_outer, y_test_outer,
        y_pred_outer, knn, threshold_strict
    )
    
    # Appliquer l'AD avec le seuil souple
    r2_soft, nb_in_soft, nb_out_soft = apply_AD_and_score(
        X_train_outer, X_test_outer, y_test_outer,
        y_pred_outer, knn, threshold_soft
    )
    
    # Création du dictionnaire des résultats
    dict_resultat = {
        "no_AD": [r2_noAD, nb_in_noAD, nb_out_noAD],
        "strict": [r2_strict, nb_in_strict, nb_out_strict],
        "soft": [r2_soft, nb_in_soft, nb_out_soft]
    }
    
    return dict_resultat