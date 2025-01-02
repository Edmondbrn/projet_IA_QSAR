import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import make_scorer, r2_score
from scipy.stats import loguniform  # distributions possibles
import ast

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
    

def compute_score(model : MLPRegressor | RandomForestRegressor, 
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



def filter_AD(model : MLPRegressor | RandomForestRegressor, 
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


def lineplot(x_data : list[float],
             y_data : list[float],
             title : str,
             x_label : str,
             y_label : str,
             line_color : str = "black",
             line_style : str = "solid") -> None:
    """
    Fonction qui génère un graphique linéaire
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 10))
    sns.lineplot(x = x_data, y = y_data, color = line_color, linestyle = line_style)
    plt.xlabel(x_label, fontsize = 14)
    plt.ylabel(y_label, fontsize = 14)
    plt.title(title, fontsize = 15)
    plt.tick_params(labelsize = 12)
    plt.show()

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


    
def barplot_perf(data : pd.DataFrame, x_col : str, y_col : str , yerr_col : str, 
                 x_label : str, y_label : str, title : str,
                 ylim : tuple[float] = (0.5, 0.625)) -> None:
    """
    Fonction qui créé un graphique en barres pour visualiser les performances du modèle
    Les barres sont les moyennes et les erreurs sont les écarts-types
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))

    ax = sns.barplot(
        x=x_col,
        y=y_col,
        data=data,
        hue=x_col,
        palette="Blues_d",
        errorbar=None, 
        legend=False
    )
    positions = range(len(data))

    ax.errorbar(
        x=positions,
        y=data[y_col],
        yerr=data[yerr_col],
        fmt='none',
        c='black',
        capsize=5
    )

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16)
    plt.ylim(ylim)
    plt.show()

def boxplot_perf(data : pd.DataFrame, x_col : str, y_col : str, x_label : str, y_label : str, title : str) -> None:
    """
    Fonction qui créé un graphique en boîte pour visualiser les performances du modèle
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=x_col, y=y_col, data=data, hue = x_col,palette="Blues_d", legend=False)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16)
    plt.show()

def load_data(csv_path="data/qsar_fish_toxicity_norm.csv", drop_col="LC50"):
    """
    Charge et prépare les données : dataframe normalisée 
    et cible LC50.
    """
    df_norm = pd.read_csv(csv_path)
    data = df_norm.drop(columns=[drop_col])
    LC50 = df_norm[drop_col]
    return data, LC50

def repartition_egale(T, L):
    """
    Retourne un tuple de L valeurs qui se répartissent T neurones
    au plus équitablement possible.
    """
    base = T // L
    reste = T % L
    result = [base]*L
    for i in range(reste):
        result[i] += 1
    return tuple(result)

def generate_configurations(nb_total_neuron : int, nb_layers_max : list[int], file_name : str = "nestedCV_MLP/congiguration_default.txt") -> None:
    """
    Génère des fichiers de configuration pour différentes
    architectures (nb neurones totaux répartis sur nb de couches).
    """
    file_name = file_name.replace(".txt", "")
    for T in nb_total_neuron:
        with open(file_name + f"_{T}.txt", "w") as fh:
            for L in nb_layers_max:
                config = repartition_egale(T, L)
                fh.write(f"{config},")

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
        knn.fit(X_train) # on calculce les distances entre les points de test et d'entrainement
        dist, _ = knn.kneighbors(X_test)
        mean_dist = dist.mean(axis=1)
        in_AD_mask = mean_dist < threshold
        y_test_in = y_test[in_AD_mask]
        y_pred_in = y_pred_test[in_AD_mask]
        nb_out = np.sum(~in_AD_mask) # somme des molécules hors AD
        if len(y_test_in) == 0:
            return np.nan, 0, nb_out
        return r2_score(y_test_in, y_pred_in), len(y_test_in), nb_out
    


def load_neural_configuration(file_path):
    """
    Charge une configuration de neurones depuis un fichier.
    """
    with open(file_path, "r") as fh:
        config_line = fh.readlines()[0].strip().rstrip(',')
        list_of_tuples = ast.literal_eval(f"[{config_line}]")
    return list_of_tuples


def evaluate_MLP(X_train_outer : pd.DataFrame, 
                 X_test_outer : pd.DataFrame, 
                 y_test_outer : pd.DataFrame, 
                 y_pred_outer : pd.DataFrame, 
                 knn : NearestNeighbors, 
                 threshold_strict : float, 
                 threshold_soft : float,
                 threshold_none : float = None) -> dict[str : list[float]]:
    """
    Cette fonction permet d'obtenir les performances du modèle MLP selon 3 niveaux d'AD
    X_train_outer : données d'entraînement
    X_test_outer : données de test
    y_test_outer : valeurs de toxicité expérimentale
    y_pred_outer : valeurs de toxicité prédite
    knn : modèle k-NN
    threshold_strict : seuil strict pour l'AD
    threshold_soft : seuil souple pour l'AD
    threshold_none : pas d'AD
    """
    r2_noAD, nb_in_noAD, nb_out_noAD = apply_AD_and_score(
        X_train_outer, X_test_outer, y_test_outer,
        y_pred_outer, knn, threshold_none
    )
    r2_strict, nb_in_strict, nb_out_strict = apply_AD_and_score(
        X_train_outer, X_test_outer, y_test_outer,
        y_pred_outer, knn, threshold_strict
    )
    r2_soft, nb_in_soft, nb_out_soft = apply_AD_and_score(
        X_train_outer, X_test_outer, y_test_outer,
        y_pred_outer, knn, threshold_soft
    )
    dict_resultat = {
        "no_AD": [r2_noAD, nb_in_noAD, nb_out_noAD],
        "strict": [r2_strict, nb_in_strict, nb_out_strict],
        "soft": [r2_soft, nb_in_soft, nb_out_soft]
    }
    return dict_resultat

def run_nested_cv_MLP(  data : pd.DataFrame, 
                        LC50 : pd.DataFrame, 
                        param_distributions : dict,
                        num_config : int,
                        threshold_strict : float = 0.13,
                        threshold_soft : float = 0.20,
                        k : int = 6, 
                        csv_temp : str ="nestedCV_MLP/nestedCV_results_temp_new.csv"
                    ):
    """
    Lit la config, effectue la nested CV et sauvegarde les résultats
    pour chaque configuration de neurones (fichiers config).
    """
    # Définition du modèle des plus proches voisins pour l'AD
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    X_scaled, y = data, LC50
    # Définition de la cross-validation par la méthode des k-folds externes (ici pour évaluer les performances du modèle)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)
    my_scorer = make_scorer(r2_score, greater_is_better=True) # fonction de score pour la cross-validation

    results_per_fold = []
    fold_id = 0
    # parcours des différents folds pour la cross validation (CV) pour l'évaluation des performances du modèle
    for train_idx, test_idx in outer_cv.split(X_scaled):
        fold_id += 1
        # Séparation des données en train et test
        X_train_outer = X_scaled.iloc[train_idx]
        X_test_outer  = X_scaled.iloc[test_idx]
        y_train_outer = y.iloc[train_idx]
        y_test_outer  = y.iloc[test_idx]
        # définition des folds internes pour la cross validation (pour l'optimisation des hyperparamètres)
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        mlp = MLPRegressor(solver='adam', max_iter=100000, random_state=42)
        # balayage aléatoire des hyperparamètres sur 100 itérations
        random_search = RandomizedSearchCV( # Ici on prend les folds d'entrainement externes que l'on va recouper en folds internes (entrainement + test) pour évaluer les hyperparamètres choisis
            estimator = mlp,
            param_distributions = param_distributions,
            scoring = my_scorer,
            cv = inner_cv,
            n_iter = 100,
            n_jobs = -1, # parallélisation
            verbose = 1, # pour voir les logs
            random_state=42
        )
        # entrainement du modèle sur les données d'entraînement avec un lot d'hyperparamètres aléatoires
        random_search.fit(X_train_outer, y_train_outer)
        # récupération du meilleur modèle et de ses hyperparamètres
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score  = random_search.best_score_

        # évaluation des performances du modèle sur les données d'entrainement (folds externe)
        best_model.fit(X_train_outer, y_train_outer)
        y_pred_outer = best_model.predict(X_test_outer) # prediction sur le jeu de données de test qui n'a encore été jamais vu

        # application de l'AD et calcul du R²
        dict_perf = evaluate_MLP(X_train_outer, X_test_outer, y_test_outer, y_pred_outer, knn, threshold_strict, threshold_soft)

        fold_result = {
            'fold': fold_id,
            'best_params': best_params,
            'best_score_innerCV': best_score,
            'r2_noAD': dict_perf["no_AD"][0],
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

    df_results = pd.DataFrame(results_per_fold)
    df_results.to_csv(f"nestedCV_final_results_config_{num_config}.csv", index=False)
    print(df_results)
