import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def q2_score(y_true, y_pred):
    """
    Calcule le Q² = 1 - SSE/SST
    où SSE = sum of squared errors,
        SST = sum of squared differences from the mean.
    """
    sse = np.sum((y_pred - y_true)**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    return 1 - sse/sst if sst != 0 else 0

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