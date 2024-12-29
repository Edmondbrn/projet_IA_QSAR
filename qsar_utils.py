import matplotlib.pyplot as plt
import seaborn as sns



def scatter_QSAR(x_data : list[float], 
                 y_data : list[float], 
                 corr_coef : float,
                 xlim : tuple[float, float] = (0,10),
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
    plt.xlim(xlim)
    sns.regplot(x = x_data, y = y_data, color = dot_color, line_kws={"color" : line_color, "linestyle" : line_style})
    plt.xlabel("LC50 prédite", fontsize = 14)
    plt.ylabel("LC50 expérimentale", fontsize = 14)
    plt.title(f"Droite de régression depuis un modèle {model_type} entre \nla LC50 prédite et l'expérimentale", fontsize = 15)
    plt.tick_params(labelsize = 12)
    plt.text(0.05, 0.95, f"r = {corr_coef:.2f}", color = "red", fontsize = 14, ha = "left", va = "center", transform = plt.gca().transAxes ) # transform sert à indiquer que les coordonnées x et y sont relatives et non absolues
    plt.show()