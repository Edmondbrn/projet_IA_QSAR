import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint, uniform
import qsar_utils_rfr as qu  # Assurez-vous que les fonctions nécessaires sont dans ce module
import pickle

# Charger les données
df_norm = pd.read_csv("data/qsar_fish_toxicity_norm_train.csv")
LC50 = df_norm["LC50"]
df_norm = df_norm.drop(columns=["LC50"])

# Définir les distributions des hyperparamètres à tester
param_distributions = {
    'n_estimators': randint(100, 10000),           # Nombre d'arbres
    'max_depth': randint(1, 100),                # Profondeur maximale
    'min_samples_split': randint(2, 20),        # Nombre minimum d'échantillons pour diviser un nœud
    'min_samples_leaf': randint(1, 10)       # Nombre minimum d'échantillons dans une feuille
         # Fraction des caractéristiques à considérer pour chaque split
}

# Exécuter la validation croisée imbriquée
best_models = qu.run_nested_cv_rf(
data=df_norm,
target=LC50,
param_distributions=param_distributions,
n_splits=5,
n_iter=200,
random_state=42
)
