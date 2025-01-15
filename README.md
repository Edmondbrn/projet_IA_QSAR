# Projet mise en place d'un modèle QSAR pour prédire la toxicité d'une molécule pour les poissons
Le but premier de ce projet est de mettre en place 3 modèles QSAR différents pour prédire la LC50 de molécules chimiques. Nous avons utilisé une régression linéaire pour obtenir des performances étalons à dépasser avec les autres modèles. Un RandomForest et un Perceptron multicouche (MLP) ont été mis ua point afin de comparer les méthodes de machine-learning classique et les réseaux de neurones dans le contexte d'un modèle QSAR.


## Installation

Vous pourrez trouver un fichier d'environnement virtuel Anaconda pour mettre au point les dépendances du projet.

Vous pouvez créer l'environnement avec la commande suivante:
```bash
conda env create -f environnement.yml
```

Et l'activer avec :
```bash
conda activate <env_name>
```

## Fichiers
- Perceptron multicouche

`def_AD_opti_param.ipynb`: Fichier pour définir le domain d'applicabilit" du MLP et lancer l'optimisationd es hyperparamètres et visualiser les résultats

`regression_MLP_evaluation_modeles.ipynb`: Fichier pour évaluer et sélectionner le meilleur modèle de MLP après l'optimisation

- Regréssion linéaire

`regresssion_lineaire.ipynb`: Fichier pour obtenir les performances du modèle de régression

- Exploration des données

`visualisation_data.ipynb`: Fichier pour avoir un bref aperçu des données et de leur qualité

## Auteurs
- [@Berne Edmond](https://github.com/Edmondbrn)
- [@Martin François](https://github.com/exovie)



