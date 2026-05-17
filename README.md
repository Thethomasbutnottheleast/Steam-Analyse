# Steam-Analyse
Nous avons décidé d'analyser la base de données steam trouvable sur ce lien https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data en tant que projet d'Analyse de Données.

Pour cela, nous avons commencé par une analyse exploratoire du jeu de données, retrouvable sur steam_python.ipynb, suivi d'une analyse temporelle sur Time evolution.ipynb, en faisant un pré-traitement des données directement sur les notebooks.
Ensuite, nous avons décidé d'utiliser la même base de données mise à jour clean_df.py pour toute la suite du projet:

La première question que nous posons c'est sur la structure du marché des jeux. L'idée est de faire une réduction de dimension via MCA sur des genres afin de trouver des axes latents expliquant le marché. Une fois que l'espace MCA était construit, nous avons appliqué des méthodes de clustering (Kmeans et AHC) pour trouver des groups des jeux dans le marché. Dans l'analyse, nous avons utilisé des graphs en 3D et les avons sauvegardés sous les fichier html pour visualiser la structure du marché ainsi que le résultat de clustering.

La deuxième question porte sur le succès commercial des jeux. L'objectif est de construire et valider un score synthétique de succès combinant popularité (log_owners), 
recommandations (log_recommendations) et qualité perçue (wilson score). 

Une fois le score construit, nous validons sa robustesse via un test de sensibilité aux pondérations et sa cohérence 
via une ACP sur les features du jeu. Un clustering hiérarchique permet d'identifier des archétypes de jeux et de dégager des profils distincts de succès commercial.
Finalement, nous regardons l'effet individuel de chaque facteur (prix, DLC, languages, plateforme) sur le succès via des tests statistiques et visualisations.
