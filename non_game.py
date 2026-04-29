import clean_df
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## import data
data = clean_df.clean()
genres = data["Genres"]
genres_dummies = genres.explode().str.get_dummies().groupby(level=0).max()

## remove rare modalities
genres_dummies_clean = genres_dummies.drop(columns = ["360 Video", "Documentary", "Episodic", "Movie", "Short", "Tutorial"])
genres_dummies_clean = genres_dummies_clean[genres_dummies_clean.sum(axis=1) != 0]

## remove non-game individuals
# 1. Define lists
suspect_list = ['Accounting', 'Animation & Modeling', 'Audio Production', 
                'Photo Editing', 'Software Training', 'Utilities', 
                'Video Production', 'Web Publishing']  # List of applications non game

strong_game_anchors = ['Action', 'Adventure', 'RPG', 'Strategy', 'Racing', 'Sports'] # List of app pure game
weak_game_anchors = ['Indie', 'Casual', 'Simulation','Gore', 'Sexual Content', 'Violent', 'Nudity','Massively Multiplayer'] # List of mixed genres
game_anchors = strong_game_anchors + weak_game_anchors

# 2. Identify pure software of weak hybrid games
is_weak_only = (
    (genres_dummies_clean[weak_game_anchors].sum(axis=1) > 0) &
    (genres_dummies_clean[strong_game_anchors].sum(axis=1) == 0)
)
is_game = genres_dummies_clean[game_anchors].sum(axis=1) > 0
has_software_tag = genres_dummies_clean[suspect_list].sum(axis=1) > 0
weak_hybrid = is_weak_only & has_software_tag

# 3. Clean the dataframe
df_final = genres_dummies_clean[is_game & ~(weak_hybrid)].copy()

## Final dataset
df_non_game = data.loc[df_final.index]