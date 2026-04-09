import kagglehub
import pandas as pd
import os
import numpy as np
import re

"""
Clean dataframe : 
- Set AppID to index
- Change order of columns
- Change to binary : Header image, Screenshots, Website, Support url, Support email, About the game
- Remove : Notes, Metacritic url
- Add : Ratio positive rate = Positive/(Positive + Negative), then delete Negative
- Remove Nan in Genres, Categories
- Double name : keep highest Peak CCU amon the games with the same name, same developpers, same release date
- Change to list : Genres, Tags, languages, Categories
- Count Reviews
"""

def clean_and_split(s):
    # remove brackets
    s = s.strip()[1:-1]
    while "&amp;" in s:
        s = s.replace("&amp;", "&")
    # basic cleaning
    s = (s.replace("&lt;strong&gt;", "")
           .replace("&lt;/strong&gt;", "")
           .replace("<strong>", "")
           .replace("</strong>", "")
           .replace("ltstronggt", "")
           .replace("lt/stronggt", "")
           .replace("&lt;", "")
           .replace("&gt;", "")
           .replace("[b]", "")
           .replace("[/b]", "")
           .replace("<br>", ",")
           .replace("br /", ",")
           .replace("\\r\\n", ",")
           .replace("\r\n", ",")
           .replace("\n", ",")
         )

    # split into list
    parts = s.split(",")

    # clean each element
    clean_list = []
    for p in parts:
        p = p.strip().strip("'").strip('"')
        if p:
            clean_list.append(p)

    return clean_list

def to_list(x):
    if isinstance(x, str):
        return [i.strip() for i in x.split(",") if i.strip()]
    return []

def count_reviews(text):
    if pd.isna(text) or text == "":
        return 0
    
    text = str(text)
    
    # split by common separators
    parts = re.split(r"\.\.\.| - |–|—", text)
    
    # clean parts
    parts = [p.strip() for p in parts if p.strip()]
    
    return len(parts)


# Download latest version
def download_df():
    path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
    
    files = os.listdir(path)
    csv_file = [f for f in files if f == 'games.csv']
    if csv_file:
        data = pd.read_csv(os.path.join(path, csv_file[0]), index_col=False) 
    else:
        print("games.csv not found in the directory.")
    return data

def change_col(data):
    cols = data.columns.values
    colsNew = np.delete(cols,7)
    colsNew = np.insert(colsNew,7,'Discount')
    colsNew = np.insert(colsNew,8,'DLC count')
    colsNew = colsNew[:-1]
    data.columns = colsNew
    return data

def clean():
    data = download_df()
    data = change_col(data)
    data = data.set_index("AppID")
    
    # Transformer en 0/1
    cols = ["Header image", "Screenshots", "Website", "Support url", "Support email", "About the game"]
    for col in cols:
        data[f"has_{col.lower()}"] = (
            data[col].notna() & (data[col] != "")
        ).astype(int)
    data = data.drop(columns=cols)
    # Enlever les variables Notes, Metacritic url
    data = data.drop(columns=["Notes", "Metacritic url"])
    
    # Ratio positive vote
    total = data["Positive"] + data["Negative"]
    data["Ratio positive vote"] = np.where(
        total == 0,
        np.nan,  # or 0
        data["Positive"] / total
    )
    data = data.drop(columns=["Negative"])

    #Remove NaN
    data = data.dropna(subset=['Genres', 'Categories'])

    #Double name
    data = (data.sort_values('Peak CCU', ascending=False)
        .drop_duplicates(subset=['Name', 'Developers', 'Release date'], keep='first'))
    
    # List variables
    data["Genres"] = data["Genres"].apply(to_list)
    data["Tags"] = data["Tags"].apply(to_list)
    data["Categories"] = data["Categories"].apply(to_list)
    data["Supported languages"] = data["Supported languages"].apply(clean_and_split)
    data["Full audio languages"] = data["Full audio languages"].apply(clean_and_split)
    data["Reviews"] = data["Reviews"].apply(count_reviews)
    
    return data

data = clean()
data.to_csv('clean.csv')