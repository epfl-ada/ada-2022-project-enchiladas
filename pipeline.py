# More info on python interactive in vs code: https://code.visualstudio.com/docs/python/jupyter-support-py
# to have an outline install the CodeMap Extension
# follow this guide:
# https://github.com/oleg-shilo/codemap.vscode/wiki/Adding-custom-mappers
# Add the following to your settings.json for the codemap extension and .py files:
# "codemap.py": [
#         {
#             "pattern": "^(\\s*)# ##### (.*)",
#             "clear": "# #####",
#             "prefix": " ---",
#             "icon": "level5"
#           },
#         {
#             "pattern": "^(\\s*)# #### (.*)",
#             "clear": "# ####",
#             "prefix": " --",
#             "icon": "level4"
#           },
#           {
#             "pattern": "^(\\s*)# ### (.*)",
#             "clear": "# ###",
#             "prefix": " -",
#             "icon": "level3"
#           },
#           {
#             "pattern": "^(\\s*)# ## (.*)",
#             "clear": "# ##",
#             "prefix": " ",
#             "icon": "level2"
#           },
#           {
#             "pattern": "^(\\s*)# # (.*)",
#             "clear": "# #",
#             "prefix": "",
#             "icon": "level1"
#           }
#     ],








# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pycountry
import geoplot as gplt
import geopandas as gpd
import geoplot.crs as gcrs
import imageio
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import mapclassify as mc
import os.path
import numpy as np

# %% [markdown]
# # Helpers
# ## Pickle
# %%
def pickle_load(path):
    """
    path: path to original txt file

    loads the corresponding pickle file or creates it if it does not exist

    returns: dataframe with the files content
    """
    pickle_path = path + ".pickle"
    if os.path.exists(pickle_path):
        return pd.read_pickle(pickle_path)
    else:
        if "ratings" in path:
            # TODO: comment this, copied from Matthieu's code.
            # Does it only work for ratings? Does it work for all ratings?

            with open(path, "r", encoding='utf-8') as file:
                beer = {}
                review_list = []
                for line in file:
                    if line.isspace():
                        review_list.append(beer)
                        beer = {}
                    else:
                        beer[line.split()[0][:-1]] =" ".join(line.split()[1:])
            df = pd.DataFrame(review_list)
            
            #making dataframe
            for col in ["beer_id", "brewery_id", "appearance", "aroma", "palate", "taste", "overall", "rating", "abv"]:
                df[col] = pd.to_numeric(df[col], errors = 'ignore')
            df["date"] = pd.to_datetime(df["date"],unit='s')
            df.set_index(df["date"], inplace = True)
            df.to_pickle(pickle_path)
            return df
        else:
            df = pd.read_csv(path, sep=",", low_memory=False)
            df.to_pickle(pickle_path)
            return df



# %% [markdown]
# # Pipeline
# What was written on the sheet of paper:
# Data sources:
# - ratebeer, beer advocate, matched ds
# - how do they compare?
# - enrichement
# Data model:
# - table (maybe network)
# Data Wrangling:
# - column description
# - data cleaning (null values, duplicates, ...)
# - completeness
# - data types
# - language
# - country-code (England-uk?)
# - matching avg
# - beer style aliases
# Andreaas Idea:
# - vocabulary: <-> Location

# %%
#data folder path
data_folder = './Data/'
path_ba = data_folder + 'BeerAdvocate/'
path_rb = data_folder + 'RateBeer/'
path_md = data_folder + 'matched_beer_data/'


# %% [markdown]
# ## Beer Advocate

# %% [markdown]
# ### Beers
# - beer_id
# - name
# - brewery_id
# - brewery_name
# - style
# - nbr_ratings
# - nbr_reviews
# - avg
# - ba_score
# - bros_score
# - abv
# - avg_computed
# - zscore
# - nbr_matched_valid_ratings
# - avg_matched_valid_ratings

# %%
#read in beers.csv
df_ba_beers = pd.read_csv(path_ba + 'beers.csv')
df_ba_beers.head()


# %% [markdown]
# ### Breweries
df_ba_breweries = pd.read_csv(path_ba + "breweries.csv")
df_ba_breweries.set_index(df_ba_breweries['id'], inplace = True)
df_ba_breweries.pop('id')
df_ba_breweries.sort_index(inplace=True)
df_ba_breweries.head()

# %% [markdown]
# ### Users
df_ba_users = pd.read_csv(path_ba + "users.csv")
df_ba_users.head()
# %% [markdown]
# ### Reviews
# TODO: pickle...
# df_ba_reviews = pickle_load(path_ba + "reviews.txt")
# df_ba_reviews.head()
# %% [markdown]
# ### Ratings
# df_ba_reviews = pickle_load(path_ba + "ratings.txt")
# df_ba_reviews.head()


# %% [markdown]
# ## Rate Beer

# %% [markdown]
# ### Beers
# %% [markdown]
# ### Breweries
# %% [markdown]
# ### Users
# %% [markdown]
# ### Reviews
# %% [markdown]
# ### Ratings


# %% [markdown]
# ## Matched Dataset

# %% [markdown]
# ### Beers
# %% [markdown]
# ### Breweries
# %% [markdown]
# ### Users
# %% [markdown]
# ### Users Approx
# %% [markdown]
# ### Reviews
# %% [markdown]
# ### Ratings
# %% [markdown]
# #### Ratings with text ba
df_ratings_with_text_ba = pickle_load(path_md + "ratings_with_text_ba.txt")
# %% [markdown]
# #### Ratings with text rb

