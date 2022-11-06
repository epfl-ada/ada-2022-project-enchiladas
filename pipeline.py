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
from pickling import pickle_load
print("import completed")



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
df_ba_beers.set_index(df_ba_beers['beer_id'], inplace = True)
df_ba_beers.pop('beer_id')
df_ba_beers.sort_index(inplace=True)
df_ba_beers.head()
# %%
# Check for null values
df_ba_beers.isna().sum()
# We do have a few nan values in avg, ba_score, bros_score, abv, avg_comuted,zsocre and avg_matched_valid_ratings
# %%
# Are the avg nan values unrated beers?
df_ba_beers[df_ba_beers["avg"].isna() & df_ba_beers["nbr_ratings"]!=0 ].head()
# Yes, they are. O/w this dataframe would not be empty.

# %%
# Give basic statistics about the ba beer table
df_ba_beers.describe()
# TODO plot some interesting values like ratings vs review distribution...

# %% [markdown]
# ### Breweries
# - brewery_id -> id
# - location
# - name
# - nbr_beers
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
# TODO: describe the difference between ratings and reviews
df_ba_reviews = pickle_load(path_ba + "reviews.txt")
df_ba_reviews.head()
# %% [markdown]
# ### Ratings
df_ba_reviews = pickle_load(path_ba + "ratings.txt")
df_ba_reviews.head()


# %% [markdown]
# ## Rate Beer

# %% [markdown]
# ### Beers

# - beer_id
# - beer_name
# - brewery_name
# - style
# - nbr_ratings
# - overall_score (A score that ranks this beer against all other beers on RateBeer (max=100))
# - style_score (A score that ranks this beer against all beers within its own style (max=100))
# - avg (weighted average)
# - abv (alcohol content)
# - avg_computed (average rating)
# - zscore
# - nbr_matched_valid_ratings
# - avg_matched_valid_ratings

# %%
#read in beers.csv
df_rb_beers = pd.read_csv(path_rb + 'beers.csv')
df_rb_beers.head()

print("Total # of beers: ", df_rb_beers[df_rb_beers.columns[0]].count())

# %% 
#check if # of elements missing

for column in df_rb_beers:
    print(column, " : ", df_rb_beers[column].isnull().sum())

# %%
# check if any beers appear multiple times
# we should not sum them up... because avg depends non-linearly on number of ratings

df_rb_beers_same_name = df_rb_beers[df_rb_beers["beer_name"].duplicated(keep=False)]
df_rb_beers_same_name.sort_values(by=["beer_name"])
print("# of beers with identical name: ", df_rb_beers_same_name[df_rb_beers_same_name.columns[0]].count())

# but we have beers with same name from different breweries
# only make list with same name and same brewery

df_rb_beers_same_name_and_brewery = df_rb_beers_same_name[df_rb_beers_same_name["brewery_name"].duplicated(keep=False)]
df_rb_beers_same_name_and_brewery.sort_values(by="beer_name")
print("# of beers with identical name & brewery: ", df_rb_beers_same_name_and_brewery[df_rb_beers_same_name_and_brewery.columns[0]].count())

# %% 
# list all beer styles

rb_styles = list(df_rb_beers["style"].unique())
print("# of beer styles", len(rb_styles))

# %%
# check if there is any difference etween avg_computed and avg_matched valid_ratings



# %% [markdown]
# ### Breweries
# %% [markdown]
# ### Users
# %% [markdown]
# ### Reviews
df_rb_reviews = pickle_load(path_ba + "reviews.txt")
df_rb_reviews.head()
# %% [markdown]
# ### Ratings
df_rb_ratings = pickle_load(path_rb + "ratings.txt")
df_rb_ratings.head()

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

