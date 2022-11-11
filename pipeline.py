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
from helpers import *
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
# - should we combine or merge rb and ba? On what can we merge? Do both ds have the same set of beer-styles?
# - Sanity check: do number of reviews in users table match number of ratings or number of reviews?
# - data cleaning (null values, duplicates, ...)
# - completeness
# - data types
# - language
# - country-code (England-uk?)
# - matching avg
# - crawl beer styles from https://www.beeradvocate.com/beer/styles/ and https://www.ratebeer.com/beerstyles/ (for some textual description, we do have style columns in beer tables)
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
#
# Source of column description: https://www.beeradvocate.com/community/threads/beeradvocate-ratings-explained.184726/
#<
#
# - beer_id
# - name
# - brewery_id
# - brewery_name
# - style
# - nbr_ratings
# - nbr_reviews
# - avg
# - ba_score (The BA Score is the beer's overall score based on its ranking within its style category. It's based on the beer's truncated (trimmed) mean and a custom Bayesian (weighted rank) formula that takes the beer's style into consideration. Its purpose is to provide consumers with a quick reference using a format that's familiar to the wine and liquor worlds.)
#       95-100 = world-class
#       90-94 = outstanding
#       85-89 = very good
#       80-84 = good
#       70-79 = okay
#       60-69 = poor
#       < 60 = awful
# - bros_score
# - abv (alcohol by volume)
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
print(df_ba_beers.isna().sum())
print(f"number of beers in the dataframe: {len(df_ba_beers)}")
# We do have a few nan values in avg, ba_score, bros_score, abv, avg_comuted,zscore and avg_matched_valid_ratings
# According to https://www.beeradvocate.com/community/threads/beeradvocate-ratings-explained.184726/ it may be, that averages are not computed for beers with less than 10 reviews/ratings (not sure what the difference is)
# %%
# Are the avg nan values unrated beers?
df_ba_beers[df_ba_beers["avg"].isna() & df_ba_beers["nbr_ratings"]!=0 ].head()
# Yes, they are. O/w this dataframe would not be empty.
# Do these beers have reviews?
df_ba_beers[df_ba_beers["avg"].isna() & df_ba_beers["nbr_reviews"]!=0 ].head()
# Neither.

# %%
# What is the column "zscore" the zscore of? avg or ba_score ? 
#avgscore_mean = df_ba_beers["avg_computed"].mean()
#avgscore_var = df_ba_beers["avg_computed"].var()
#df_ba_beers["zscore_avg"] = df_ba_beers["avg"].apply(lambda x: (x-avgscore_mean)/avgscore_var)
#df_ba_beers[~df_ba_beers['zscore'].isnull()]
# %%
# Give basic statistics about the ba beer table
df_ba_beers.describe()
# TODO plot some interesting values like ratings vs review distribution...

# %% [markdown] 
# ###### data visualizations
# Histogram of alcohol content
plt.hist(df_ba_beers['abv'].dropna(), bins=25, range=(0, 20))
plt.xlabel('alcohol content in [%]')
plt.ylabel('Number of beers')
plt.title('Alcohol content histogram');
plt.show()

# %% [markdown]
# Number of beers per beer style
style_counts = df_ba_beers["style"].value_counts()
print("Number of beerstyles in BA dataset",len(style_counts))
fig, ax = plt.subplots(figsize = (5, 20))
ax.barh(style_counts.index, style_counts, align='center')
#ax.set_yticks(style_counts.index(), labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('number of beer')
ax.set_title('number of beer per category')
plt.show()

# %%
# american and german styles:
american_style = [x for x in style_counts.index if "American" in x]
german_style = [x for x in style_counts.index if ("German" in x or "Munich" in x)]
print("Number of american beerstyles in BA dataset",len(american_style))
print(american_style)
print("Number of german beerstyles in BA dataset",len(german_style))
print(german_style)

# %% [markdown]
# Alcohol content among some of the most popular beer styles
result = df_ba_beers.groupby(df_ba_beers["style"]).filter(lambda x: len(x) > 8000)
fig, ax = plt.subplots(figsize=(10,10))
sns.boxplot(x='style', y='abv', data=result, ax=ax)
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
plt.title("alcohol content for top 7 beer style")
plt.show()

# %% [markdown]
# Let's create a beer ranking
df_ba_beers.loc[df_ba_beers['nbr_reviews'] > 100].sort_values(by = "avg", ascending = False).head(10) 

# information about "Bud Light" - according to Matthieu most popular us beer (double check))
df_ba_beers.loc[df_ba_beers["beer_name"] == 'Bud Light']

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
# #### Breweries location processing
# TODO: document what's going on...

# %%
#removing html tags
df_ba_breweries['location'] = df_ba_breweries['location'].apply(lambda x : x.split('<', 1)[0])
#separating country from states (for USA)
df_ba_breweries['country'] = df_ba_breweries['location'].apply(lambda x : x.split(',', 1)[0])
country_list = df_ba_breweries['country'].unique()
#the function find_iso do a matching of common "non-official" country name to their iso code using the pycountry library
#For example, "united states" is matched to "United states of America" and is assigned the iso code USA
country_iso = {x: find_iso(x) for x in country_list}
df_ba_breweries['country_code'] = df_ba_breweries['country'].apply(lambda x: country_iso[x])
print(df_ba_breweries.head())


# %%
print("number of missing country codes:",df_ba_breweries['country_code'].isna().sum())
print(df_ba_breweries.isna().sum())
# TODO only country codes are unknown for a few breweries...

# %% [markdown]
# ##### Create a us-specific dataframe
#special detailed dataframe for usa
df_ba_breweries_usa = df_ba_breweries[df_ba_breweries['country'] == 'United States']
states = df_ba_breweries_usa['location'].apply(lambda x : str(x.split(',', 1)[1])[1:] if ',' in x else None)
df_ba_breweries_usa = df_ba_breweries_usa.assign(states = states)
print(df_ba_breweries_usa.sample(10))

# %%
print("number of missing country codes:",df_ba_breweries_usa['country_code'].isna().sum())
print("number of unknown states",df_ba_breweries_usa['states'].isna().sum())
print(df_ba_breweries_usa.isna().sum())
# TODO Some states are unknown

# %% [markdown]
# Beer count per country
#beer count per country
beer_country = df_ba_breweries.groupby(df_ba_breweries["country_code"]).sum().sort_values(by = 'nbr_beers', ascending = False)
beer_states = df_ba_breweries_usa.groupby(df_ba_breweries_usa["states"]).sum().sort_values(by = 'nbr_beers', ascending = False)

# %% [markdown]
# ##### data visualizations
# Geographical distribution of beers
world = gpd.read_file(data_folder + "maps/world-administrative-boundaries.shp", encoding = 'utf-8')
world = world.merge(beer_country, how = "left", left_on="iso3", right_index = True, )
world = world.sort_values(by = "nbr_beers", ascending = False)
world['log_beers'] = world['nbr_beers'].apply(lambda x: np.log10(x) if x >= 1 else 0)
#number of beers per country (log)
scheme = mc.FisherJenks(world['log_beers'], k=8)
gplt.choropleth(world, hue="log_beers", legend=True, scheme=scheme)
plt.title("nb of beer per country (log10)")
plt.show()

# %% [markdown]
# Number of beers per state (USA)
contiguous_usa = gpd.read_file(gplt.datasets.get_path("contiguous_usa"))
contiguous_usa = contiguous_usa.merge(beer_states, how = "left", left_on="state", right_index = True)
contiguous_usa = contiguous_usa.sort_values(by = "nbr_beers", ascending = False)
contiguous_usa["beer_per_capita"] = contiguous_usa.apply(lambda x: 0 if x["nbr_beers"] == np.nan else x["nbr_beers"]/x["population"], axis = 1)
gplt.choropleth(contiguous_usa, hue="nbr_beers", legend=True)
plt.title("nb of beer per state")
plt.show()

# %% [markdown]
# ### Users
# - nbr_ratings
# - nbr_reviews
# - user_id
# - user_name
# - joined
# - location
df_ba_users = pd.read_csv(path_ba + "users.csv")
print(df_ba_users.head())
print(df_ba_users.isna().sum())
# We do not know 1 username, 2652 join dates and 31279 locations
# TODO: 1 username are you sure ? isna().sum() is 1 for user_name...

# %%
# TODO: location processing
#removing html tags
df_ba_users = df_ba_users[~df_ba_users.isnull().any(axis=1)]
df_ba_users['location'] = df_ba_users['location'].apply(lambda x : x.split('<', 1)[0])
#separating country from states (for USA)
df_ba_users['country'] = df_ba_users['location'].apply(lambda x : x.split(',', 1)[0])
country_list = df_ba_users['country'].unique()
country_iso = {x: find_iso(x) for x in country_list} #look up table
df_ba_users['country_code'] = df_ba_users['country'].apply(lambda x: country_iso[x])
print(df_ba_users.head())

# %%
print("number of missing country codes:",df_ba_users['country_code'].isna().sum())
print(df_ba_users.isna().sum())
# 54 missing country codes... TODO

# %%
df_ba_users_usa = df_ba_users[df_ba_users['country'] == 'United States']
states = df_ba_users_usa['location'].apply(lambda x : str(x.split(',', 1)[1])[1:] if ',' in x else None)
df_ba_users_usa = df_ba_users_usa.assign(states = states)
print(df_ba_users_usa.sample(10))

# %%
users_states = df_ba_users_usa.groupby(df_ba_users_usa["states"]).sum().sort_values(by = 'nbr_ratings', ascending = False)
users_contiguous_usa = gpd.read_file(gplt.datasets.get_path("contiguous_usa"))
users_contiguous_usa = users_contiguous_usa.merge(users_states, how = "left", left_on="state", right_index = True)
print(users_contiguous_usa.sample(10))
users_contiguous_usa = users_contiguous_usa.sort_values(by = "nbr_ratings", ascending = False)
#TODO: not done yet...
users_contiguous_usa["ratings_per_capita"] = users_contiguous_usa.apply(lambda x: 0 if x["nbr_ratings"] == np.nan else x["nbr_ratings"]/x["population"], axis = 1)
gplt.choropleth(users_contiguous_usa, hue="ratings_per_capita", legend=True)
plt.title("nb of ratings per capita per state")
plt.show()

# %% [markdown]
# Number of reviews per user
# ba_users_review_counts = df_ba_users['nbr_reviews'].value_counts()
# logbins = np.geomspace(ba_users_review_counts.min(), ba_users_review_counts.max(), 100)
# print(logbins)
# plt.hist(ba_users_review_counts,  density=True, log=True,bins=logbins)
# plt.xscale("log")
# plt.xlabel("Number of reviews per user")
# plt.ylabel("Frequency")
# plt.title("Distribution of number of review per users")
# plt.show()

nonzero_nbr_reviews = df_ba_users.loc[df_ba_users['nbr_reviews'] > 0 & df_ba_users['nbr_reviews'].notna()]['nbr_reviews']
logbins = np.geomspace(nonzero_nbr_reviews.min(), nonzero_nbr_reviews.max(), 20)
nonzero_nbr_reviews.plot(kind='hist', logx=True, logy=True,density=True, bins=logbins)
plt.xlabel("number of reviews")
plt.ylabel("number of users")
plt.title("number of review per users")
plt.show()
# TODO: something seems wrong about this plot. Are there no users with less than 100 reviews?
# TODO: plot the same for ratings
# TODO: what is the difference between ratings and reviews?

df_ba_users["nbr_ratings"].plot(kind='hist', logx=True, logy=True,density=True, bins=1000)
plt.xlabel("number of ratings")
plt.ylabel("number of users")
plt.title("number of ratings per users")
plt.show()

# %% [markdown]
# Sanity Check
# Number of users with 4 reviews/ratings
print("number of users with 4 reviews:",df_ba_users[df_ba_users["nbr_reviews"] == 4].shape[0])
print("number of users with 4 ratings:",df_ba_users[df_ba_users["nbr_ratings"] == 4].shape[0])
print("number of users with 2 reviews:",df_ba_users[df_ba_users["nbr_reviews"] == 2].shape[0])
print("number of users with 2 ratings:",df_ba_users[df_ba_users["nbr_ratings"] == 2].shape[0])
# Are users unique?
df_ba_users_grouped_by_id = df_ba_users.groupby(df_ba_users["user_id"])
print(df_ba_users_grouped_by_id.head())
print(df_ba_users.head())
# We lose 4 users when grouping by user_id


# %% [markdown]
# ### Reviews
# - beer_name
# - beer_id
# - brewery_name
# - brewery_id
# - style
# - abv
# - date
# - user_name
# - user_id
# - appearance
# - aroma
# - palate
# - taste
# - overall
# - rating
# - text (dropped in pickling)
df_ba_reviews = pickle_load(path_ba + "reviews.txt")
df_ba_reviews.set_index(df_ba_reviews["date"], inplace = True)
df_ba_reviews.head()

# %%
df_ba_reviews.describe()
df_ba_reviews.isna().sum()

# %% [markdown]
# #### Data visualizations
# BA reviews as time-series
ba_reviews_month = df_ba_reviews.groupby(df_ba_reviews.index.to_period('M')).size()
ba_reviews_month.loc[ba_reviews_month.index.year > 2010].plot()


# %% [markdown]
# ### Ratings
# - beer_name
# - beer_id
# - brewery_name
# - brewery_id
# - style
# - abv
# - date
# - user_name
# - user_id
# - appearance
# - aroma
# - palate
# - taste
# - overall
# - rating
# - text (dropped in pickling)
# - review (dropped in pickling)
df_ba_ratings = pickle_load(path_ba + "ratings.txt")
df_ba_ratings.set_index(df_ba_ratings["date"], inplace = True)
df_ba_ratings.head()

# %%
df_ba_ratings.describe()
df_ba_ratings.isna().sum()

# %% [markdown]
# #### Data visualizations
# BA ratings as time-series
ba_ratings_month = df_ba_reviews.groupby(df_ba_reviews.index.to_period('M')).size()
ba_ratings_month.plot()

# %% [markdown]
# #### Ratings vs Reviews
# Compute intersection
df_ba_ratings_reviews = df_ba_ratings.merge(df_ba_reviews, how = "inner", on = ["beer_name", "beer_id", "brewery_name", "brewery_id", "style", "abv",  "user_name", "user_id", "appearance", "aroma", "palate", "taste", "overall", "rating"])

# %%
df_ba_ratings_reviews.describe()
df_ba_ratings_reviews.isna().sum()
# %%
print("size of intersection in review/ratings BA", df_ba_ratings_reviews[~df_ba_ratings_reviews.isnull().any(axis=1)].count())
print("size of reviews BA", df_ba_reviews[~df_ba_reviews.isnull().any(axis=1)].count())
# Reviews are a subset of ratings for the BA dataset!

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

# print("Total # of beers: ", df_rb_beers[df_rb_beers.columns[0]].count())

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
# how important are those beers
print("# of ratings for beers with identical name & brewery: ", df_rb_beers_same_name_and_brewery["nbr_ratings"].sum())
print("gives an average of {} ratings/beer.".format(df_rb_beers_same_name_and_brewery["nbr_ratings"].sum()/df_rb_beers_same_name_and_brewery[df_rb_beers_same_name_and_brewery.columns[0]].count()))

# %% 
# list all beer styles

rb_styles = df_rb_beers["style"].value_counts()
print("There exist {} different beer styles".format(len(rb_styles)))

fig, ax = plt.subplots()
ax.barh(rb_styles.index, rb_styles, align='center')
ax.invert_yaxis()  # labels read top-to-bottom
ax.tick_params(axis='y', which='major', labelsize=4)
ax.set_xlabel('numbers of beer')
ax.set_ylabel("Beer style")
ax.set_title('number of beer per style')
plt.show()

# %%
# check if there is any difference etween avg_computed and avg_matched valid_ratings

no_match = ~df_rb_beers["avg_matched_valid_ratings"].isnull() & df_rb_beers["avg_matched_valid_ratings"].ne(df_rb_beers["avg_computed"])
print("there are {} dataframes where avg_matched_valid_ratings is not NaN or not equal avg_computed".format(df_rb_beers[no_match].shape[0]))

# %%
# how many ratings did the beers get?

fig, ax = plt.subplots()
ax.hist(df_rb_beers["nbr_ratings"], bins=1000, log=True)
ax.set_xlabel("Number of ratings")
ax.set_ylabel("Number of beers")
plt.show()

df_rb_beers.sort_values(by=["nbr_ratings"], ascending=False).head()

# %% 
# how many overall ratings per style
rb_ratings = df_rb_beers.groupby(["style"])["nbr_ratings"].sum()
# Average Rating per beer per beer style

rb_avg_rating_per_style = rb_ratings/rb_styles.sort_index()

fig, ax = plt.subplots()
ax.barh(rb_avg_rating_per_style.index, rb_avg_rating_per_style, align='center')
# ax.invert_yaxis()  # labels read top-to-bottom
ax.tick_params(axis='y', which='major', labelsize=4)
ax.set_xlabel('Average ratings per beer per style')
ax.set_ylabel("Beer style")
ax.set_title('Average ratings per beer per style')
plt.show()



# %% [markdown]
# ### Breweries

df_rb_brew = pd.read_csv(path_rb + 'breweries.csv')
df_rb_brew.head()

for column in df_rb_brew:
        print(column, " : ", df_rb_brew[column].isnull().sum())

# split US states
df_rb_brew['country'] = df_rb_brew['location'].apply(lambda x : x.split(',', 1)[0])


# %% 
# add country code
country_list = df_rb_brew['country'].unique()
country_iso = {x: find_iso(x) for x in country_list} #look up table
df_rb_brew['country_code'] = df_rb_brew['country'].apply(lambda x: country_iso[x])

# %%

df_rb_brew_per_country = df_rb_brew.groupby(["country_code"])["country_code"].count()
print(df_rb_brew_per_country.head())
df_rb_beers_per_country = df_rb_brew.groupby(["country_code"])["nbr_beers"].sum()
print(df_rb_beers_per_country.head())

# %%
# nb of breweries per country

print(type(df_rb_brew_per_country.sort_values(ascending=False).head()))
# Uncomment me for a world map:

# world = gpd.read_file(data_folder + "maps/world-administrative-boundaries.shp", encoding = 'utf-8')
# world = world.merge(df_rb_brew_per_country, how = "left", left_on="iso3", right_index = True, )
# world = world.sort_values(by = "country_code", ascending = False)
# world['log_beers'] = world['country_code'].apply(lambda x: np.log10(x) if x >= 1 else 0)
# #number of breweries per country (log)
# scheme = mc.FisherJenks(world['log_beers'], k=8)
# gplt.choropleth(world, hue="log_beers", legend=True, scheme=scheme)
# plt.title("nb of beer per country (log10)")
# plt.show()

# %%
# nb of beers per country
print(df_rb_beers_per_country.sort_values(ascending=False).head())
# Uncomment me for a world map:

world = gpd.read_file(data_folder + "maps/world-administrative-boundaries.shp", encoding = 'utf-8')
world = world.merge(df_rb_beers_per_country, how = "left", left_on="iso3", right_index = True, )
world = world.sort_values(by = "nbr_beers", ascending = False)
world['log_beers'] = world['nbr_beers'].apply(lambda x: np.log10(x) if x >= 1 else 0)

#number of beers per country (log)
scheme = mc.FisherJenks(world['log_beers'], k=8)
gplt.choropleth(world, hue="log_beers", legend=True, scheme=scheme)
plt.title("nb of beer per country (log10)")
plt.show()

# %% [markdown]
# ### Users

df_rb_users = pd.read_csv(path_rb + 'users.csv')
df_rb_users.head()

for column in df_rb_users:
        print(column, " : ", df_rb_users[column].isnull().sum())

# remove lines with no location, there are still states with <joined> NaN
df_rb_users = df_rb_users[~df_rb_users["location"].isnull()]

# split US states
df_rb_users['country'] = df_rb_users['location'].apply(lambda x : x.split(',', 1)[0])

country_list = df_rb_users['country'].unique()
country_iso = {x: find_iso(x) for x in country_list} #look up table
df_rb_users['country_code'] = df_rb_users['country'].apply(lambda x: country_iso[x])

df_rb_users.head()

# %%
# print(df_rb_users["country_code"].unique())
print("No country code assignment to the following countries: ", df_rb_users[df_rb_users["country_code"].isnull()]["country"].unique())

# %%
# Users per country

rb_users_per_country = df_rb_users.groupby(["country"])["country"].count().sort_values(ascending=False)
print(type(rb_users_per_country))

# %% 
# Number of ratings per country

rb_ratings_per_country = df_rb_users.groupby(["country"])["nbr_ratings"].sum().sort_values(ascending=False)
print(type(rb_ratings_per_country))

# %%
# average nb of rating per user

rb_avg_rating_per_user_per_country = rb_ratings_per_country/rb_users_per_country

fig, ax = plt.subplots()
ax.hist(rb_avg_rating_per_user_per_country.sort_values(ascending=False).head(80), bins=200)
ax.set_xscale('log')
ax.set_xlabel("Avg ratings per user per country")
ax.set_ylabel("Number of countries")
ax.set_title("Top 80 countries with highest avg ratings per user")
plt.show()









# %% [markdown]
# ### Reviews
df_rb_reviews = pickle_load(path_rb + "reviews.txt")
df_rb_reviews.head()




# %% [markdown]
# ### Ratings
df_rb_ratings = pickle_load(path_rb + "ratings.txt")
df_rb_ratings.head()

# %%
df_rb_ratings.isna().sum()

# %% [markdown]
# ## Matched Dataset

# %% [markdown]
# ### Beers

#loading and naming columns
df_beers = pd.read_csv(path_md + "beers.csv", header=1)
df_beers = df_beers.drop(["ba_score", "bros_score", "nbr_reviews", "overall_score", "style_score"], axis=1) #dropping features that only exist in 1 dataset
df_beers = df_beers.rename(columns={key: f"{key}_ba" for key in df_beers.columns[0:13]}) #rename columns to highlight dataset of origine
df_beers = df_beers.rename(columns={key: f"{key[:-2]}_rb" for key in df_beers.columns[13:26]})

#sanity checks:
df_beers.describe()
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