# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pycountry # match country names to iso code
import geoplot as gplt #plotting maps
import geopandas as gpd
import geoplot.crs as gcrs
import imageio #not used ?
import pathlib #not used ?
import matplotlib.pyplot as plt
import mapclassify as mc #???
import os.path
from helpers import * #custom made functions
print("import completed")

# number of pandas columns to display
pd.set_option('display.max_columns', 50)


# %% [markdown]
# 
#  The data for the `Data` folder can be downloaded from [here](https://drive.google.com/drive/folders/1Wz6D2FM25ydFw_-41I9uTwG9uNsN4TCF)
# 
#  The initial directory structure for this notebook should be the following:
# 
#  ```
#  ├── Data/
#  │   ├── BeerAdvocate/
#  │       ├── beers.csv
#  |       ├── breweries.csv
#  |       ├── users.csv
#  |       ├── reviews.txt
#  |       ├── ratings.txt
#  │   ├── RateBeer/
#  │       ├── beers.csv
#  │       ├── breweries.csv
#  │       ├── users.csv
#  |       ├── reviews.txt
#  │       ├── ratings.txt
#  │   ├── matched_beer_data/
#  │       ├── beers.csv
#  |       ├── breweries.csv
#  |       ├── ratings.csv
#  |       ├── ratings_with_text_ba.txt
#  |       ├── ratings_with_text_rb.txt
#  |       ├── ratings_ba.txt
#  |       ├── ratings_rb.txt
#  |       ├── users.csv
#  |       ├── users_approx.csv
#  ├── README.md
#  ├── notebook.ipynb
#  ├── requirements.txt
#  ├── helpers.py
#  └── .gitignore
#  ```

# %%
#data folder path
data_folder = './Data/'
path_ba = data_folder + 'BeerAdvocate/'
path_rb = data_folder + 'RateBeer/'
path_md = data_folder + 'matched_beer_data/'


# %% [markdown]
#  # Table Description
# 
#  ### Beers
# 
#  Beer Advocate offers a partial column description [here](https://www.beeradvocate.com/community/threads/beeradvocate-ratings-explained.184726/)
#  [Here](https://www.beeradvocate.com/community/threads/how-to-review-a-beer.241156/) is a guide by Beer Advocate on how to review a beer.
# 
#  | Column name (RB/BA) | Data type | RB | BA | Description |
#  |---|---|---|---|---|
#  | beer_id | int | ✓ | ✓ | beer id |
#  | beer_name/name | str | ✓ | ✓ | name of the beer |
#  | brewery_id | int | ✓ | ✓ | id of the brewery making the beer |
#  | brewery_name | str | ✓ | ✓ | name of the brewery making the beer |
#  | style | str | ✓ | ✓ | specific sub-category of the beer. On review websites, beers are classified into general categories (lager, belgian ales, etc...) which are broken down into more detailed sub-categories. The list of the styles is not universal and is available on [Rate Beer](https://www.ratebeer.com/beerstyles) and [Beer Advocate](https://www.beeradvocate.com/beer/styles/) website's respectively |
#  | nbr_ratings | int | ✓ | ✓ | number of ratings received by the beer |
#  | overall_score | int | ✓ | - | score that ranks the beer against all other beers on RateBeer (max=100), uses ratings as well as total number of ratings |
#  | style_score | int | ✓ | - | score that ranks the beer against all beers within its own style (max=100), uses ratings as well as total number of ratings |
#  | nbr_reviews | int | - | ✓ | on Beer Advocate only, reviews are a subset of ratings which only includes user's review with a score on every category (taste, feel, etc...) as well as a text review|
#  | avg | float | ✓ | ✓ | for BA, this is the arithmetic mean (same as avg_computed). However, for RB, it is the average of scores weighted on the amount of reviews. The more reviews a beer gets, the closer the 'avg' gets to the 'avg_computed' (artihmetic mean)|
#  | ba_score | int | - | ✓ | BA Score is the beer's overall score based on its ranking within its style category. It's based on the beer's truncated (trimmed) mean and a custom Bayesian (weighted rank) formula that takes the beer's style into consideration. Its purpose is to provide consumers with a quick reference using a format that's familiar to the wine and liquor worlds. <br> 95-100 = world-class <br> 90-94 = outstanding <br> 85-89 = very good <br> 80-84 = good <br> 70-79 = okay <br> 60-69 = poor <br> < 60 = awful |
#  | bros_score | float | - | ✓ | Similar to the BA score but only based on the opinion of the website's founders (see [Who are the bros?](https://www.beeradvocate.com/community/threads/the-bros-vs-ba-score.307092/))|
#  | abv (alcohol by volume) | float | ✓ | ✓ | Alcohol content By Volume, metric of the percentage of pure ethanol contained in the beer |
#  | avg_computed | float | ✓ | ✓ | arithmetic means of ratings |
#  | zscore | float | ✓ | ✓ | z-score for the beer ratings |
#  | nbr_matched_valid_ratings | int | ✓ | ✓ | number of valid ratings for the beer (from matched dataset)|
#  | avg_matched_valid_ratings | int | ✓ | ✓ | average number of valid ratings for the beer (from matched dataset)|
# 
# 
#  ### Breweries
# 
#  | Column name (RB/BA) | Data type | RB | BA | Description |
#  |---|---|---|---|---|
#  | id/ brewery_id | int | ✓ | ✓ | brewery id |
#  | location | str | ✓ | ✓ | country of brewery (incl. state for the US) |
#  | name | str | ✓ | ✓ | name of the brewery |
#  | nbr_beers | int | ✓ | ✓ | number of different beers names produced by that brewery |
# 
# 
#  ### Users
# 
#  | Column name (RB/BA) | Data type | RB | BA | Description |
#  |---|---|---|---|---|
#  | nbr_ratings | int | ✓ | ✓ | number of ratings a user has given on the website |
#  | nbr_reviews | int | ✓ | ✓ | number of reviews a user has given on the website |
#  | user_id | int | ✓ | ✓ | user id |
#  | user_name | str | ✓ | ✓ | user name |
#  | joined | int | ✓ | ✓ | UNIX time stamp for user account creation |
#  | location | str | ✓ | ✓ | country of origin of the user (incl. state for the US) |
# 
# 
#  ### Ratings
# 
#  The difference in columns between reviews and ratings is only the boolean flag `reviews` for the BA dataset.
#  For RB there is no difference between reviews and ratings.
# 
#  | Column name (RB/BA) | Data type (RB/BA) | RB | BA | Description |
#  |---|---|---|---|---|
#  | beer_name | str | ✓ | ✓ | name of the beer |
#  | beer_id | int | ✓ | ✓ | id of the beer |
#  | brewery_name | str | ✓ | ✓ | name of the brewery which the beer belongs to |
#  | brewery_id | int | ✓ | ✓ | id of the brewery the beer belongs to |
#  | style | str | ✓ | ✓ | classification of the type of beer (website dependent) |
#  | abv | float | ✓ | ✓ | alcohol content by volume |
#  | date | int | ✓ | ✓ | UNIX time stamp for user account creation |
#  | user_name | str | ✓ | ✓ | name of the user giving the rating |
#  | user_id | int | ✓ | ✓ | id of the user giving the rating |
#  | appearance | int/float | ✓ | ✓ | rating for the appearance of the beer |
#  | aroma | int/float | ✓ | ✓ | rating for the aroma of the beer |
#  | palate | int/float | ✓ | ✓ | rating for the palate of the beer |
#  | taste | int/float | ✓ | ✓ | rating for the taste of the beer |
#  | overall | int/float | ✓ | - | score that ranks the beer against all other beers on RateBeer (max=100), uses ratings as well as total number of ratings - explained [here](https://www.ratebeer.com/our-scores#:~:text=Aroma%20and%20Taste%20are%20scored,of%205%20for%20each%20rating) |
#  | rating | float | ✓ | ✓ | this is the weighted average rating for the beer e.g see details for BA [here](https://www.beeradvocate.com/community/threads/how-to-review-a-beer.241156/) |
#  | text | str | ✓ | ✓ | text of the rating |
#  | review | bool | ✓ | - | true if the rating is a review |
# 

# %% [markdown]
#  # Pipeline
#  ## Subsetting to the matched beer/brewery data
#  We conduct our analyses separately for both the BeerAdvocate and RateBeer datasets in order to be able to compare our results
#  and check if they are robust between the two datasets. The goal of this is to try and remove an extra source of variation
#  due to the different user bases on the two websites.
# 
#  Based on the merged dataset, we subset to beers and breweries that are presents in both datasets, provided in `matched_beer_data/beers.csv` and `matched_beer_data/breweries.csv`.
#  We also conduct some preliminary cleaning and analysis of this matched dataset.
# 
#  Column dropping justification:
#  - `ba_score`: only available in BA dataset (internal BA metric for overall ranking)
#  - `bros_score`: only available in BA dataset (internal BA metric)
#  - `nbr_ratings`: need to be recomputed at a later stage since we are subseting ratings
#  - `nbr_reviews`: same argument as in `nbr_ratings`
#  - `overall_score`: only available in RB dataset (internal RB metric for overall ranking)
#  - `style_score`: only avaiable in RB dataset (internal RB metric for style ranking)
#  - `avg_matched_...` : not necessary for our analysis
# 
#  We also drop beers (rows) that don't have at least 1 review on each website (NaN avg score in either RB or BA dataset)
# 

# %% [markdown]
#  ### Load and filter merged beer/brewery data
#  #### Loading Beer Dataset (merged)

# %%
#loading and naming columns
df_beers = pd.read_csv(path_md + "beers.csv", header=1)
#dropping features that only exist in 1 dataset
df_beers = df_beers.drop(["ba_score", "bros_score", "nbr_reviews", "nbr_ratings", "nbr_ratings.1", "overall_score", "style_score", "avg_matched_valid_ratings", "nbr_matched_valid_ratings", "avg_matched_valid_ratings.1", "nbr_matched_valid_ratings.1", "zscore", "zscore.1"], axis=1)

df_beers = df_beers.rename(columns={key: f"{key}_ba" for key in df_beers.columns[0:9]}) #rename columns to highlight dataset of origin
df_beers = df_beers.rename(columns={key: f"{key[:-2]}_rb" for key in df_beers.columns[9:18]})

#nan checks:
print("--- Initial number of NaN Values: ---")
print(df_beers.isna().sum())
#NaN values in avg means a beer from one website doesn't have any review on the other website. => we only keep beers with at least 1 review on both website (drop NaNs)
df_beers = df_beers.loc[df_beers[["avg_ba","avg_computed_ba", "avg_rb", "avg_computed_rb"]].isna().sum(axis=1) == 0]
print(f"number of beers in matched dataset after cleaning {len(df_beers)}")
print("--- Number of NaN Values after Processing: ---")
print(df_beers.isna().sum())
#no NAN lefts !
print("--- Print Info ---")
print(df_beers.info())


# %% [markdown]
#  How many beer did we drop with respect to the Beer Advocate dataset?

# %%
df_ba_beers = pd.read_csv(path_ba + 'beers.csv')
df_rb_beers = pd.read_csv(path_rb + 'beers.csv')

print(f"number of beers in BeerAdvocate dataset {len(df_ba_beers)}")
print(f"number of beers in RateBeer dataset {len(df_rb_beers)}")
print(f"number of beers in matched dataset {len(df_beers)}")

# %% [markdown]
#  With respect to BA, we lost about 87% of the beers.
#  With respect to RB, we lost about 92% of the beers.
#  38k is still a significant number and this quantity suffices for our analysis

# %% [markdown]
#  Sanity checks
#  Are the abv contents the same in both datasets?

# %%
print(df_beers.loc[df_beers["abv_ba"] != df_beers["abv_rb"]])

# %% [markdown]
#  Do the names match ? Most of them don't. BA don't include the brewery name in the beer name but RB often does. We will keep the RB names since they are more informative.

# %%
df_beers.loc[df_beers["beer_name_ba"] != df_beers["beer_name_rb"]].head()


# %% [markdown]
#  ##### Include Style classes
# %%
style_lookup = dict([
    # taken from: https://www.beeradvocate.com/beer/styles/
    ('Bocks', ['Bock', 'Doppelbock', 'Eisbock', 'Mailbock', 'Weizenbock', 'Maibock / Helles Bock']),
    ('Brown Ales', ['Altbier', 'American Brown Ale', 'Belgian Dark Ale', 'English Brown Ale', 'English Dark Mild Ale']),
    ('Dark Ales', ['Belgian Strong Dark Ale','Dubbel', 'Roggenbier', 'Scottish Ale', 'Winter Warmer']),
    ('Dark Lagers', ['American Amber / Red Lager', 'Czech Amber Lawger','Czech Dark Lager', 'Munich Helles Lager', 'European Dark Lager', 'Märzen','Munich Dunkel', 'Munich Dunkel Lager', 'Rauchbier', 'Schwarzbier', 'Vienna Lager', 'Euro Dark Lager', 'Märzen / Oktoberfest', 'Rauchbier']),
    ('Hybrid Beers', ['Bière de Champagne / Bière Brut', 'Braggot', 'California Common / Steam Beer', 'Cream Ale']),
    ('India Pale Ales', ['English India Pale Ale (IPA)', 'American IPA', 'Belgian IPA', 'Black IPA', 'Brut IPA', 'English IPA', 'Imperial IPA', 'Milkshake IPA', 'New England IPA', 'American Double / Imperial IPA', 'American Black Ale']),
    ('Pale Ales', ['Belgian Strong Pale Ale', 'American Amber / Red Ale', 'Saison / Farmhouse Ale', 'American Blonde Ale', 'American Pale Ale','American Pale Ale (APA)', 'Belgian Blonde Ale', 'Belgian Pale Ale', 'Bière de Garde', 'English Bitter', 'English Pale Ale', 'English Pale Mild Ale', 'Extra Special / Strong Bitter (ESB)', 'Grisette', 'Irish Red Ale', 'Kölsch', 'Saison', 'American Pale Wheat Ale']),
    ('Pale Lagers', ['American Pale Lager', 'Euro Pale Lager', 'American Double / Imperial Pilsner', 'American Adjunct Lager', 'American Lager', 'Bohemian / Czech Pilsner', 'Czech Pale Lager', 'European / Dortmunder Export Lager', 'European Pale Lager', 'European Strong Lager', 'Festbier / Wiesnbier', 'German Pilsner', 'German Pilsener', 'Czech Pilsener', 'Helles', 'Imperial Pilsner', 'India Pale Lager (IPL)', 'Kellerbier / Zwickelbier', 'Light Lager', 'Malt Liquor', 'American Malt Liquor', 'Dortmunder / Export Lager', 'Euro Strong Lager']),
    ('Porters', ['American Porter', 'Baltic Porter', 'English Porter', 'Imperial Porter', 'Robust Porter', 'Smoked Porter']),
    ('Speciality Beers', ['Chile Beer', 'Fruit and Field Beer', 'Gruit / Ancient Herbed Ale', 'Happoshu', 'Herb and Spice Beer', 'Japanese Rice Lager', 'Kvass', 'Low-Alcohol Beer', 'Pumpkin Beer', 'Rye Beer', 'Sahti', 'Smoked Beer']),
    ('Stouts', ['American Imperial Stout', 'American Stout', 'English Stout', 'Foreign / Export Stout', 'Irish Dry Stout', 'Oatmeal Stout', 'Russian Imperial Stout', 'Sweet / Milk Stout', 'Milk / Sweet Stout', 'American Double / Imperial Stout', 'Black & Tan']),
    ('Strong Ales', ['American Barleywine', 'American Strong Ale', 'Belgian Dark Strong Ale', 'Belgian Pale Strong Ale', 'English Barleywine', 'English Strong Ale', 'Imperial Red Ale', 'Old Ale', 'Quadrupel (Quad)', 'Scotch Ale / Wee Heavy', 'Tripel', 'Wheatwine']),
    ('Wheat Beers', ['American Dark Wheat Beer', 'American Pale Wheat Beer', 'Dunkelweizen', 'Grodziskie', 'Hefeweizen', 'Kristallweizen', 'Witbier', 'American Dark Wheat Ale', 'Kristalweizen']),
    ('Wild/Sour Beers', ['Berliner Weisse', 'Berliner Weissbier', 'Brett Beer', 'Faro', 'Flanders Oud Bruin', 'Flanders Red Ale', 'Fruit Lambic', 'Fruited Kettle Sour', 'Gose', 'Gueuze', 'Lambic', 'Wild Ale', 'American Wild Ale', 'Lambic - Unblended']),
    ('Other', ['Fruit / Vegetable Beer', 'Low Alcohol Beer', 'Scottish Gruit / Ancient Herbed Ale', 'Lambic - Fruit', 'Herbed / Spiced Beer', 'Pumpkin Ale'])
])

df_beers['style_class'] = 'UNASSIGNED'

for style_keys in style_lookup.keys():
    df_beers['style_class'] = np.where(df_beers['style_ba'].isin(style_lookup[style_keys]), style_keys, df_beers['style_class'])


print(df_beers['style_class'].value_counts(sort=True))

# %% [markdown]
#  ### Breweries

# %%
#loading and renaming columns
df_brew = pd.read_csv(path_md + "breweries.csv", header = 1)
df_brew = df_brew.rename(columns={key: f"{key}_ba" for key in df_brew.columns[0:4]}) #rename columns to highlight dataset of origine
df_brew = df_brew.rename(columns={key: f"{key[:-2]}_rb" for key in df_brew.columns[4:8]})

#updating breweries list and beer count according to the beer we removed in the previous section
nb_beers_ba = df_beers.groupby("brewery_id_ba").size()
nb_beers_rb = df_beers.groupby("brewery_id_rb").size()

def map_element(element, df):
    """
    try to find an element in a dataframe, returns 0 if not found.
    """
    try:
        return df.loc[element]
    except:
        return 0

# create new columns for later use
df_brew["nbr_beers_ba"] = df_brew.apply(lambda x: map_element(x["id_ba"], nb_beers_ba), axis=1)
df_brew["nbr_beers_rb"] = df_brew.apply(lambda x: map_element(x["id_rb"], nb_beers_rb), axis=1)
df_brew = df_brew.loc[df_brew["nbr_beers_ba"] > 0] #subset
df_brew = df_brew.loc[df_brew["nbr_beers_rb"] > 0] #subset
print(f"number of beers in the beers dataframe {len(df_beers)}")
nb = df_brew["nbr_beers_ba"].sum() #find count
print(f"number of beers in the brewery dataframe {nb}")
#do number of beers per brewery matches between the two datasets ?
non_matching_brew = df_brew.loc[df_brew["nbr_beers_ba"] != df_brew["nbr_beers_rb"]] # find when not matching
non_matching_brew.head()

# %%
#In most cases yes. In BA, some breweries appear twice, either because there are duplicate (Brewery company, Brewery co.) or because some brewery have two distinct physical location (eg. Brewery North, Brewery South).
#This problem only occurs for tiny minority of beers and brewery. We then simply drop them and as well as the corresponding beers from those breweries.
df_brew = df_brew.loc[df_brew["nbr_beers_ba"] == df_brew["nbr_beers_rb"]]
df_beers = df_beers.loc[~df_beers["brewery_id_ba"].isin(non_matching_brew["id_ba"])] # subset to when in both


# %% [markdown]
#  ###### Location processing
#  countries sometimes have several names, the official name ("United States of America") and common names ("US, united states, the states, etc..."). In order to plot location on a map, we decide to replace all location names with ISO 3166-1 alpha-3 (3 letter country abbreviation). This is done using the ["search_fuzzy"](https://pypi.org/project/pycountry/) function in the pycountry library. The state column is only valid for breweries (and at a later stage users) within the united states, for which we have state-wide granularity

# %%
def location_processing(location):
    print(location)
    # removing html tags
    location = location.apply(lambda x : x.split('<', 1)[0])
    # separating country from states (for USA)
    country = location.apply(lambda x : x.split(',', 1)[0])
    states = location.apply(lambda x : str(x.split(',', 1)[1])[1:] if ',' in x else np.nan)

    #get the list of country and match it to an iso code utilizing the find_iso function
    country_list = country.unique()
    country_iso = {x: find_iso(x) for x in country_list}
    country_code = country.apply(lambda x: country_iso[x])

    return country, states, country_code


# %%
# location processing for breweries
country, states, country_code = location_processing(df_brew['location_ba'])
df_brew['country']=country
df_brew['states']=states
df_brew['country_code']=country_code
print("No country code assignment to the following countries: ", df_brew[df_brew["country_code"].isnull()]["country"].value_counts())


# %%
# assign a location by looking up the location of each beers' brewery

df_beers = df_beers.merge(df_brew[["country", "states", "country_code", "id_ba"]], left_on="brewery_id_ba", right_on="id_ba")
# country = df_b eers.apply(lambda beer: df_brew.loc[df_brew['name_ba'] == beer['brewery_name_ba']]['country'], axis = 1)
# print(country)


# %% [markdown]
#  ### Filtering and Transforming - Create dataframes for later use
# 
#  In this previous section, we created matched beer and breweries. However, we need access to users and ratings in order to conduct our analysis.
# 
#  For both datasets, we take the ratings and not the reviews. This is to maximise the amount of textual data we can obtain since reviews is a subset of reviews.
# 
#  To this end, we conduct an analysis to find all the users and ratings related to the subsetted beers for the previous section.
#  We do this in a number of steps
#  1. We preprocess by dropping rows where we find NaN values for both users and ratings. We add a location and a country column for the users.
#  2. We subset to consider only ratings on the matched beers.
#  3. We subset users to only the users who wrote the reviews we just found in step 2.
#  4. We save all of these new datasets to pickles.

# %% [markdown]
#  #### Beer Advocate Ratings
#  ##### Load BA User Data

# %%
df_ba_users = pd.read_csv(path_ba + "users.csv")
# We do not know 1 username, 2652 join dates and 31279 locations (i.e. they are nan)
# Since we are doing some geographical analysis, locations are mandatory and we drop users without location infos
df_ba_users = df_ba_users.dropna()


# %% [markdown]
#  ### location processing for ba users

# %%
country, states, country_code = location_processing(df_ba_users['location'])
df_ba_users['country'] = country
df_ba_users['states'] = states
df_ba_users['country_code'] = country_code
print("No country code assignment to the following countries: ", df_ba_users[df_ba_users["country_code"].isnull()]["country"].value_counts())
#there are only 5 regions which don't find a matching iso-code, which corresponds to 54 users.

# manually add New Zealand and Australia, and then remove the other outliers (not worth considering)
df_ba_users['country_code'] = np.where(df_ba_users['location']=="Aotearoa", "NZL", df_ba_users['country_code'])
df_ba_users['country_code'] = np.where(df_ba_users['location']=="Heard and McDonald Islands", "AUS", df_ba_users['country_code'])
df_ba_users = df_ba_users[~df_ba_users['country_code'].isna()]



# %% [markdown]
#  We the UK as one region as opposed to its sub-countries England, Wales, Scotland and Northern Ireland since there are not a lot users in the separate regions

# %%
print(df_ba_users[df_ba_users['country_code']=='GBR']['location'].value_counts())


# %%
df_ba_users.head()


# %%
df_ba_users.info()


# %% [markdown]
#  ##### Load BA Ratings
#  We consider only the ratings for Beer Advocate as the reviews are a subset of the ratings. We then filter out all NaN values in the ratings.

# %%
df_ba_ratings = pickle_load(path_ba + "ratings.txt",load_with_text=True)
df_ba_ratings.set_index(df_ba_ratings["date"], inplace = True) # Set date as index for ratings
df_ba_ratings = df_ba_ratings.convert_dtypes(infer_objects=True) # infer correct string types (does not filter nan values)
df_ba_ratings[["overall","taste","palate","aroma","appearance","abv"]] = df_ba_ratings[["overall","taste","palate","aroma","appearance","abv"]].apply(lambda x : pd.to_numeric(x,errors="coerce")) # coerce to numeric
df_ba_ratings = df_ba_ratings.astype({"review":bool}) # convert to boolean
df_ba_ratings["text"] = df_ba_ratings["text"].apply(lambda x: x if str.lower(x) != "nan" else "") # convert nan to empty string
df_ba_ratings = df_ba_ratings.dropna() # drop nan values
df_ba_ratings.info() # print info

# %%
df_ba_ratings.head()

# %%
print("Number of nan values",df_ba_ratings.isna().sum())
print("Number of non-reviews:",df_ba_ratings[df_ba_ratings["review"] == False].shape[0])
print("Number of text that is 'nan':",df_ba_ratings[df_ba_ratings["text"].str.lower() == "nan"].shape[0])
for column in ["beer_name","beer_id","brewery_name","brewery_name","style","user_name","user_id"]:
        print(f"number of values in {column} that is 'nan'", df_ba_ratings[df_ba_ratings[column].str.lower() == "nan"].shape[0])


# %% [markdown]
#  ##### Merge BA Ratings and Users for beers from the matched Dataset
#  Filter BA Ratings to only keep beers in matched beer dataset

# %%
df_md_beer_ids = df_beers["beer_id_ba"].unique().astype(str)
df_ba_beer_ids = df_ba_ratings["beer_id"].unique()
df_ba_ratings_filtered_md_beers = df_ba_ratings[df_ba_ratings["beer_id"].isin(df_md_beer_ids)]
df_ba_ratings_filtered_beers_merged_users = df_ba_ratings_filtered_md_beers.merge(df_ba_users, left_on="user_id", right_on="user_id")
df_ba_ratings_filtered_beers_merged_users.isnull().sum()

# %% [markdown]
#  ###### Add Style_class to BA ratings
df_ba_ratings_filtered_beers_merged_users['style_class'] = 'UNASSIGNED'

for style_keys in style_lookup.keys():
    df_ba_ratings_filtered_beers_merged_users['style_class'] = np.where(df_ba_ratings_filtered_beers_merged_users['style'].isin(style_lookup[style_keys]), style_keys, df_ba_ratings_filtered_beers_merged_users['style_class'])


print(df_ba_ratings_filtered_beers_merged_users['style_class'].value_counts(sort=True))


# %%
ba_pickle_filename = "df_ba_ratings_filtered_beers_merged_users.pickle"

# %%
df_ba_ratings_filtered_beers_merged_users.to_pickle(f"Data/{ba_pickle_filename}")

# %%
df_ba_ratings_filtered_beers_merged_users = pd.read_pickle(f"Data/{ba_pickle_filename}")
df_ba = df_ba_ratings_filtered_beers_merged_users
df_ba.pop("user_name_y")
df_ba.rename(columns={"user_name_x": "user_name", "nbr_ratings": "user_nbr_ratings", "location": "user_location", "country": "user_country", "states": "user_state", "country_code": "user_country_code"}, inplace=True)

# create a new column on df_ba which contains the country, state, country code and the style class of the beer reviewed.
df_ba["beer_id"] = df_ba["beer_id"].astype('int64')
df_ba = df_ba.merge(df_beers[["beer_id_ba", "country", "states", "country_code", "style_class", "avg_computed_ba"]], left_on='beer_id', right_on='beer_id_ba', how='left')
df_ba.pop("beer_id_ba")

df_ba.rename(columns={"country": "beer_country", "states": "beer_state", "country_code": "beer_country_code", "avg_computed_ba": "avg_beer_rating"}, inplace=True)
# check if there are nan values in df_ba["beer_country"]
df_ba[df_ba["beer_country"].isna()]
print(df_ba.columns)

# %% [markdown]
#  #### RateBeer Ratings
#  ##### Load RB User Data

# %%
df_rb_users = pd.read_csv(path_rb + 'users.csv')
df_rb_users = df_rb_users.convert_dtypes(infer_objects=True) 
df_rb_users.head()


# %%
df_rb_users.isna().sum()


# %%
# remove lines with nan
df_rb_users = df_rb_users.dropna()

country, states, country_code = location_processing(df_rb_users['location'])
df_rb_users['country'] = country
df_rb_users['states'] = states
df_rb_users['country_code'] = country_code
print("No country code assignment to the following countries: ", df_rb_users[df_rb_users["country_code"].isnull()]["country"].value_counts())
# We keep the most import regions but remove the others. We only consider US states so we drop the US Virgin Islands.
df_rb_users['country_code'] = np.where(df_rb_users['country']=="Virgin Islands (British)", "GBR", df_rb_users['country_code'])
df_rb_users['country_code'] = np.where(df_rb_users['country']=="South Korea", "ROK", df_rb_users['country_code'])
df_rb_users['country_code'] = np.where(df_rb_users['country']=="North Korea", "PRK", df_rb_users['country_code'])
df_rb_users = df_rb_users[~df_rb_users['country_code'].isna()]
# check that we got rid of missing country codes
print("No country code assignment to the following countries: ", df_rb_users[df_rb_users["country_code"].isnull()]["country"].value_counts())

df_rb_users = df_rb_users.dropna()
df_rb_users.isna().sum()

# %%
df_rb_users.info()



# %% [markdown]
#  ##### Load RB Ratings
#  For RB Ratings and Reviews are equivalent

# %%
df_rb_reviews = pickle_load(path_rb + "reviews.txt",load_with_text=True)
df_rb_reviews.set_index(df_rb_reviews["date"], inplace = True) # Set date as index for ratings
df_rb_reviews = df_rb_reviews.convert_dtypes(infer_objects=True) # infer correct string types (does not filter nan values)
df_rb_reviews[["overall","taste","palate","aroma","appearance","abv","user_id"]] = df_rb_reviews[["overall",
"taste","palate","aroma","appearance","abv","user_id"]].apply(lambda x : pd.to_numeric(x,errors="coerce")) # coerce to numeric
df_rb_reviews["text"] = df_rb_reviews["text"].apply(lambda x: x if str.lower(x) != "nan" else "") # convert nan to empty string
df_rb_reviews = df_rb_reviews.dropna() # drop nan values
df_rb_reviews.info() # print info

# %%
print("Number of nan values",df_rb_reviews.isna().sum())
print("Number of text that is 'nan':",df_rb_reviews[df_rb_reviews["text"].str.lower() == "nan"].shape[0])
for column in ["beer_name","beer_id","brewery_name","brewery_name","style","user_name"]:
        print(f"number of values in {column} that is 'nan'", df_rb_reviews[df_rb_reviews[column].str.lower() == "nan"].shape[0])

# %% [markdown]
#  ##### Merge BA Ratings and Users for beers from the matched Dataset
#  Filter RB Ratings to only keep beers in matched beer dataset

# %%
df_md_beer_ids = df_beers["beer_id_rb"].unique().astype(str)
df_rb_beer_ids = df_rb_reviews["beer_id"].unique()
df_rb_reviews_filtered_md_beers = df_rb_reviews[df_rb_reviews["beer_id"].isin(df_md_beer_ids)]
df_rb_reviews_filtered_beers_merged_users = df_rb_reviews_filtered_md_beers.merge(df_rb_users, left_on="user_id", right_on="user_id")
df_rb_reviews_filtered_beers_merged_users.isnull().sum()

# %% [markdown]
#  ###### Add Style_class to RB ratings
# %%
df_rb_reviews_filtered_beers_merged_users['style_class'] = 'UNASSIGNED'

for style_keys in style_lookup.keys():
    df_rb_reviews_filtered_beers_merged_users['style_class'] = np.where(df_rb_reviews_filtered_beers_merged_users['style'].isin(style_lookup[style_keys]), style_keys, df_rb_reviews_filtered_beers_merged_users['style_class'])


print(df_rb_reviews_filtered_beers_merged_users['style_class'].value_counts(sort=True))
# TODO: Problem! 307716 unassigned!!! What if we merge on beer_id?


# %%
rb_pickle_filename = "df_rb_reviews_filtered_beers_merged_users.pickle"

# %%
df_rb_reviews_filtered_beers_merged_users.to_pickle(f"Data/{rb_pickle_filename}")
# %%
df_rb_reviews_filtered_beers_merged_users = pd.read_pickle(f"Data/{rb_pickle_filename}")
df_rb = df_rb_reviews_filtered_beers_merged_users
# TODO: move this preprocessing up the pipeline
# some column cleaning
df_rb.pop("user_name_y")
df_rb.rename(columns={"user_name_x": "user_name", "nbr_ratings": "user_nbr_ratings", "location": "user_location", "country": "user_country", "states": "user_state", "country_code": "user_country_code"}, inplace=True)
df_rb_reviews_filtered_beers_merged_users.head()

# create a new column on df_rb which contains the country, state, country code and the style class of the beer reviewed.
df_rb["beer_id"] = df_rb["beer_id"].astype('int64')
df_rb = df_rb.merge(df_beers[["beer_id_rb", "country", "states", "country_code", "style_class", "avg_computed_rb"]], left_on='beer_id', right_on='beer_id_rb', how='left')
df_rb.pop("beer_id_rb")

df_rb.rename(columns={"country": "beer_country", "states": "beer_state", "country_code": "beer_country_code", "avg_computed_rb": "avg_beer_rating"}, inplace=True)
# check if there are nan values in df_rb["beer_country"]
df_rb[df_rb["beer_country"].isna()]
print(df_rb.columns)
#TODO: do the same for ba
# %%
df_rb["user_country"].value_counts()
# %% [markdown]
#  #### Print all columns

# %%
df_rb.columns

# %%
df_ba.columns

# %%
df_beers.columns

# %%
df_brew.columns


# %% [markdown]
#  # Data exploration
# 
#  GEOGRAPHICAL DISTRIBUTION
#  - #beers per country
# 
#  TEXT LENGTHS
#  - #review lengths
# 
#  DISTRIBUTIONS
#  - Average rating per beer
#  - Average rating for each of the characteristics per beer
# 
#  VISUALISE MERGED STYLES (DATA ENRICHMENT)

# %%
# TODO: warning when comparing ratings between RB and BA: Appearance, Aroma, Palate, Taste and overall have different ranges in both datasets (rating to be double checked)
# for the average rating comparison:
plt.hist(df_beers["avg_computed_ba"], bins=50, alpha=0.5, label='BeerAdvocate rating', density = True)
plt.hist(df_beers["avg_computed_rb"], bins=50, alpha=0.5, label='RateBeer rating', density = True)
plt.title("histogram of ratings from both dataset within the merge dataset")
plt.legend(loc='upper right')
plt.xlabel("rating")
plt.ylabel("density")
plt.show()
#We notice that rate beer is more critical on average that beer advocate. Caution should be taken if two datasets


# %%
for trait in ["aroma", "palate", "taste", "appearance", "overall"]:
    plt.hist(df_ba[trait], bins=10, alpha=0.5, label=f'BeerAdvocate {trait}', density = True)
    plt.hist(df_rb[trait], bins=10, alpha=0.5, label=f'RateBeer {trait}', density = True)
    plt.title(f"histogram of {trait} from both dataset")
    plt.legend(loc='upper right')
    plt.xlabel(f"{trait}")
    plt.ylabel("density")
    plt.show()


# %% [markdown]
#  We see from the previous cell that the scales are very different for the BA and RB, since they use very different schemes to quantify beer quality.
#  Apply minmax scaling so the scales are comparable.
#  We see that RB and BA has very different distributions since for RB the data can only take integer values whilst for BA it can be a float.

# %%
for trait in ["aroma", "palate", "taste", "appearance", "overall"]:
    plt.hist((df_ba[trait]-df_ba[trait].min())/(df_ba[trait].max()-df_ba[trait].min()), bins=15, alpha=0.5, label=f'BeerAdvocate {trait}', density = True)
    plt.hist((df_rb[trait]-df_rb[trait].min())/(df_rb[trait].max()-df_rb[trait].min()), bins=15, alpha=0.5, label=f'RateBeer {trait}', density = True)
    plt.title(f"histogram of {trait} from both datasets")
    plt.legend(loc='upper right')
    plt.xlabel(f"{trait}")
    plt.ylabel("density")
    plt.show()


# %% [markdown]
#  We see that the review distibution is heavy tailed, showing that there exist some reviews that are extremely long in comparison to the average reviews.
#  This occurs both for the number of words and the number of characters.

# %%
plt.hist(df_ba['text'].str.split(' ').str.len(), bins=10, alpha=0.5, label=f'BeerAdvocate {trait}', density = True)
plt.hist(df_rb['text'].str.split(' ').str.len(), bins=10, alpha=0.5, label=f'RateBeer {trait}', density = True)
plt.title(f"histogram of number of words per review from both datasets")
plt.legend(loc='upper right')
plt.xlabel("Number of words")
plt.ylabel("density")
plt.show()

plt.hist(df_ba['text'].str.len(), bins=10, alpha=0.5, label=f'BeerAdvocate {trait}', density = True)
plt.hist(df_rb['text'].str.len(), bins=10, alpha=0.5, label=f'RateBeer {trait}', density = True)
plt.title(f"histogram of number of characters per review from both datasets")
plt.legend(loc='upper right')
plt.xlabel("Number of characters")
plt.ylabel("density")
plt.show()


# %% [markdown]
#  ### map visualization:
# load world maps (from [open data soft](https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/export/)) and us maps from [geopandas default library](https://geopandas.org/en/stable/docs/user_guide/io.html).

# %%
url_world_map = "https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/download/?format=shp&timezone=Europe/Berlin&lang=fr"
world = gpd.read_file(url_world_map)
usa = gpd.read_file(gplt.datasets.get_path("contiguous_usa")) #loading usa map from geopandas default library

# need to create per country stats:

# %%
# beer count per country
beer_country = df_beers["country_code"].groupby(df_beers["country_code"]).count()
display(beer_country)
# beer count per state
beer_usa = df_beers.dropna() #countries outside of us have NaN values for the state
beer_states = beer_usa["states"].groupby(beer_usa["states"]).count()

# plotting log number of beers per country
beer_country = world.merge(beer_country, how = "left", left_on="iso3", right_index = True, )
beer_country['log_beers'] = beer_country['country_code'].apply(lambda x: np.log10(x) if x >= 1 else 0)
#number of breweries per country (log)
#scheme = mc.FisherJenks(beer_country['log_beers'], k=8)
gplt.choropleth(beer_country, hue="log_beers", legend=True)
plt.title("nb of beer per country (log10)")
plt.show()

# plotting log number of beers per states
beer_states = usa.merge(beer_states, how = "left", left_on="state", right_index = True)
beer_states['log_beers'] = beer_states['states'].apply(lambda x: np.log10(x) if x >= 1 else 0)

gplt.choropleth(beer_states, hue="log_beers", legend=True)
plt.title("nb of beer per states (log10)")
plt.show()


# %% [markdown]
# We can see that most of our data is concentrated in the USA and the North Americas in general. Thus, we make an extra effort to also include US states.

# %% [markdown]
#  ## Data enrichment
#  ### style clustering
#  We want to create better style groupings in order to understand our data better.




# %%
counts = df_beers["style_class"].value_counts().sort_values(ascending=False)
names = counts.index
plt.bar(names, counts, alpha=0.5)
# plt.hist(df_ba_beers["style_class"], )
plt.title(f"Amount of beers per style for BA dataset")
plt.legend(loc='upper right')
plt.xlabel("Beer Style")
plt.ylabel("Number of beers")
plt.xticks(rotation=90)
plt.show()


# %% [markdown]
# The predominant beer styles are Pale Ales and Indian Pale Ales. In total we have 15 classes of beer styles.

# %% [markdown]
#  # Plan for Investigation/Methods:
#  ### RQ1: Are beer preferences influenced by geography/culture?
#  We investigate this by figuring out preferred beer styles.
#  We can conduct t-tests to see if there is a significant difference in ratings per beer style between countries, using the Sidak correction.
#  ### RQ2: Do different cultures prioritise/prefer different aspects of beers such as feel? Are some cultures more critical of beer?
#  We can conduct the exact same analysis as in RQ1 for each of the beer characteristics (palate, etc.) and also for beer ratings in general.
#  ### RQ3: Do different cultures have stylistically different ways of writing reviews and discussing beer? Do users talk about foreign beers differently than they talk about their local ones?
#  Reviews for each country will just be aggregated into a large piece of text since we strictly interested in comparing per country. We focus on English speaking countries.
#  To do textual analysis, we will first conduct textual preprocessing steps. This will involving removing punctuation, removing stopwords, capitalisation and most importantly choosing indexing terms. Depending on results, we may find that other steps should also be taken.
#  After this preprocessing, we will vectorise the corpus. To do this, we can use a count vectoriser (bag of words model), tf-idf or other methods. Finally, we will compute distances between textual reviews by choosing either existing research methods e.g. [Ruzicka](https://github.com/mikekestemont/ruzicka), [PyStyl](https://github.com/mikekestemont/pystyl), or by using cosine distance metric.
# 
#  Depending on the results of this, we can then investigate if the distances between the corpuses can represent the cultural similarity between countries.
#  If so, it may be interesting to run a dendrogram clustering using [sklearn](https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html) to see if we can recreate geographical regions.
#  It may be that we can recreate a North American group, Oceanic group and European group. This will obviously depend on our results.
# 
#  Furthermore, we will also create a wordmap of the language used in reviews. Either we can use simple frequency analysis, or again we can leverage existing libraries such as [word-cloud](https://github.com/amueller/word_cloud).
# 
#  We can also rerun our distance analysis for each country between reviews for local and foreign beer to see if there is a difference. We can then see what countries exhibit this difference the most.
# 
# ### RQ4: Is there a "home bias" for reviewers?
# I.e. do users rate local beers higher than their foreign counterparts?
# The hypothesis is the following:
# $H_0$: $\mu_l = \mu_f$
# $H_a$: $\mu_l \neq \mu_f$
# 
# where $\mu_l$ ($\mu_f$) is the average user ratings given to local (foreign) beers.
# 
# We have to control the effect of confounders. Here the "treatment" might be identified as "do the users and the beers come from the same country". 
# We call an experiment treated (treatment=1), if the user and the beer come from the same country. On the otherhand, if the user and the beer do not come from the same country, we set treatment = 0.
# However, there are some covariates (such as users' taste, whether they are overall more critical or not, etc...). These might influence both the outcome (the users' rating of that beer) and the likelihood to be from the same place as the beer. To mitigate this effect, we match users based on propensity. A propensity score measures the probability of a user to rate beer from his own country/state (treatment = 1) vs. a foreign beer (treatment = 0) given observed covariates. Is is learned using logistic regression with labels being 1 if the reviewed beer is local, 0 if it is foreign. Some features considered are for example users' average ratings, number of ratings, users' "taste" (ratings per style), country, etc...
# 

# %% [markdown]
#  # Plan for Communication/Visualisation:
#  We plan to tell the following story (_obviously result dependent_):
#  1. [HeatMap] Show that there are differences in ratings and beer preferences using our initial analyses.
#  2. [HeatMap, WordCloud] Show that there are differences in how different countries talk about beers.
#  3. [Dendrogram] Try and relate this to cultural or geographical proximities of the countries.
#  3. [HeatMap] See if there is a difference for each country in how they talk about local and foreign beers.
#  4. [HeatMap] Show the results for our detailed analysis into home bias to determine if users have a preference for local produce or not.
#  5. Give our main takeaways.


# %% [markdown]
# RQ4: home bias
# %%
df = df_ba
# here we can subset if needed
#df = df[df['user_country'] != 'United States']
#df = df[df['beer_country'] != 'United States']
# %%
# definition of the treatment variable
df["treatment"] = df.apply(lambda row: 1 if row["user_country"] == row["beer_country"] else 0, axis=1)
# %% [markdown]
# ### data exploration
# TODO: put this into the data exploration
# %%
# feature number of reviews for each beer
df_groupby_beer = df.groupby(by = "beer_id").agg({"beer_id": "count", "treatment": "sum"})
df["nb_reviews_per_beer"] = df.apply(lambda row: df_groupby_beer.loc[row["beer_id"]]["beer_id"], axis=1)
df["nb_reviews_per_beer_local"] = df.apply(lambda row: df_groupby_beer.loc[row["beer_id"]]["treatment"], axis=1)
df["nb_reviews_per_beer_foreign"] = df["nb_reviews_per_beer"] - df["nb_reviews_per_beer_local"]
df["share_local_reviews"] = df["nb_reviews_per_beer_local"] / df["nb_reviews_per_beer"]
# %%
# plot the proportion of local reviews for each beer
df["share_local_reviews"].hist(bins=50, alpha=0.5, label="local reviews")

# %% [markdown]
# ### analysis prior to matching
# %%
# plot the rating distribution of the treatment and control groups prior to matching
df[df["treatment"] == 1]["rating"].hist(bins=20, alpha=0.5, label="local reviews")
df[df["treatment"] == 0]["rating"].hist(bins=20, alpha=0.5, label="foreign reviews")
plt.legend()
plt.title("distribution of local vs. foreign reviews")
# %%
# run a t-test to see if there is a significant difference in ratings between reviews with treatment = 1 and reviews with treatment = 0 (prior to matching)
from scipy.stats import ttest_ind
res = ttest_ind(df[df["treatment"] == 1]["rating"], df[df["treatment"] == 0]["rating"], equal_var=False)
print(res)
# print the average difference of mean ratings between treatment = 1 and treatment = 0
print("average difference of mean ratings between treatment = 1 and treatment = 0: ", df[df["treatment"] == 1]["rating"].mean() - df[df["treatment"] == 0]["rating"].mean())
# %% [markdown]
# ## propensity score computation using random forest
# %%
# feature mean user rating
df_users = df.groupby(by = "user_id").agg({"rating": "mean"})
df["avg_user_rating"] = df.apply(lambda row: df_users.loc[row["user_id"]]["rating"], axis=1)

# feature beer country and user country as categorical integers
df["beer_country_cat"] = df["beer_country"].astype("category").cat.codes
df["user_country_cat"] = df["user_country"].astype("category").cat.codes
df["style_class_cat"] = df["style_class"].astype("category").cat.codes
# %%
# create a feature vector using the following features:
# - avg user rating
# - number of reviews
# - user country
# - beer country
# - beer style class
# - beer average rating

# TODO: complete this for the notebook
X = df.values
# create label vector (treatment column)
y = df["treatment"].values

# %%
# split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train a random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, y_train)

# get the probabilities on the test set
y_pred = clf.predict_proba(X_test)[:,1]
# predict the treatment with a threshold of 0.5
y_pred_treated = clf.predict(X_test)

# measure f1 score
from sklearn.metrics import f1_score
print("f1 score: ", f1_score(y_test, y_pred_treated))
# print confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred_treated))

# %%
# add propensity score to dataframe
y_pred = clf.predict_proba(X)[:,1]
df["propensity_score"] = y_pred
# %%
# plot the distribution of propensity score for treatment = 1 and treatment = 0
df[df["treatment"] == 1]["propensity_score"].hist(bins=20, alpha=0.5, label="local reviews")
df[df["treatment"] == 0]["propensity_score"].hist(bins=20, alpha=0.5, label="foreign reviews")
plt.legend()
plt.title("distribution of propensity score for local vs. foreign reviews")
# %%
# perform matching between treated and control group using propensity score

# %% [markdown]
# ### matrix factorization
# %%
# reset beer id and user id to be from 0 to nb_beer and nb_user
df_users = df.groupby("user_id").agg({"treatment": "mean"}).reset_index()
df_groupby_beer = df.groupby("beer_id").agg({"treatment": "mean"}).reset_index()
# map the user_id in df to the index in df_users
df["user_id"] = df["user_id"].apply(lambda x: df_users[df_users["user_id"] == x].index[0]).astype(int)
df["beer_id"] = df["beer_id"].apply(lambda x: df_groupby_beer[df_groupby_beer["beer_id"] == x].index[0]).astype(int)
# %%
# Matrix factorization with biases
import surprise.prediction_algorithms.matrix_factorization as mf
from surprise.model_selection import cross_validate
from surprise import Reader, Dataset

algo = mf.SVD(n_factors=100, n_epochs=20, biased=True, lr_all=0.005, reg_all=0.02, verbose=True)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user_id", "beer_id", "rating"]].rename(columns={"user_id": "userID", "beer_id": "itemID", "rating": "rating"}), reader)
print(df.shape)
user_bias = algo.fit(data.build_full_trainset()).bu
beer_bias = algo.fit(data.build_full_trainset()).bi
plt.hist(user_bias)
plt.hist(beer_bias)
# %%
# add feature user bias and beer bias
df["user_bias"] = df.apply(lambda row: user_bias[row["user_id"]], axis=1)
df["beer_bias"] = df.apply(lambda row: beer_bias[row["beer_id"]], axis=1)
# %%
X = df[["user_bias", "beer_bias"]].values
# create label vector (treatment column)
y = df["treatment"].values

# %%
df["idx"] = df.reset_index().index
# %%
# %%
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *


psm = PsmPy(df[["user_bias", "beer_bias", "treatment", "idx"]], treatment='treatment', indx='idx', exclude = [])
psm.logistic_ps(balance = True)