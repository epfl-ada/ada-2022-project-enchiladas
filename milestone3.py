# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pycountry # match country names to iso code
import geoplot as gplt #plotting maps
import geopandas as gpd
import geoplot.crs as gcrs
import matplotlib.pyplot as plt
import mapclassify as mc #???
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
# %%
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
#  Do the names match ? Most of them don't. BA doesn't include the brewery name in the beer name but RB often does. We will keep the RB names since they are more informative.

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
# %%
df_beers.info()


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
df_ba_ratings_filtered_md_beers = df_ba_ratings[df_ba_ratings["beer_id"].isin(df_md_beer_ids)] # Only keep ratings for beers in the matched dataset
df_ba_ratings_filtered_beers_merged_users = df_ba_ratings_filtered_md_beers.merge(df_ba_users, left_on="user_id", right_on="user_id") # Merge with users
df_ba_ratings_filtered_beers_merged_users.isnull().sum()


# %%
df_ba_ratings_filtered_beers_merged_users.rename(columns={"user_name_x": "user_name", "nbr_ratings": "user_nbr_ratings", "location": "user_location", "country": "user_country", "states": "user_state", "country_code": "user_country_code"}, inplace=True)
# create a new column on df_ba which contains the country, state, country code and the style class of the beer reviewed.
df_ba_ratings_filtered_beers_merged_users["beer_id"] = df_ba_ratings_filtered_beers_merged_users["beer_id"].astype('int64')
df_ba_ratings_filtered_beers_merged_users = df_ba_ratings_filtered_beers_merged_users.merge(df_beers[["beer_id_ba", "country", "states", "country_code", "style_class", "avg_computed_ba"]], left_on='beer_id', right_on='beer_id_ba', how='left')
df_ba_ratings_filtered_beers_merged_users.pop("beer_id_ba")

df_ba_ratings_filtered_beers_merged_users.rename(columns={"country": "beer_country", "states": "beer_state", "country_code": "beer_country_code", "avg_computed_ba": "avg_beer_rating"}, inplace=True)
# check if there are nan values in df_ba_ratings_filtered_beers_merged_users["beer_country"]
df_ba_ratings_filtered_beers_merged_users[df_ba_ratings_filtered_beers_merged_users["beer_country"].isna()]
print(df_ba_ratings_filtered_beers_merged_users.columns)

# %% [markdown]
#  # Rescale Ratings
# Some users rate on average more positively or more negativly with respect to others. Some of this assumed effect will occur, because users rate different beers which indeed are better or worse on average. We hereby assume, that some users are just more positive or negative in general, even if they rate the same beers.
# To counteract this effect, we rescale the ratings to a -1,1 scale where 0 is the users average rating. This way, we can compare users on an equal footing.

# TODO: Formalize what we do. Ideally add citations.
# %%
def scale(rating, user_average, top=5, bottom=1):
    """
    Returns the scale of the rating with respect to the users average rating.
    """
    if rating > user_average:
        return top-user_average
    elif rating < user_average:
        return user_average-bottom
    else:
        return bottom
def rescale(rating, user_average, top = 5, bottom = 1):
    """
    Rescales the rating to a -1,1 scale where 0 is the users average rating.
    """
    return (rating - user_average) / scale(rating, user_average, top, bottom)

# %% [markdown]
#  ## BA
# Columns to be rescaled in BA: 
# - aroma (1-5)
# - appearance (1-5)
# - taste (1-5)
# - palate (1-5)
# - overall (1-5)
# - rating (1-5)

# TODO: here might be a good place to explain the different usages of the individual ratings/aspects across the two datasets

# %%
# Sanity check the ranges of the ratings and aspects
print("aroma", df_ba_ratings_filtered_beers_merged_users["aroma"].min(), df_ba_ratings_filtered_beers_merged_users["aroma"].max())
print("appearance", df_ba_ratings_filtered_beers_merged_users["appearance"].min(), df_ba_ratings_filtered_beers_merged_users["appearance"].max())
print("taste", df_ba_ratings_filtered_beers_merged_users["taste"].min(), df_ba_ratings_filtered_beers_merged_users["taste"].max())
print("palate", df_ba_ratings_filtered_beers_merged_users["palate"].min(), df_ba_ratings_filtered_beers_merged_users["palate"].max())
print("overall", df_ba_ratings_filtered_beers_merged_users["overall"].min(), df_ba_ratings_filtered_beers_merged_users["overall"].max())
print("rating", df_ba_ratings_filtered_beers_merged_users["rating"].min(), df_ba_ratings_filtered_beers_merged_users["rating"].max())

# %%
# Compute the average rating for each user
df_ba_ratings_filtered_beers_merged_users["user_average_aroma"] = df_ba_ratings_filtered_beers_merged_users.groupby("user_id")["aroma"].transform("mean")
df_ba_ratings_filtered_beers_merged_users["user_average_appearance"] = df_ba_ratings_filtered_beers_merged_users.groupby("user_id")["appearance"].transform("mean")
df_ba_ratings_filtered_beers_merged_users["user_average_taste"] = df_ba_ratings_filtered_beers_merged_users.groupby("user_id")["taste"].transform("mean")
df_ba_ratings_filtered_beers_merged_users["user_average_palate"] = df_ba_ratings_filtered_beers_merged_users.groupby("user_id")["palate"].transform("mean")
df_ba_ratings_filtered_beers_merged_users["user_average_overall"] = df_ba_ratings_filtered_beers_merged_users.groupby("user_id")["overall"].transform("mean")
df_ba_ratings_filtered_beers_merged_users["user_average_rating"] = df_ba_ratings_filtered_beers_merged_users.groupby("user_id")["rating"].transform("mean")

# %%
# Rescale the ratings and aspects to a -1,1 scale where 0 is the users average rating

df_ba_ratings_filtered_beers_merged_users["aroma_rescaled"] = df_ba_ratings_filtered_beers_merged_users.apply(lambda row: rescale(row["aroma"], row["user_average_aroma"]), axis=1)
df_ba_ratings_filtered_beers_merged_users["appearance_rescaled"] = df_ba_ratings_filtered_beers_merged_users.apply(lambda row: rescale(row["appearance"], row["user_average_appearance"]), axis=1)
df_ba_ratings_filtered_beers_merged_users["taste_rescaled"] = df_ba_ratings_filtered_beers_merged_users.apply(lambda row: rescale(row["taste"], row["user_average_taste"]), axis=1)
df_ba_ratings_filtered_beers_merged_users["palate_rescaled"] = df_ba_ratings_filtered_beers_merged_users.apply(lambda row: rescale(row["palate"], row["user_average_palate"]), axis=1)
df_ba_ratings_filtered_beers_merged_users["overall_rescaled"] = df_ba_ratings_filtered_beers_merged_users.apply(lambda row: rescale(row["overall"], row["user_average_overall"]), axis=1)
df_ba_ratings_filtered_beers_merged_users["rating_rescaled"] = df_ba_ratings_filtered_beers_merged_users.apply(lambda row: rescale(row["rating"], row["user_average_rating"]), axis=1)

# %%
# Sanity check the ranges of the rescaled ratings and aspects
print("aroma_rescaled", df_ba_ratings_filtered_beers_merged_users["aroma_rescaled"].min(), df_ba_ratings_filtered_beers_merged_users["aroma_rescaled"].max())
print("appearance_rescaled", df_ba_ratings_filtered_beers_merged_users["appearance_rescaled"].min(), df_ba_ratings_filtered_beers_merged_users["appearance_rescaled"].max())
print("taste_rescaled", df_ba_ratings_filtered_beers_merged_users["taste_rescaled"].min(), df_ba_ratings_filtered_beers_merged_users["taste_rescaled"].max())
print("palate_rescaled", df_ba_ratings_filtered_beers_merged_users["palate_rescaled"].min(), df_ba_ratings_filtered_beers_merged_users["palate_rescaled"].max())
print("overall_rescaled", df_ba_ratings_filtered_beers_merged_users["overall_rescaled"].min(), df_ba_ratings_filtered_beers_merged_users["overall_rescaled"].max())
print("rating_rescaled", df_ba_ratings_filtered_beers_merged_users["rating_rescaled"].min(), df_ba_ratings_filtered_beers_merged_users["rating_rescaled"].max())



# %%
ba_pickle_filename = "df_ba_ratings_filtered_beers_merged_users.pickle"


# %%
df_ba_ratings_filtered_beers_merged_users.to_pickle(f"Data/{ba_pickle_filename}")

# %%
df_ba = pd.read_pickle(f"Data/{ba_pickle_filename}")

# %%
df_ba.info()

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


# %% 
# some column cleaning
df_rb_reviews_filtered_beers_merged_users.pop("user_name_y")
df_rb_reviews_filtered_beers_merged_users.rename(columns={"user_name_x": "user_name", "nbr_ratings": "user_nbr_ratings", "location": "user_location", "country": "user_country", "states": "user_state", "country_code": "user_country_code"}, inplace=True)
df_rb_reviews_filtered_beers_merged_users.head()

# create a new column on df_rb which contains the country, state, country code and the style class of the beer reviewed.
df_rb_reviews_filtered_beers_merged_users["beer_id"] = df_rb_reviews_filtered_beers_merged_users["beer_id"].astype('int64')
df_rb_reviews_filtered_beers_merged_users = df_rb_reviews_filtered_beers_merged_users.merge(df_beers[["beer_id_rb", "country", "states", "country_code", "style_class", "avg_computed_rb"]], left_on='beer_id', right_on='beer_id_rb', how='left')
df_rb_reviews_filtered_beers_merged_users.pop("beer_id_rb")

df_rb_reviews_filtered_beers_merged_users.rename(columns={"country": "beer_country", "states": "beer_state", "country_code": "beer_country_code", "avg_computed_rb": "avg_beer_rating"}, inplace=True)
# check if there are nan values in df_rb_reviews_filtered_beers_merged_users["beer_country"]
df_rb_reviews_filtered_beers_merged_users[df_rb_reviews_filtered_beers_merged_users["beer_country"].isna()]
print(df_rb_reviews_filtered_beers_merged_users.columns)



# %% [markdown]
#  ## RB Aspects
# Columns to be rescaled in RB:
# - aroma (1-10)
# - appearance (1-5)
# - taste (1-10)
# - palate (1-5)
# - overall (1-20)
# - rating (0-5)

# %%
# Add 1 to every rating to make it 1-6 instead of 0-5 to prevent divison by zero
df_rb_reviews_filtered_beers_merged_users["translated_rating"] = df_rb_reviews_filtered_beers_merged_users["rating"] + 1

# %%
# Sanity check the ranges of the ratings and aspects
print("aroma", df_rb_reviews_filtered_beers_merged_users["aroma"].min(), df_rb_reviews_filtered_beers_merged_users["aroma"].max())
print("appearance", df_rb_reviews_filtered_beers_merged_users["appearance"].min(), df_rb_reviews_filtered_beers_merged_users["appearance"].max())
print("taste", df_rb_reviews_filtered_beers_merged_users["taste"].min(), df_rb_reviews_filtered_beers_merged_users["taste"].max())
print("palate", df_rb_reviews_filtered_beers_merged_users["palate"].min(), df_rb_reviews_filtered_beers_merged_users["palate"].max())
print("overall", df_rb_reviews_filtered_beers_merged_users["overall"].min(), df_rb_reviews_filtered_beers_merged_users["overall"].max())
print("translated_rating", df_rb_reviews_filtered_beers_merged_users["translated_rating"].min(), df_rb_reviews_filtered_beers_merged_users["translated_rating"].max())

# %%
# Compute the average rating for each user
df_rb_reviews_filtered_beers_merged_users["user_average_aroma"] = df_rb_reviews_filtered_beers_merged_users.groupby("user_id")["aroma"].transform("mean")
df_rb_reviews_filtered_beers_merged_users["user_average_appearance"] = df_rb_reviews_filtered_beers_merged_users.groupby("user_id")["appearance"].transform("mean")
df_rb_reviews_filtered_beers_merged_users["user_average_taste"] = df_rb_reviews_filtered_beers_merged_users.groupby("user_id")["taste"].transform("mean")
df_rb_reviews_filtered_beers_merged_users["user_average_palate"] = df_rb_reviews_filtered_beers_merged_users.groupby("user_id")["palate"].transform("mean")
df_rb_reviews_filtered_beers_merged_users["user_average_overall"] = df_rb_reviews_filtered_beers_merged_users.groupby("user_id")["overall"].transform("mean")
df_rb_reviews_filtered_beers_merged_users["user_average_rating"] = df_rb_reviews_filtered_beers_merged_users.groupby("user_id")["translated_rating"].transform("mean")

# %%
# Rescale the ratings and aspects to a -1,1 scale where 0 is the users average rating

df_rb_reviews_filtered_beers_merged_users["aroma_rescaled"] = df_rb_reviews_filtered_beers_merged_users.apply(lambda row: rescale(row["aroma"], row["user_average_aroma"],top=10), axis=1)
df_rb_reviews_filtered_beers_merged_users["appearance_rescaled"] = df_rb_reviews_filtered_beers_merged_users.apply(lambda row: rescale(row["appearance"], row["user_average_appearance"]), axis=1)
df_rb_reviews_filtered_beers_merged_users["taste_rescaled"] = df_rb_reviews_filtered_beers_merged_users.apply(lambda row: rescale(row["taste"], row["user_average_taste"],top=10), axis=1)
df_rb_reviews_filtered_beers_merged_users["palate_rescaled"] = df_rb_reviews_filtered_beers_merged_users.apply(lambda row: rescale(row["palate"], row["user_average_palate"]), axis=1)
df_rb_reviews_filtered_beers_merged_users["overall_rescaled"] = df_rb_reviews_filtered_beers_merged_users.apply(lambda row: rescale(row["overall"], row["user_average_overall"],20), axis=1)
df_rb_reviews_filtered_beers_merged_users["rating_rescaled"] = df_rb_reviews_filtered_beers_merged_users.apply(lambda row: rescale(row["translated_rating"], row["user_average_rating"],top=6), axis=1)

# %%
# Sanity check the ranges of the rescaled ratings and aspects
print("aroma_rescaled", df_rb_reviews_filtered_beers_merged_users["aroma_rescaled"].min(), df_rb_reviews_filtered_beers_merged_users["aroma_rescaled"].max())
print("appearance_rescaled", df_rb_reviews_filtered_beers_merged_users["appearance_rescaled"].min(), df_rb_reviews_filtered_beers_merged_users["appearance_rescaled"].max())
print("taste_rescaled", df_rb_reviews_filtered_beers_merged_users["taste_rescaled"].min(), df_rb_reviews_filtered_beers_merged_users["taste_rescaled"].max())
print("palate_rescaled", df_rb_reviews_filtered_beers_merged_users["palate_rescaled"].min(), df_rb_reviews_filtered_beers_merged_users["palate_rescaled"].max())
print("overall_rescaled", df_rb_reviews_filtered_beers_merged_users["overall_rescaled"].min(), df_rb_reviews_filtered_beers_merged_users["overall_rescaled"].max())
print("rating_rescaled", df_rb_reviews_filtered_beers_merged_users["rating_rescaled"].min(), df_rb_reviews_filtered_beers_merged_users["rating_rescaled"].max())

# %% [markdown]
# It is normal that the min of the rescaled ratings is -0.82, the worst score in the rb df is not 0, but 0.5. I.e. nobody rated a beer with 0 stars.



# %%
rb_pickle_filename = "df_rb_reviews_filtered_beers_merged_users.pickle"

# %%
df_rb_reviews_filtered_beers_merged_users.to_pickle(f"Data/{rb_pickle_filename}")
# %%
df_rb = pd.read_pickle(f"Data/{rb_pickle_filename}")

# %%
df_rb.info()

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




# %%
