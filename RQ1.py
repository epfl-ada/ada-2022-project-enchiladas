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
import scipy.stats as stats
from helpers import * #custom made functions
import plotly.express as px
print("import completed")

# number of pandas rows to display
pd.set_option('display.max_rows', 100)

# %% [markdown]
#  # RQ1 & RQ 2
# Are beer style preferences influenced by geography/culture?
# Investigate RQ1 by viewing ratings for different beer styles for each country. Specifically, we want conduct t-tests investigating if there are differences in this between countries, using the Sidak correction.

# For each country / for each state:
# compute the CI Sidak correction
# For Each Beerstyle: 
# - Perform a t-test comparing means of each country

# %%
#  ## Load BA data
ba_pickle_filename = "df_ba_ratings_filtered_beers_merged_users.pickle"
df_ba = pd.read_pickle(f"Data/{ba_pickle_filename}")
df_ba.columns
# %%
#  ## Load RB data
rb_pickle_filename = "df_rb_reviews_filtered_beers_merged_users.pickle"
df_rb = pd.read_pickle(f"Data/{rb_pickle_filename}")
df_rb.columns
# TODO: wait on RB data, still unassigned style_classes
# You sould be able to deduce this. Since we assigned style_classes on matched, after having filtered this df, we should have all style_classes assigned...
# For now, drop 'UNASSIGNED' style_classes
df_rb = df_rb[df_rb["style_class"] != "UNASSIGNED"]

# %%
# print value counts of style_class 
print(df_ba["style_class"].value_counts())
print(df_rb["style_class"].value_counts())

# %% [markdown]
#  # Rescale Ratings
# Some users rate on average more positively or more negativly with respect to others. Some of this assumed effect will occur, because users rate different beers which indeed are better or worse on average. We hereby assume, that some users are just more positive or negative in general, even if they rate the same beers.
# To counteract this effect, we rescale the ratings to a -1,1 scale where 0 is the users average rating. This way, we can compare users on an equal footing.

# TODO: Should we show that users actually rate things differently on average somehow? if yes, how? Propensity match users on rated beers accross entire dataset and compare averages... And then? 
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
print("aroma", df_ba["aroma"].min(), df_ba["aroma"].max())
print("appearance", df_ba["appearance"].min(), df_ba["appearance"].max())
print("taste", df_ba["taste"].min(), df_ba["taste"].max())
print("palate", df_ba["palate"].min(), df_ba["palate"].max())
print("overall", df_ba["overall"].min(), df_ba["overall"].max())
print("rating", df_ba["rating"].min(), df_ba["rating"].max())

# %%
# Compute the average rating for each user
df_ba["user_average_aroma"] = df_ba.groupby("user_id")["aroma"].transform("mean")
df_ba["user_average_appearance"] = df_ba.groupby("user_id")["appearance"].transform("mean")
df_ba["user_average_taste"] = df_ba.groupby("user_id")["taste"].transform("mean")
df_ba["user_average_palate"] = df_ba.groupby("user_id")["palate"].transform("mean")
df_ba["user_average_overall"] = df_ba.groupby("user_id")["overall"].transform("mean")
df_ba["user_average_rating"] = df_ba.groupby("user_id")["rating"].transform("mean")

# %%
# Rescale the ratings and aspects to a -1,1 scale where 0 is the users average rating

df_ba["aroma_rescaled"] = df_ba.apply(lambda row: rescale(row["aroma"], row["user_average_aroma"]), axis=1)
df_ba["appearance_rescaled"] = df_ba.apply(lambda row: rescale(row["appearance"], row["user_average_appearance"]), axis=1)
df_ba["taste_rescaled"] = df_ba.apply(lambda row: rescale(row["taste"], row["user_average_taste"]), axis=1)
df_ba["palate_rescaled"] = df_ba.apply(lambda row: rescale(row["palate"], row["user_average_palate"]), axis=1)
df_ba["overall_rescaled"] = df_ba.apply(lambda row: rescale(row["overall"], row["user_average_overall"]), axis=1)
df_ba["rating_rescaled"] = df_ba.apply(lambda row: rescale(row["rating"], row["user_average_rating"]), axis=1)

# %%
# Sanity check the ranges of the rescaled ratings and aspects
print("aroma_rescaled", df_ba["aroma_rescaled"].min(), df_ba["aroma_rescaled"].max())
print("appearance_rescaled", df_ba["appearance_rescaled"].min(), df_ba["appearance_rescaled"].max())
print("taste_rescaled", df_ba["taste_rescaled"].min(), df_ba["taste_rescaled"].max())
print("palate_rescaled", df_ba["palate_rescaled"].min(), df_ba["palate_rescaled"].max())
print("overall_rescaled", df_ba["overall_rescaled"].min(), df_ba["overall_rescaled"].max())
print("rating_rescaled", df_ba["rating_rescaled"].min(), df_ba["rating_rescaled"].max())

# %% [markdown]
#  ## RB
# Columns to be rescaled in RB:
# - aroma (1-10)
# - appearance (1-5)
# - taste (1-10)
# - palate (1-5)
# - overall (1-20)
# - rating (0-5)

# %%
# Add 1 to every rating to make it 1-6 instead of 0-5 to prevent divison by zero
df_rb["translated_rating"] = df_rb["rating"] + 1

# %%
# Sanity check the ranges of the ratings and aspects
print("aroma", df_rb["aroma"].min(), df_rb["aroma"].max())
print("appearance", df_rb["appearance"].min(), df_rb["appearance"].max())
print("taste", df_rb["taste"].min(), df_rb["taste"].max())
print("palate", df_rb["palate"].min(), df_rb["palate"].max())
print("overall", df_rb["overall"].min(), df_rb["overall"].max())
print("translated_rating", df_rb["translated_rating"].min(), df_rb["translated_rating"].max())

# %%
# Compute the average rating for each user
df_rb["user_average_aroma"] = df_rb.groupby("user_id")["aroma"].transform("mean")
df_rb["user_average_appearance"] = df_rb.groupby("user_id")["appearance"].transform("mean")
df_rb["user_average_taste"] = df_rb.groupby("user_id")["taste"].transform("mean")
df_rb["user_average_palate"] = df_rb.groupby("user_id")["palate"].transform("mean")
df_rb["user_average_overall"] = df_rb.groupby("user_id")["overall"].transform("mean")
df_rb["user_average_rating"] = df_rb.groupby("user_id")["translated_rating"].transform("mean")

# %%
# Rescale the ratings and aspects to a -1,1 scale where 0 is the users average rating

df_rb["aroma_rescaled"] = df_rb.apply(lambda row: rescale(row["aroma"], row["user_average_aroma"],top=10), axis=1)
df_rb["appearance_rescaled"] = df_rb.apply(lambda row: rescale(row["appearance"], row["user_average_appearance"]), axis=1)
df_rb["taste_rescaled"] = df_rb.apply(lambda row: rescale(row["taste"], row["user_average_taste"],top=10), axis=1)
df_rb["palate_rescaled"] = df_rb.apply(lambda row: rescale(row["palate"], row["user_average_palate"]), axis=1)
df_rb["overall_rescaled"] = df_rb.apply(lambda row: rescale(row["overall"], row["user_average_overall"],20), axis=1)
df_rb["rating_rescaled"] = df_rb.apply(lambda row: rescale(row["translated_rating"], row["user_average_rating"],top=6), axis=1)

# %%
# Sanity check the ranges of the rescaled ratings and aspects
print("aroma_rescaled", df_rb["aroma_rescaled"].min(), df_rb["aroma_rescaled"].max())
print("appearance_rescaled", df_rb["appearance_rescaled"].min(), df_rb["appearance_rescaled"].max())
print("taste_rescaled", df_rb["taste_rescaled"].min(), df_rb["taste_rescaled"].max())
print("palate_rescaled", df_rb["palate_rescaled"].min(), df_rb["palate_rescaled"].max())
print("overall_rescaled", df_rb["overall_rescaled"].min(), df_rb["overall_rescaled"].max())
print("rating_rescaled", df_rb["rating_rescaled"].min(), df_rb["rating_rescaled"].max())

# %% [markdown]
# It is normal that the min of the rescaled ratings is -0.82, the worst score in the rb df is not 0, but 0.5.

# %% [markdown]
#  # Country and state filtering
# Find states and countries with enough ratings
# TODO: arbitrarily pick 1000 as the minimum sample size for now, but this should be changed to something more statistically sound. How?
# For t-tests, pick a lower sample size, e.g. 100.
# %%
MIN_SAMPLE_SIZE = 1000
MIN_TEST_SAMPLE_SIZE = 100

# %%
# Show top 10 countries with most ratings in BA
df_ba["user_country"].value_counts().head(10)

# %%
# Show top 10 states with most ratings in BA
df_ba["user_state"].value_counts().head(10)

# %%
# Show top 10 countries with most ratings in RB
df_rb["user_country"].value_counts().head(10)

# %%
# Show top 10 states with most ratings in RB
df_rb["user_state"].value_counts().head(10)

# %%
# Get list of countries with less than MIN_SAMPLE_SIZE ratings
countries_ba = df_ba["user_country"].value_counts()
countries_ba = countries_ba[countries_ba < MIN_SAMPLE_SIZE].index.tolist()
countries_rb = df_rb["user_country"].value_counts()
countries_rb = countries_rb[countries_rb < MIN_SAMPLE_SIZE].index.tolist()
print("Countries with less than", MIN_SAMPLE_SIZE, "ratings in BA:", len(countries_ba))
print("Countries with less than", MIN_SAMPLE_SIZE, "ratings in RB:", len(countries_rb))
print("Total number of countries in BA:", len(df_ba["user_country"].unique()))
print("Total number of countries in RB:", len(df_rb["user_country"].unique()))

# %%
# Drop all countries in the countries_ba/rb list
df_ba = df_ba[~df_ba["user_country"].isin(countries_ba)]
df_rb = df_rb[~df_rb["user_country"].isin(countries_rb)]

# %%
# How many countries and states are left?
print("Number of countries in BA:", len(df_ba["user_country"].unique()))
print("Number of states in BA:", len(df_ba["user_state"].unique()))
print("Number of countries in RB:", len(df_rb["user_country"].unique()))
print("Number of states in RB:", len(df_rb["user_state"].unique()))

# %%
# Are there states with less than MIN_SAMPLE_SIZE ratings?
states_ba = df_ba["user_state"].value_counts()
states_ba = states_ba[states_ba < MIN_SAMPLE_SIZE].index.tolist()
states_rb = df_rb["user_state"].value_counts()
states_rb = states_rb[states_rb < MIN_SAMPLE_SIZE].index.tolist()
print("States with less than", MIN_SAMPLE_SIZE, "ratings in BA:", len(states_ba))
print("States with less than", MIN_SAMPLE_SIZE, "ratings in RB:", len(states_rb))

# %%
# Drop all states in the states_ba/rb list
df_ba = df_ba[~df_ba["user_state"].isin(states_ba)]
df_rb = df_rb[~df_rb["user_state"].isin(states_rb)]

# %% [markdown]
#  # Test average ratings accross states and countries
# Do states and countries have different average ratings?
# The null hypothesis is that the average ratings are the same for all states and countries.
#
# In order to test this, we do pairwise t-tests among all possible pairs of states and countries respectively. We arbitrarily pick a minimum number of ratings per state/country (`MIN_SAMPLE_SIZE`). If a state/country has less than `MIN_SAMPLE_SIZE` ratings, we do not consider it for the t-tests.
# 
# We use the Sidak correction to correct for multiple comparisons. Suppose we have a list of states/countries with `n` elements. We perform `n(n-1)/2` t-tests. The Sidak correction is `1 - (1 - alpha)^(1/n)`, where `alpha` is the significance level (default = 0.05).
#
# Next, we want to figure out, if the average rating differs among style_classes, styles or even individual beers. Thus, we repeat the same procedure for each style_class, style and beer. Here we allow for a lower sample size (`MIN_TEST_SAMPLE_SIZE`).
# 

# Function to perform pairwise t-tests
# %%
def pairwise_ttests(df, entity_list, column_name_to_test,alpha = 0.05, entity_column_name = "entity", recompute = False):
    """
    Perform pairwise t-tests on a given column for each pair of entities in a given entity list
    Computes the Sidak correction.
    arguments:
        df: dataframe to perform t-tests on
        entity_list: list of entities to perform t-tests on
        entity_list: list of entities to perform t-tests on (will most likely be a list of countries or states)
        column_name_to_test: column name to perform t-tests on (should be a rating column)
        alpha: significance level (optional, default = 0.05)
        entity_column_name: name of the entity column (optional, default = "entity")
        recompute: boolean to recompute the results (optional, default = False)
    returns:
        df_t_test_results: dataframe containing the results of the t-tests
        alpha_1: the Sidak corrected significance level
    """
    # Test the arguments for valid input
    # Check if the df is a dataframe and non-empty
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise TypeError("df must be a non-empty pandas dataframe")
    # Check if the entity_list is a list and non-empty
    if not isinstance(entity_list, list) or len(entity_list) == 0:
        raise TypeError("entity_list must be a non-empty list")
    # Check if the column_name_to_test is a string and non-empty
    if not isinstance(column_name_to_test, str) or len(column_name_to_test) == 0:
        raise TypeError("column_name_to_test must be a non-empty string")
    # Check if the entity_column_name is a string and non-empty
    if not isinstance(entity_column_name, str) or len(entity_column_name) == 0:
        raise TypeError("entity_column_name must be a non-empty string")
    # Check if the alpha is a float and between 0 and 1
    if not isinstance(alpha, float) or alpha < 0 or alpha > 1:
        raise TypeError("alpha must be a float between 0 and 1")
    # Check if the recompute is a boolean
    if not isinstance(recompute, bool):
        raise TypeError("recompute must be a boolean")
    # Check if the column_name_to_test is in the dataframe
    if column_name_to_test not in df.columns:
        raise ValueError("column_name_to_test must be in the dataframe")
    # Check if the entity_column_name is in the dataframe
    if entity_column_name not in df.columns:
        raise ValueError("entity_column_name must be in the dataframe")
    # Check if the entity_list is a subset of the entity_column_name
    if not set(entity_list).issubset(set(df[entity_column_name].unique())):
        raise ValueError("entity_list must be a subset of the unique values in the entity_column_name")
    


    # Compute the Sidak correction
    # Compute n: the number of independent t-tests performed
    combined_entity_count = len(entity_list)
    # (n-1) + (n-2) + ... + 1
    n = combined_entity_count * (combined_entity_count - 1) / 2
    # Compute the Sidak correction
    alpha_1 = 1-(1-alpha)**(1/n)


    # Create a hash of the arguments to check if the results have already been computed
    # make a string of all the arguments
    arguments = str(entity_list) + str(column_name_to_test) + str(alpha) + str(entity_column_name)
    print("arguments:", arguments)
    # hash the string
    hashed_arguments = hash(arguments)
    print("hashed_arguments:", hashed_arguments)
    if not recompute:
        # check if a folder for t-test results exists
        if not os.path.exists("t_test_results"):
            os.mkdir("t_test_results")
        # check if a csv file with the hashed arguments exists
        if os.path.exists(f"t_test_results/{hashed_arguments}.csv"):
            # load the csv file
            print("loading t-test results from csv file")
            df_t_test_results = pd.read_csv(f"t_test_results/{hashed_arguments}.csv")
            return df_t_test_results, alpha_1


    df_t_test_results = pd.DataFrame(columns = [f"{entity_column_name}1", f"{entity_column_name}2", "t-statistic", "p-value", "significant","average_entity1","average_entity2","std_entity1","std_entity2"])
    

    # Perform t-tests for each pair of countries
    for i in range(combined_entity_count):
        entity1 = entity_list[i]
        for j in range(i+1, combined_entity_count):
            entity2 = entity_list[j]
            # Perform t-test
            print("performing t-test for", entity1, "and", entity2)
            data_entity1 = df[df[entity_column_name] == entity1][column_name_to_test].to_numpy(dtype=float)
            data_entity2 = df[df[entity_column_name] == entity2][column_name_to_test].to_numpy(dtype=float)

            
            t_statistic, p_value = stats.ttest_ind(data_entity1, data_entity2, equal_var=False) # perform t-test, make sure to flag that the variances are not equal

            # Check if the result is significant
            significant = p_value < alpha_1
            # Add result to dataframe
            nr = pd.DataFrame({f"{entity_column_name}1":[ entity1], f"{entity_column_name}2":[ entity2], "t-statistic":[ t_statistic], "p-value":[ p_value], "significant":[ significant],"average_entity1":[np.mean(data_entity1)],"std_entity1":[np.std(data_entity1)], "average_entity2":[np.mean(data_entity2)],"std_entity2":[np.std(data_entity2)]})
            df_t_test_results = pd.concat([df_t_test_results, nr], ignore_index=True)

    # Save the results
    if not os.path.exists("t_test_results"):
        os.mkdir("t_test_results")
    df_t_test_results.to_csv(f"t_test_results/{hashed_arguments}.csv", index=False)

    return df_t_test_results, alpha_1


# %% [markdown]
# ### T-Testing for different countries
# %%
# Create a list of aspects for ba
ba_aspects = ["aroma", "appearance", "taste", "palate", "overall", "rating"] 
# Get a list of all countries
country_list = list(df_ba["user_country"].unique())
# Perform t-tests for each pair of countries
dfs_t_tests_countries = {}
for aspect in ba_aspects:
    dfs_t_tests_countries[aspect] = pairwise_ttests(df_ba,country_list,aspect,entity_column_name="user_country")
    aspect_rescaled = aspect + "_rescaled"
    dfs_t_tests_countries[aspect_rescaled] = pairwise_ttests(df_ba,country_list,aspect_rescaled,entity_column_name="user_country")

# TODO: rank countries by average rating for each aspect
# TODO: plot the results
# TODO: plot and discuss number of significant results for each aspect
# TODO: compare to rescaled results and discuss

# %%
# Compare the results for the different aspects
# How many significant results are there for each aspect?
for aspect in ba_aspects:
    print(aspect, ":", len(dfs_t_tests_countries[aspect][dfs_t_tests_countries[aspect]["significant"] == True]))
    aspect_rescaled = aspect + "_rescaled"
    print(aspect_rescaled, ":", len(dfs_t_tests_countries[aspect_rescaled][dfs_t_tests_countries[aspect_rescaled]["significant"] == True]))


# %% [markdown]
# ### T-Testing for different US states


# %% [markdown]
# ## Function to perform pairwise t-tests for each pair of styles within a country
# %%
def perform_independend_t_tests_on_given_country(df, country, column_name_to_test, alpha = 0.05):
    """
    Perform t-tests on a given column for each pair of styles within a given country
    Computes the Sidak correction.
    arguments:
        df: dataframe to perform t-tests on
        country: country to perform t-tests on
        column_name_to_test: column name to perform t-tests on (should be a rating column)
    returns:
        df_t_test_results: dataframe containing the results of the t-tests
    """
    # TODO:...

# TODO: similar fct. for pairwise t-tests for different styles within a country
# TODO: how to do cross-country comparisons in beer-styles?


# %% [markdown]
# Testing the above function
# %%
# Test the above function
country_list = ["Germany", "United States", "Canada"]
state_list = ["California", "New York"]
column_name = "rating"
df_t_test_results, alpha_1 = perform_independend_t_tests_on_given_country_list(df_ba, country_list, state_list, column_name)
print("alpha_1:", alpha_1)
df_t_test_results
# %% [markdown]
# Compare all US states
# %%
# Compare all US states on the BA dataset
country_list = ["United States"]
state_list = list(df_ba["states"].dropna().unique())
column_name = "rating"
df_ba_t_test_results_us_states, alpha_1 = perform_independend_t_tests_on_given_country_list(df_ba, country_list, state_list, column_name)
print("alpha_1:", alpha_1)
df_ba_t_test_results_us_states
# %%
df_ba_t_test_results_us_states["significant"].value_counts()

# %%
# Compare all US states on the RB dataset
country_list = ["United States"]
state_list = list(df_rb["states"].dropna().unique())
column_name = "rating"
df_rb_t_test_results_us_states, alpha_1 = perform_independend_t_tests_on_given_country_list(df_rb, country_list, state_list, column_name)
print("alpha_1:", alpha_1)
df_rb_t_test_results_us_states
# %%
df_rb_t_test_results_us_states["significant"].value_counts()

# %%
# Compare all US states on the RB dataset on palate
country_list = ["United States"]
state_list = list(df_rb["states"].dropna().unique())
column_name = "palate"
df_rb_t_test_results_us_states, alpha_1 = perform_independend_t_tests_on_given_country_list(df_rb, country_list, state_list, column_name)
print("alpha_1:", alpha_1)
df_rb_t_test_results_us_states
# %%
df_rb_t_test_results_us_states["significant"].value_counts()




# %%

states_codes = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    # 'DC': 'District of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'IA': 'Iowa',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'MA': 'Massachusetts',
    'MD': 'Maryland',
    'ME': 'Maine',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MO': 'Missouri',
    'MS': 'Mississippi',
    'MT': 'Montana',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'NE': 'Nebraska',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NV': 'Nevada',
    'NY': 'New York',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VA': 'Virginia',
    'VT': 'Vermont',
    'WA': 'Washington',
    'WI': 'Wisconsin',
    'WV': 'West Virginia',
    'WY': 'Wyoming'
}

def get_key(val):
    for key, value in states_codes.items():
        if val == value:
            return key


# %%
df_t_test_to_plot = df_rb_t_test_results_us_states[["entity1", "entity2", "p-value"]]

df_t_test_to_plot = df_t_test_to_plot[~(df_t_test_to_plot["entity1"]=='United States')]

df_t_test_to_plot["state_code_entity1"] = df_t_test_to_plot["entity1"].apply(lambda x: get_key(x))
df_t_test_to_plot["state_code_entity2"] = df_t_test_to_plot["entity2"].apply(lambda x: get_key(x))

df_t_test_to_plot.drop("entity1", axis=1, inplace=True)
df_t_test_to_plot.drop("entity2", axis=1, inplace=True)


#%%

df_plot_p_value = pd.DataFrame()
for key in states_codes.keys():
    c = df_t_test_to_plot[df_t_test_to_plot["state_code_entity1"].isin([key]) | df_t_test_to_plot["state_code_entity2"].isin([key])]
    c["state_code_entity1"] = np.where(c["state_code_entity1"] == key, c["state_code_entity2"], c["state_code_entity1"])
    c["state_code_entity2"] = states_codes.get(key)
    c.rename(columns={"state_code_entity2": "State"}, inplace=True)
    c = c.append({'p-value': 0, "state_code_entity1": key, "year": states_codes.get(key)}, ignore_index = True)
    c = c.sort_values(by=["state_code_entity1"])
    frames = [df_plot_p_value, c]
    df_plot_p_value = pd.concat(frames)
    
    

# %%
fig = px.choropleth(df_plot_p_value,
                    locations='state_code_entity1', 
                    locationmode="USA-states", 
                    color='p-value',
                    color_continuous_scale="Viridis_r", 
                    scope="usa",
                    title="Independant T-test between 2 US States",
                    animation_frame='State') #make sure 'period_begin' is string type and sorted in ascending order



fig.show()
print("If the p-value is smaller than our threshold, then we have evidence against the null hypothesis of equal population means.")




# %% [markdown]
# # Subsetting the data to specific style classes, styles and maybe some beers...
# %%
# Load styles





