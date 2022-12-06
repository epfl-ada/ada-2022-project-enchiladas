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

# %%
# # Are beer style preferences influenced by geography/culture?
# Investigate RQ1 by viewing ratings for different beer styles for each country. Specifically, we want conduct t-tests investigating if there are differences in this between countries, using the Sidak correction.

# For each country / for each state:
# compute the CI Sidak correction
# For Each Beerstyle: 
# - Perform a t-test comparing means of each country

# %%
# Load BA data
ba_pickle_filename = "df_ba_ratings_filtered_beers_merged_users.pickle"
df_ba = pd.read_pickle(f"Data/{ba_pickle_filename}")
df_ba.columns
# %%
# Load RB data
rb_pickle_filename = "df_rb_reviews_filtered_beers_merged_users.pickle"
df_rb = pd.read_pickle(f"Data/{rb_pickle_filename}")
df_rb.columns
# TODO: wait on RB data, still unassigned style_classes

# %% [markdown]
# # Function to perform pairwise t-tests
# %%
def pairwise_ttests(df, entity_list, column_name_to_test,alpha = 0.05, entity_column_name = "entity"):
    """
    Perform pairwise t-tests on a given column for each pair of entities in a given entity list
    Computes the Sidak correction.
    arguments:
        df: dataframe to perform t-tests on
        entity_list: list of entities to perform t-tests on
        column_name_to_test: column name to perform t-tests on (should be a rating column)
        alpha: significance level (optional, default = 0.05)
    returns:
        df_t_test_results: dataframe containing the results of the t-tests
        alpha_1: the Sidak corrected significance level
    """
    # Compute n: the number of independent t-tests performed
    combined_entity_count = len(entity_list)
    # (n-1) + (n-2) + ... + 1
    n = combined_entity_count * (combined_entity_count - 1) / 2
    # Compute the Sidak correction
    alpha_1 = 1-(1-alpha)**(1/n)

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

    return df_t_test_results, alpha_1

# %% [markdown]
# ## Function to perform t-tests of a given column for each pair of countries in a given country list/ or state list
# %%
def perform_independend_t_tests_on_given_country_list(df, country_list, state_list, column_name_to_test, alpha = 0.05):
    """
    Perform t-tests on a given column for each pair of countries in a given country list and state list
    Computes the Sidak correction.
    arguments:
        df: dataframe to perform t-tests on
        country_list: list of countries to perform t-tests on
        state_list: list of states to perform t-tests on (if both country_list and state_list are given, combine them)
        column_name_to_test: column name to perform t-tests on (should be a rating column)
    returns:
        df_t_test_results: dataframe containing the results of the t-tests 
        alpha_1: the Sidak corrected significance level
        TODO: define how exactly the results are stored and if it should print out significant results
    """
    # TODO: improve edge-cases like empty country_list, empty state_list, empty df, and empty column_name_to_test for a given country/state
    
    # Create filtered dataframe containing only the given countries and states and the given column
    df_filtered = df[df["country"].isin(country_list) | df["states"].isin(state_list)]
    df_filtered = df_filtered[[column_name_to_test, "country", "states"]]
    print("successfully filtered dataframe")

    # Assign entity names to each row
    def assign_entity_name(row):
        if row["country"] in country_list and row["country"] != "United States":
            return row["country"]
        elif row["states"] in state_list:
            return row["states"]
        else:
            return None

    df_filtered["entity"] = df_filtered.apply(lambda row: assign_entity_name(row), axis=1)
    # Create duplicate entries for the US
    
    if "United States" in country_list:
        df_filtered_us = df_filtered[df_filtered["country"] == "United States"]
        df_filtered_us["entity"] = "United States"
        df_filtered = pd.concat([df_filtered, df_filtered_us])
    df_filtered = df_filtered.drop(columns=["country", "states"])
    df_filtered = df_filtered.dropna()
    print("NaN-values: ",df_filtered.isna().sum())
    # df_filtered = df_filtered.reset_index() # TODO: necessary?

    print(df_filtered.head())

    df_t_test_results, alpha_1 = pairwise_ttests(df_filtered, country_list + state_list, column_name_to_test, alpha)            
            
    return df_t_test_results, alpha_1

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

# %% [markdown]
# # Subsetting the data to specific styles...
# %%
# Load styles



# %% [markdown]
# Make comparisons between beer styles
# TODO: figure out how to do this meaningfully
# Problem: What exactly are we trying to achieve here? 
# 
# %%
df_ba.columns
beer_styles = df_ba["style"].unique()




# %% [markdown]
# # Ideas for further analysis
# What do we do with the highly significant differences between states/countries in average rating?
# My Issue is, that I don't know how to tell a story about this... Do we just pick examples of similarities?
# Furthermore, just stating that average ratings are significantly different between states/countries is naked. 
# Things we could do:
# - Order countries based on their average rating. I.e. we figure out who rates best/worst on average? Do they actually rate the same beers?
# - Would one have to do propensity matching on two countries? I.e. in order to compare the ratings of two countries, one would have to match the beers, or their styles that are rated by both countries. 
# - cluster countries based on their distribution of rated beers. We need an embedding for rated beer styles. 
# In order to figure out, if there exists a country that rates a give 
# This kind of diverges from our research statement...
# 


