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
df_ba_ratings_filtered_beers_merged_users = pd.read_pickle(f"Data/{ba_pickle_filename}")
df_ba_ratings_filtered_beers_merged_users.columns
# %%
# Load RB data
rb_pickle_filename = "df_rb_reviews_filtered_beers_merged_users.pickle"
df_rb_reviews_filtered_beers_merged_users = pd.read_pickle(f"Data/{rb_pickle_filename}")
df_rb_reviews_filtered_beers_merged_users.columns

# %% [markdown]
# # Function to perform t-tests of a given column for each pair of countries in a given country list/ or state list
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
        TODO: define how exactly the results are stored and if it should print out significant results
    """
    # TODO: improve edge-cases like empty country_list, empty state_list, empty df, and empty column_name_to_test for a given country/state

    # Compute n: the number of independent t-tests performed
    entity_list = country_list + state_list
    combined_entity_count = len(entity_list)
    # (n-1) + (n-2) + ... + 1
    n = combined_entity_count * (combined_entity_count - 1) / 2
    # Compute the Sidak correction
    alpha_1 = 1-(1-alpha)**(1/n)
    
    
    # Create filtered dataframe containing only the given countries and states and the given column
    df_filtered = df[df["country"].isin(country_list) | df["states"].isin(state_list)]
    df_filtered = df_filtered[[column_name_to_test, "country", "states"]]
    print("successfully filtered dataframe")

    # TODO: figure out how to deal with the US...

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

    df_t_test_results = pd.DataFrame(columns = ["entity1", "entity2", "t-statistic", "p-value", "significant","average_entity1","average_entity2","std_entity1","std_entity2"])
    

    # Perform t-tests for each pair of countries
    for i in range(combined_entity_count):
        entity1 = entity_list[i]
        for j in range(i+1, combined_entity_count):
            entity2 = entity_list[j]
            # Perform t-test
            print("performing t-test for", entity1, "and", entity2)
            data_entity1 = df_filtered[df_filtered["entity"] == entity1][column_name_to_test].to_numpy(dtype=float)
            data_entity2 = df_filtered[df_filtered["entity"] == entity2][column_name_to_test].to_numpy(dtype=float)
            # print("entity1:", data_entity1.shape, data_entity1.dtype)
            # print("entity2:", data_entity2.shape, data_entity2.dtype)

            # TODO: This test assumes that the populations have identical variances by default.
            # -> rescaling necessary?
            t_statistic, p_value = stats.ttest_ind(data_entity1, data_entity2, equal_var=False)

            # Check if the result is significant
            significant = p_value < alpha_1
            # Add result to dataframe
            nr = pd.DataFrame({"entity1":[ entity1], "entity2":[ entity2], "t-statistic":[ t_statistic], "p-value":[ p_value], "significant":[ significant],"average_entity1":[np.mean(data_entity1)],"std_entity1":[np.std(data_entity1)], "average_entity2":[np.mean(data_entity2)],"std_entity2":[np.std(data_entity2)]})
            df_t_test_results = pd.concat([df_t_test_results, nr], ignore_index=True)
            

            # if entity1 == "Germany" and entity2 == "United States":
            #     plt.hist(data_entity1, bins=20, alpha=0.5, label=entity1,density=True)
            #     plt.hist(data_entity2, bins=20, alpha=0.5, label=entity2,density=True)
            #     plt.legend(loc='upper right')
            #     plt.show()
            
    return df_t_test_results, alpha_1

# %% [markdown]
# Testing the above function
# %%
# Test the above function
country_list = ["Germany", "United States", "Canada"]
state_list = ["California", "New York"]
column_name = "rating"
df_t_test_results, alpha_1 = perform_independend_t_tests_on_given_country_list(df_ba_ratings_filtered_beers_merged_users, country_list, state_list, column_name)
print("alpha_1:", alpha_1)
df_t_test_results
# %% [markdown]
# Compare all US states
# %%
# Compare all US states on the BA dataset
country_list = ["United States"]
state_list = list(df_ba_ratings_filtered_beers_merged_users["states"].dropna().unique())
column_name = "rating"
df_ba_t_test_results_us_states, alpha_1 = perform_independend_t_tests_on_given_country_list(df_ba_ratings_filtered_beers_merged_users, country_list, state_list, column_name)
print("alpha_1:", alpha_1)
df_ba_t_test_results_us_states
# %%
df_ba_t_test_results_us_states["significant"].value_counts()

# %%
# Compare all US states on the RB dataset
country_list = ["United States"]
state_list = list(df_rb_reviews_filtered_beers_merged_users["states"].dropna().unique())
column_name = "rating"
df_rb_t_test_results_us_states, alpha_1 = perform_independend_t_tests_on_given_country_list(df_rb_reviews_filtered_beers_merged_users, country_list, state_list, column_name)
print("alpha_1:", alpha_1)
df_rb_t_test_results_us_states
# %%
df_rb_t_test_results_us_states["significant"].value_counts()

# %% [markdown]
# Make comparisons between beer styles
# TODO: figure out how to do this meaningfully
# Problem: What exactly are we trying to achieve here? 
# 
# %%
df_ba_ratings_filtered_beers_merged_users.columns
beer_styles = df_ba_ratings_filtered_beers_merged_users["style"].unique()




# %%
