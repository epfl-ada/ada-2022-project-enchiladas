# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import scipy.stats as stats
from helpers import * #custom made functions
import plotly.express as px
from prettify import * #custom made functions
import hashlib
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
# TODO: deepen this explanation

# %%
#  ## Load BA data
ba_pickle_filename = "df_ba_ratings_filtered_beers_merged_users.pickle"
df_ba = pd.read_pickle(f"Data/{ba_pickle_filename}")
df_ba.info()
# %%
df_ba.isna().sum()

# %%
#  ## Load RB data
rb_pickle_filename = "df_rb_reviews_filtered_beers_merged_users.pickle"
df_rb = pd.read_pickle(f"Data/{rb_pickle_filename}")
df_rb.info()

# %%
df_rb.isna().sum()

# %%
# print value counts of style_class 
print(df_ba["style_class"].value_counts())
print(df_rb["style_class"].value_counts())

# %% [markdown]
#  # Country and state filtering
# Find states and countries with enough ratings. We want to have at least 1000 ratings per country/state. Otherwise we discard the country/state. Looking at the distributions of ratings per country, we see that there are not many countries with more than 1000 ratings. Our data is very US-centric.

# %%
# Plot the number of ratings per country in BA top 20
df_ba["user_country"].value_counts().head(20).plot(kind="bar", title="Number of ratings per country in BA",logy=True)
plt.xlabel("Country")
plt.ylabel("Number of ratings")
plt.tight_layout()
plt.savefig("Plots/BA_ratings_per_country.png",bbox_inches='tight')

# %%
# Plot the number of ratings per user_state in BA last 20
df_ba["user_state"].value_counts().tail(20).plot(kind="bar", title="Number of ratings per state in BA (top 20)",logy=True)
plt.xlabel("State")
plt.ylabel("Number of ratings")
plt.tight_layout()
plt.savefig("Plots/BA_ratings_per_state.png",bbox_inches='tight')

# %%
# Plot the number of ratings per user_state in RB last 20
df_rb["user_state"].value_counts().tail(20).plot(kind="bar", title="Number of ratings per state in RB (top 20)",logy=True)
plt.xlabel("State")
plt.ylabel("Number of ratings")
plt.tight_layout()
plt.savefig("Plots/RB_ratings_per_state.png",bbox_inches='tight')

# %%
MIN_SAMPLE_SIZE = 1000

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

# %% [markdown]
# Note that we do not drop states with less that 'MIN_SAMPLE_SIZE' ratings. We are aware of the fact that there are some states with few ratings, but we prefer keeping the data in order to gain a more complete picture.

# %% [markdown]
#  # Test average ratings accross states and countries
# Do states and countries have different average ratings?
# The null hypothesis is that the average ratings are the same for all states and countries.
#
# In order to test this, we do pairwise t-tests among all possible pairs of states and countries respectively. We arbitrarily pick a minimum number of ratings per state/country (`MIN_SAMPLE_SIZE`). If a state/country has less than `MIN_SAMPLE_SIZE` ratings, we do not consider it for the t-tests.
# 
# We use the Sidak correction to correct for multiple comparisons. Suppose we have a list of states/countries with `n` elements. We perform `n(n-1)/2` t-tests. The Sidak correction is `1 - (1 - alpha)^(1/n)`, where `alpha` is the significance level (default = 0.05).
#
# Next, we want to figure out, if the average rating differs among style_classes, styles or even individual beers. Thus, we repeat the same procedure for each style_class and style. Here we allow for a lower sample size (`MIN_TEST_SAMPLE_SIZE`) which we set to 100. We do this to be able to perform tests even on countries/states with only a few ratings per style/style class.
# 
# %%
MIN_TEST_SAMPLE_SIZE = 100

# %%
# Function to perform pairwise t-tests
def pairwise_ttests(df, entity_list, column_name_to_test,alpha = 0.05, entity_column_name = "entity", recompute = False, datasource="ba", min_sample_size=MIN_SAMPLE_SIZE):
    """
    Perform pairwise t-tests on a given column for each pair of entities in a given entity list
    Computes the Sidak correction.
    arguments:
        df: dataframe to perform t-tests on
        entity_list: list of entities to perform t-tests on (will most likely be a list of countries or states)
        column_name_to_test: column name to perform t-tests on (should be a rating column)
        alpha: significance level (optional, default = 0.05)
        entity_column_name: name of the entity column (optional, default = "entity")
        recompute: boolean to recompute the results (optional, default = False)
        datasource: string to specify the datasource (optional, default = "ba")
    returns:
        df_t_test_results: dataframe containing the results of the t-tests
        alpha_1: the Sidak corrected significance level
    """
    # Test the arguments for valid input
    # Check if the df is a dataframe and non-empty
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise TypeError("df must be a non-empty pandas dataframe")
    # Check if the entity_list is a list and non-empty
    if not isinstance(entity_list, list) or len(entity_list) == 0 or len(entity_list) == 1:
        raise TypeError("entity_list must be a non-empty list bigger than 1")
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

    hashfct = hashlib.sha256()
    # Create a hash of the arguments to check if the results have already been computed
    # Make sure the list only contains strings
    entity_list = [str(entity) for entity in entity_list]
    # make a string of all the arguments
    entity_list.sort()
    arguments = "-".join(entity_list) + str(column_name_to_test) + f"{alpha:.5f}"+ str(entity_column_name) + str(datasource) + str(min_sample_size)
    
    # hash the string
    hashfct.update(arguments.encode("utf8"))
    hashed_arguments = hashfct.hexdigest()
    
    if not recompute:
        # check if a folder for t-test results exists
        if not os.path.exists("t_test_results"):
            os.mkdir("t_test_results")
        # check if a csv file with the hashed arguments exists
        if os.path.exists(f"t_test_results/{hashed_arguments}.csv"):
            # load the csv file
            print("loading t-test results from a precomputed csv file")
            df_t_test_results = pd.read_csv(f"t_test_results/{hashed_arguments}.csv")
            return df_t_test_results, alpha_1


    df_t_test_results = pd.DataFrame(columns = [f"{entity_column_name}1", f"{entity_column_name}2", "t-statistic", "p-value", "significant","average_entity1","average_entity2","std_entity1","std_entity2","had_enough_data"])
    

    # Perform t-tests for each pair of countries
    for i in range(combined_entity_count):
        entity1 = entity_list[i]
        for j in range(i+1, combined_entity_count):
            entity2 = entity_list[j]
            # Perform t-test
            print("performing t-test for", entity1, "and", entity2)
            data_entity1 = df[df[entity_column_name] == entity1][column_name_to_test].to_numpy(dtype=float)
            data_entity2 = df[df[entity_column_name] == entity2][column_name_to_test].to_numpy(dtype=float)

            # Test if there is enough data to perform the t-test
            if len(data_entity1) < min_sample_size or len(data_entity2) < min_sample_size:
                print("not enough data to perform t-test for", entity1, "and", entity2)
                nr = pd.DataFrame({f"{entity_column_name}1":[ entity1], f"{entity_column_name}2":[ entity2], "t-statistic":[ np.nan], "p-value":[ np.nan], "significant":[ np.nan],"average_entity1":[np.nan],"std_entity1":[np.nan], "average_entity2":[np.nan],"std_entity2":[np.nan],"had_enough_data":[False]})
                df_t_test_results = pd.concat([df_t_test_results, nr], ignore_index=True)
                continue

            
            t_statistic, p_value = stats.ttest_ind(data_entity1, data_entity2, equal_var=False) # perform t-test, make sure to flag that the variances are not equal

            # Check if the result is significant
            significant = p_value < alpha_1
            # Add result to dataframe
            nr = pd.DataFrame({f"{entity_column_name}1":[ entity1], f"{entity_column_name}2":[ entity2], "t-statistic":[ t_statistic], "p-value":[ p_value], "significant":[ significant],"average_entity1":[np.mean(data_entity1)],"std_entity1":[np.std(data_entity1)], "average_entity2":[np.mean(data_entity2)],"std_entity2":[np.std(data_entity2)],"had_enough_data":[True]})
            df_t_test_results = pd.concat([df_t_test_results, nr], ignore_index=True)

    # Save the results
    if not os.path.exists("t_test_results"):
        os.mkdir("t_test_results")
    df_t_test_results.to_csv(f"t_test_results/{hashed_arguments}.csv", index=False)

    return df_t_test_results, alpha_1


# %% [markdown]
#  ### T-Testing for different countries
# For each aspect, we perform a t-test for each pair of countries. We use the Sidak correction to account for multiple comparisons.
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


# %%
dfs_t_tests_countries["aroma"][0].head()

# %%
# Compare the results for the different aspects
# How many significant results are there for each aspect?
df_t_tests_countries_results = {"significant":[], "total":[], "had_enough_data":[],"aspect":[]}
for aspect in ba_aspects:
    # Fill the results dictionary
    df_t_tests_countries_results["significant"].append(len(dfs_t_tests_countries[aspect][0][dfs_t_tests_countries[aspect][0]["significant"] == True]))
    df_t_tests_countries_results["total"].append(len(dfs_t_tests_countries[aspect][0]))
    df_t_tests_countries_results["had_enough_data"].append(len(dfs_t_tests_countries[aspect][0][dfs_t_tests_countries[aspect][0]["had_enough_data"] == True]))
    df_t_tests_countries_results["aspect"].append(aspect)
    # Fill the results dictionary for the rescaled aspects
    aspect_rescaled = aspect + "_rescaled"
    df_t_tests_countries_results["significant"].append(len(dfs_t_tests_countries[aspect_rescaled][0][dfs_t_tests_countries[aspect_rescaled][0]["significant"] == True]))
    df_t_tests_countries_results["total"].append(len(dfs_t_tests_countries[aspect_rescaled][0]))
    df_t_tests_countries_results["had_enough_data"].append(len(dfs_t_tests_countries[aspect_rescaled][0][dfs_t_tests_countries[aspect_rescaled][0]["had_enough_data"] == True]))
    df_t_tests_countries_results["aspect"].append(aspect_rescaled)
df_t_tests_countries_results = pd.DataFrame(df_t_tests_countries_results)    

# %%
df_t_tests_countries_results

# %%
# Compute the ratio of significant results
df_t_tests_countries_results["ratio"] = df_t_tests_countries_results["significant"] / df_t_tests_countries_results["had_enough_data"]

# %%
# The significance level for the Sidak correction (it's the same for all aspects)
alpha_l = dfs_t_tests_countries["aroma"][1]
print(f"The Sidak corrected significance level {alpha_l:.2}")


# %% [markdown]
# Rescaling the ratings by their user's average decreases the number of significant t-tests for each aspect. It is interesting to see that there are still some countries that rate on average differently. We will investigate this further later on.

# %%
# Plot the distribution of ratings of a pair where the t-test is significant
# Get the first significant result
aspect = "rating"
aspect_rescaled = aspect + "_rescaled"
df_t_test_results_of_rating = dfs_t_tests_countries[aspect][0]
df_t_test_results_of_rating_rescaled = dfs_t_tests_countries[aspect_rescaled][0]
# Only keep significant results
df_t_test_results_of_rating = df_t_test_results_of_rating[df_t_test_results_of_rating["significant"] == True]
df_t_test_results_of_rating_rescaled = df_t_test_results_of_rating_rescaled[df_t_test_results_of_rating_rescaled["significant"] == True]
# Get the first significant result
first_significant_result = df_t_test_results_of_rating.iloc[0]
first_significant_result_rescaled = df_t_test_results_of_rating_rescaled.iloc[0]
# Get the data for the first significant result
entity1 = first_significant_result["user_country1"]
entity2 = first_significant_result["user_country2"]
entity1_rescaled = first_significant_result_rescaled["user_country1"]
entity2_rescaled = first_significant_result_rescaled["user_country2"]
data_entity1 = df_ba[df_ba["user_country"] == entity1]["rating"].to_numpy(dtype=float)
data_entity2 = df_ba[df_ba["user_country"] == entity2]["rating"].to_numpy(dtype=float)
data_entity1_rescaled = df_ba[df_ba["user_country"] == entity1_rescaled]["rating_rescaled"].to_numpy(dtype=float)
data_entity2_rescaled = df_ba[df_ba["user_country"] == entity2_rescaled]["rating_rescaled"].to_numpy(dtype=float)
# Plot the distribution of ratings
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(data_entity1, bins=20, alpha=0.5, label=entity1, density=True)
ax[0].hist(data_entity2, bins=20, alpha=0.5, label=entity2, density=True)
ax[0].set_title("Distribution of ratings")
ax[0].set_xlabel("Rating")
ax[0].set_ylabel("Count")
ax[0].legend()
ax[1].hist(data_entity1_rescaled, bins=20, alpha=0.5, label=entity1_rescaled, density=True)
ax[1].hist(data_entity2_rescaled, bins=20, alpha=0.5, label=entity2_rescaled, density=True)
ax[1].set_title("Distribution of ratings (rescaled)")
ax[1].set_xlabel("Rating")
ax[1].set_ylabel("Count")
ax[1].legend()
plt.tight_layout()
plt.savefig("Plots/distribution_of_ratings_of_a_pair_where_the_t_test_is_significant.png",bbox_inches='tight')
plt.show()

# %%
# Plot the distribution of ratings of a pair where the t-test is not significant
df_t_test_results_of_rating = dfs_t_tests_countries[aspect][0]
df_t_test_results_of_rating_rescaled = dfs_t_tests_countries[aspect_rescaled][0]
# Only keep non-significant results
df_t_test_results_of_rating = df_t_test_results_of_rating[df_t_test_results_of_rating["significant"] == False]
df_t_test_results_of_rating_rescaled = df_t_test_results_of_rating_rescaled[df_t_test_results_of_rating_rescaled["significant"] == False]
# Get the first non-significant result
first_significant_result = df_t_test_results_of_rating.iloc[0]
first_significant_result_rescaled = df_t_test_results_of_rating_rescaled.iloc[0]
# Get the data for the first non-significant result
entity1 = first_significant_result["user_country1"]
entity2 = first_significant_result["user_country2"]
entity1_rescaled = first_significant_result_rescaled["user_country1"]
entity2_rescaled = first_significant_result_rescaled["user_country2"]
data_entity1 = df_ba[df_ba["user_country"] == entity1]["rating"].to_numpy(dtype=float)
data_entity2 = df_ba[df_ba["user_country"] == entity2]["rating"].to_numpy(dtype=float)
data_entity1_rescaled = df_ba[df_ba["user_country"] == entity1_rescaled]["rating_rescaled"].to_numpy(dtype=float)
data_entity2_rescaled = df_ba[df_ba["user_country"] == entity2_rescaled]["rating_rescaled"].to_numpy(dtype=float)
# Plot the distribution of ratings
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(data_entity1, bins=20, alpha=0.5, label=entity1, density=True)
ax[0].hist(data_entity2, bins=20, alpha=0.5, label=entity2, density=True)
ax[0].set_title("Distribution of ratings")
ax[0].set_xlabel("Rating")
ax[0].set_ylabel("Count")
ax[0].legend()
ax[1].hist(data_entity1_rescaled, bins=20, alpha=0.5, label=entity1_rescaled, density=True)
ax[1].hist(data_entity2_rescaled, bins=20, alpha=0.5, label=entity2_rescaled, density=True)
ax[1].set_title("Distribution of ratings (rescaled)")
ax[1].set_xlabel("Rating")
ax[1].set_ylabel("Count")
ax[1].legend()
plt.tight_layout()
plt.savefig("Plots/distribution_of_ratings_of_a_pair_where_the_t_test_is_not_significant.png",bbox_inches='tight')
plt.show()

# %% [markdown]
# Let's plot the distributions of rating for each aspect and all countries.

# %%
def plot_boxplot(df, aspect, countries, attribute="user_country",filename=None):
    """
    Plots a boxplot for the specified aspect
    """
    _ = plt.figure(figsize=(12,6))
    ax = plt.axes()
    for i,country in enumerate(countries):
        data = df[df[attribute] == country][aspect].to_numpy(dtype=float)
        ax.boxplot(data,positions=[i+1], notch=0, sym='+', vert=1, whis=1.5)
        # Change the color of the boxplot depending on whether 
    ax.set_title(f"Distribution of '{aspect}'")
    # set axes limits and labels
    ax.set_xlim(0.5, len(countries)+0.5)
    ticks = np.arange(1, len(countries)+1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(countries,rotation=90)
    ax.set_xlabel("Country")
    ax.set_ylabel(aspect)
    plt.tight_layout()
    plt.savefig(f"Plots/{filename}.png",bbox_inches='tight')
    plt.show()

# %%
# Get a list of countries we want to plot the aspects for
country_list = list(df_ba["user_country"].unique())
# Plot the distribution of the aspects for the countries in the list
for aspect in ba_aspects:
    plot_boxplot(df_ba, aspect, country_list,filename=f"boxplot_of_{aspect}_for_all_countries")
    # plot rescaled aspect
    aspect_rescaled = aspect + "_rescaled"
    plot_boxplot(df_ba, aspect_rescaled, country_list, filename=f"boxplot_of_{aspect_rescaled}_for_all_countries")


# %% [markdown]
#  ### T-Testing for different US states in BA
# %%
# Create a list of aspects for ba
ba_aspects = ["aroma", "appearance", "taste", "palate", "overall", "rating"] 
# Get a list of all states in the US
state_list = list(df_ba[df_ba["user_country"] == "United States"]["user_state"].unique())
# Perform t-tests for each pair of countries
dfs_ba_t_tests_states = {}
for aspect in ba_aspects:
    dfs_ba_t_tests_states[aspect] = pairwise_ttests(df_ba,state_list,aspect,entity_column_name="user_state")
    aspect_rescaled = aspect + "_rescaled"
    dfs_ba_t_tests_states[aspect_rescaled] = pairwise_ttests(df_ba,state_list,aspect_rescaled,entity_column_name="user_state")

# %%
dfs_ba_t_tests_states["aroma"][0].head()

# %%
# Compare the results for the different aspects
# How many significant results are there for each aspect?
df_ba_t_tests_states_results = {"aspect": [], "significant": [], "had_enough_data": [],"total": []}
for aspect in ba_aspects:
    df_ba_t_tests_states_results["aspect"].append(aspect)
    df_ba_t_tests_states_results["significant"].append(len(dfs_ba_t_tests_states[aspect][0][dfs_ba_t_tests_states[aspect][0]["significant"] == True]))
    df_ba_t_tests_states_results["had_enough_data"].append(len(dfs_ba_t_tests_states[aspect][0][dfs_ba_t_tests_states[aspect][0]["had_enough_data"] == True]))
    df_ba_t_tests_states_results["total"].append(len(dfs_ba_t_tests_states[aspect][0]))
    aspect_rescaled = aspect + "_rescaled"
    df_ba_t_tests_states_results["aspect"].append(aspect_rescaled)
    df_ba_t_tests_states_results["significant"].append(len(dfs_ba_t_tests_states[aspect_rescaled][0][dfs_ba_t_tests_states[aspect_rescaled][0]["significant"] == True]))
    df_ba_t_tests_states_results["had_enough_data"].append(len(dfs_ba_t_tests_states[aspect_rescaled][0][dfs_ba_t_tests_states[aspect_rescaled][0]["had_enough_data"] == True]))
    df_ba_t_tests_states_results["total"].append(len(dfs_ba_t_tests_states[aspect_rescaled][0]))
df_ba_t_tests_states_results = pd.DataFrame(df_ba_t_tests_states_results)
df_ba_t_tests_states_results

# %%
# Get the significance level
significance_level = dfs_ba_t_tests_states["aroma"][1]
# Print the significance level
print(f"Significance level: {significance_level:.5f}")


# %% [markdown]
#  ### T-Testing for different US states in RB
# Create a list of aspects for ba
# %%
rb_aspects = ["aroma", "appearance", "taste", "palate", "overall", "rating"] 
# Get a list of all states in the US
state_list = list(df_rb[df_rb["user_country"] == "United States"]["user_state"].unique())
# Perform t-tests for each pair of countries
dfs_rb_t_tests_states = {}
for aspect in ba_aspects:
    dfs_rb_t_tests_states[aspect] = pairwise_ttests(df_rb,state_list,aspect,entity_column_name="user_state",datasource="rb")
    aspect_rescaled = aspect + "_rescaled"
    dfs_rb_t_tests_states[aspect_rescaled] = pairwise_ttests(df_rb,state_list,aspect_rescaled,entity_column_name="user_state",datasource="rb")

# %%
print(dfs_rb_t_tests_states["aroma"][0].head(10))

# %%
# Compare the results for the different aspects
# How many significant results are there for each aspect?
df_rb_t_tests_states_results = {"aspect": [], "significant": [], "had_enough_data": [],"total": []}
for aspect in ba_aspects:
    df_rb_t_tests_states_results["aspect"].append(aspect)
    df_rb_t_tests_states_results["significant"].append(len(dfs_rb_t_tests_states[aspect][0][dfs_rb_t_tests_states[aspect][0]["significant"] == True]))
    df_rb_t_tests_states_results["had_enough_data"].append(len(dfs_rb_t_tests_states[aspect][0][dfs_rb_t_tests_states[aspect][0]["had_enough_data"] == True]))
    df_rb_t_tests_states_results["total"].append(len(dfs_rb_t_tests_states[aspect][0]))
    aspect_rescaled = aspect + "_rescaled"
    df_rb_t_tests_states_results["aspect"].append(aspect_rescaled)
    df_rb_t_tests_states_results["significant"].append(len(dfs_rb_t_tests_states[aspect_rescaled][0][dfs_rb_t_tests_states[aspect_rescaled][0]["significant"] == True]))
    df_rb_t_tests_states_results["had_enough_data"].append(len(dfs_rb_t_tests_states[aspect_rescaled][0][dfs_rb_t_tests_states[aspect_rescaled][0]["had_enough_data"] == True]))
    df_rb_t_tests_states_results["total"].append(len(dfs_rb_t_tests_states[aspect_rescaled][0]))
df_rb_t_tests_states_results = pd.DataFrame(df_rb_t_tests_states_results)
df_rb_t_tests_states_results

# %%
# Get the significance level
significance_level = dfs_rb_t_tests_states["aroma"][1]
# Print the significance level
print(f"Significance level: {significance_level:.5f}")

# %% [markdown]
# Compare the ratio of significant results to the total number of comparisons for each aspect in BA and RB
# %%
df_ba_t_tests_states_results["ratio"] = df_ba_t_tests_states_results["significant"] / df_ba_t_tests_states_results["had_enough_data"]
df_rb_t_tests_states_results["ratio"] = df_rb_t_tests_states_results["significant"] / df_rb_t_tests_states_results["had_enough_data"]
df_ba_t_tests_states_results["datasource"] = "ba"
df_rb_t_tests_states_results["datasource"] = "rb"
df_t_tests_states_results = pd.concat([df_ba_t_tests_states_results,df_rb_t_tests_states_results])
df_t_tests_states_results
# Group by aspect and datasource
df_t_tests_states_results_grouped = df_t_tests_states_results.groupby(["aspect","datasource"]).sum()
df_t_tests_states_results_grouped

# %% [markdown]
#  ### T-Testing for differences in rating averages per style_class for some state or country
# Take say Germany
# %%
# Subset to Germany
df_ba_germany = df_ba[df_ba["user_country"] == "Germany"]
# Get a list of all style_classes in Germany
style_class_list = list(df_ba_germany["style_class"].unique())
# Results dict
dfs_ba_t_tests_style_class_germany = {}
for aspect in ba_aspects:
    # Perform t-tests for each pair of style_classes
    dfs_ba_t_tests_style_class_germany[aspect] = pairwise_ttests(df_ba_germany,style_class_list,aspect,entity_column_name="style_class",min_sample_size=30)
    aspect_rescaled = aspect + "_rescaled"
    dfs_ba_t_tests_style_class_germany[aspect_rescaled] = pairwise_ttests(df_ba_germany,style_class_list,aspect_rescaled,entity_column_name="style_class",min_sample_size=30)

# %%
print(dfs_ba_t_tests_style_class_germany["aroma"][0].head(10))

# %%
# Compare the results for the different aspects
# How many significant results are there for each aspect?
# Print total number of tests performed
results = {"aspect": [], "aspect_rescaled": [], "significant": [], "significant_rescaled": [], "not_enough_data": [], "not_enough_data_rescaled": [], "total": [], "total_rescaled": []}
for aspect in ba_aspects:
    results["aspect"].append(aspect)
    results["aspect_rescaled"].append(aspect + "_rescaled")
    results["significant"].append(len(dfs_ba_t_tests_style_class_germany[aspect][0][dfs_ba_t_tests_style_class_germany[aspect][0]["significant"] == True]))
    results["significant_rescaled"].append(len(dfs_ba_t_tests_style_class_germany[aspect + "_rescaled"][0][dfs_ba_t_tests_style_class_germany[aspect + "_rescaled"][0]["significant"] == True]))
    results["not_enough_data"].append(len(dfs_ba_t_tests_style_class_germany[aspect][0][dfs_ba_t_tests_style_class_germany[aspect][0]["significant"] == False]))
    results["not_enough_data_rescaled"].append(len(dfs_ba_t_tests_style_class_germany[aspect + "_rescaled"][0][dfs_ba_t_tests_style_class_germany[aspect + "_rescaled"][0]["significant"] == False]))
    results["total"].append(len(dfs_ba_t_tests_style_class_germany[aspect][0]))
    results["total_rescaled"].append(len(dfs_ba_t_tests_style_class_germany[aspect + "_rescaled"][0]))
df_results_t_tests_style_class_germany = pd.DataFrame(results)

# %%
df_results_t_tests_style_class_germany

# %%
# Get the significance level
significance_level = dfs_ba_t_tests_style_class_germany["aroma"][1]
# Print the significance level
print(f"Significance level: {significance_level:.5f}")

# %% [markdown]
#  ### T-Testing for differences in rating averages per style for some state or country
# Take say Germany
# %%
# style list
style_list = list(df_ba_germany["style"].unique())
# Results dict
dfs_ba_t_tests_style_germany = {}
for aspect in ba_aspects:
    # Perform t-tests for each pair of style_classes
    dfs_ba_t_tests_style_germany[aspect] = pairwise_ttests(df_ba_germany,style_list,aspect,entity_column_name="style", min_sample_size=30)
    aspect_rescaled = aspect + "_rescaled"
    dfs_ba_t_tests_style_germany[aspect_rescaled] = pairwise_ttests(df_ba_germany,style_list,aspect_rescaled,entity_column_name="style", min_sample_size=30)


# %%
print(dfs_ba_t_tests_style_germany["aroma"][0].head(10))

# %%
# Compare the results for the different aspects
# How many significant results are there for each aspect?
results = {"aspect": [], "aspect_rescaled": [], "significant": [], "significant_rescaled": [], "not_enough_data": [], "not_enough_data_rescaled": [], "total": [], "total_rescaled": []}
for aspect in ba_aspects:
    results["aspect"].append(aspect)
    results["aspect_rescaled"].append(aspect + "_rescaled")
    results["significant"].append(len(dfs_ba_t_tests_style_germany[aspect][0][dfs_ba_t_tests_style_germany[aspect][0]["significant"] == True]))
    results["significant_rescaled"].append(len(dfs_ba_t_tests_style_germany[aspect + "_rescaled"][0][dfs_ba_t_tests_style_germany[aspect + "_rescaled"][0]["significant"] == True]))
    results["not_enough_data"].append(len(dfs_ba_t_tests_style_germany[aspect][0][dfs_ba_t_tests_style_germany[aspect][0]["had_enough_data"] == False]))
    results["not_enough_data_rescaled"].append(len(dfs_ba_t_tests_style_germany[aspect + "_rescaled"][0][dfs_ba_t_tests_style_germany[aspect + "_rescaled"][0]["had_enough_data"] == False]))
    results["total"].append(len(dfs_ba_t_tests_style_germany[aspect][0]))
    results["total_rescaled"].append(len(dfs_ba_t_tests_style_germany[aspect + "_rescaled"][0]))
df_results_t_tests_style_germany = pd.DataFrame(results)


# %%
df_results_t_tests_style_germany

# %%
# Get the significance level
significance_level = dfs_ba_t_tests_style_germany["aroma"][1]
# Print the significance level
print(f"Significance level: {significance_level:.5f}")

# %%
# box plot the most popular style classes in germany
# Get the most popular style classes
style_counts = df_ba_germany["style"].value_counts().head(10)
style_counts = style_counts.index.tolist()

# %%
plot_boxplot(df_ba_germany,"rating",style_counts,attribute="style",filename="boxplot_style_germany.png")


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
dfs_ba_t_tests_states["rating"][0].head()

# %%
df_t_test_to_plot = dfs_ba_t_tests_states["rating"][0][["user_state1", "user_state2", "p-value"]]

df_t_test_to_plot = df_t_test_to_plot[~(df_t_test_to_plot["user_state1"]=='United States')]

df_t_test_to_plot["state_code_user_state1"] = df_t_test_to_plot["user_state1"].apply(lambda x: get_key(x))
df_t_test_to_plot["state_code_user_state2"] = df_t_test_to_plot["user_state2"].apply(lambda x: get_key(x))

df_t_test_to_plot.drop("user_state1", axis=1, inplace=True)
df_t_test_to_plot.drop("user_state2", axis=1, inplace=True)


#%%

df_plot_p_value = pd.DataFrame()
for key in states_codes.keys():
    c = df_t_test_to_plot[df_t_test_to_plot["state_code_user_state1"].isin([key]) | df_t_test_to_plot["state_code_user_state2"].isin([key])]
    c["state_code_user_state1"] = np.where(c["state_code_user_state1"] == key, c["state_code_user_state2"], c["state_code_user_state1"])
    c["state_code_user_state2"] = states_codes.get(key)
    c.rename(columns={"state_code_user_state2": "State"}, inplace=True)
    c = c.append({'p-value': 0, "state_code_user_state1": key, "year": states_codes.get(key)}, ignore_index = True)
    c = c.sort_values(by=["state_code_user_state1"])
    frames = [df_plot_p_value, c]
    df_plot_p_value = pd.concat(frames)
    
# %%
fig = px.choropleth(df_plot_p_value,
                    locations='state_code_user_state1', 
                    locationmode="USA-states", 
                    color='p-value',
                    color_continuous_scale="Viridis_r", 
                    scope="usa",
                    title="Independant T-test between 2 US States",
                    animation_frame='State') #make sure 'period_begin' is string type and sorted in ascending order



fig.show()
print("If the p-value is smaller than our threshold, then we have evidence against the null hypothesis of equal population means.")




# %% [markdown]
# # Subsetting the data to specific style classes and styles
# Does the number of significant results change when we subset the data to specific style classes and styles?
# Subset the data to specific style classes and styles
# First focus on style classes
# Figure out how many ratings we have per style class
# %%
# How many ratings do we have per style class?
# In BA
df_ba["style_class"].value_counts().plot(kind="bar")
plt.title("Number of ratings per style class in BA")
plt.xlabel("Style class")
plt.ylabel("Number of ratings")
plt.show()

# %%
# Create a list with the top 5 style classes
top_5_style_classes_ba = df_ba["style_class"].value_counts().head(5).index.tolist()
top_5_style_classes_ba

# %%
def perform_t_tests_per_class(class_list, df, filter_name, df_name, entity_name, recompute = False):
    """
    Perform t-tests for each style class in the list
    arguments:
        class_list: list of style classes to perform t-tests on
        df: dataframe to perform t-tests on
        filter_name: name of the column to filter on
        df_name: name of the dataframe
        entity_name: name of the column to use as entity
        recompute: whether to recompute the t-tests or not
    returns:
        df_t_tests_style_classes: dataframe with the results of the t-tests
    """
    dfs_t_tests_style_classes = {} # Create a dictionary to store the results
    for style_class in class_list: # Loop over the style classes
        df_style_class = df[df[filter_name] == style_class] # Subset the data to the style class
        # Get a country list
        entity_list = df_style_class[entity_name].unique().tolist()
        if len(entity_list) < 2:
            continue
        # Create a dictionary to store the results
        dfs_t_tests_style_classes[style_class] = {}
        for aspect in ba_aspects:
            # Run the t-test for the aspect and the rescaled aspect
            dfs_t_tests_style_classes[style_class][aspect] = pairwise_ttests(df_style_class,entity_list,aspect,entity_column_name=entity_name,datasource=df_name, recompute=recompute,min_sample_size=MIN_TEST_SAMPLE_SIZE)
            aspect_rescaled = aspect + "_rescaled"
            dfs_t_tests_style_classes[style_class][aspect_rescaled] = pairwise_ttests(df_style_class,entity_list,aspect_rescaled,entity_column_name=entity_name, datasource=df_name, recompute=recompute,min_sample_size=MIN_TEST_SAMPLE_SIZE)
        
    # convert results to a dataframe
    results_df_t_tests_style_classes = {filter_name:[], "aspect":[], "significant_results":[],"significant_results_rescaled":[],"had_enough_data":[],"had_enough_data_rescaled":[],"total_results":[],"total_results_rescaled":[]}
    for style_class in dfs_t_tests_style_classes.keys(): # Loop over the style classes
        for aspect in ba_aspects: # Loop over the aspects
            results_df_t_tests_style_classes[filter_name].append(style_class)
            results_df_t_tests_style_classes["aspect"].append(aspect)
            aspect_rescaled = aspect + "_rescaled"
            results_df_t_tests_style_classes["significant_results"].append(len(dfs_t_tests_style_classes[style_class][aspect][0][dfs_t_tests_style_classes[style_class][aspect][0]["significant"] == True]))
            results_df_t_tests_style_classes["significant_results_rescaled"].append(len(dfs_t_tests_style_classes[style_class][aspect_rescaled][0][dfs_t_tests_style_classes[style_class][aspect_rescaled][0]["significant"] == True]))
            results_df_t_tests_style_classes["had_enough_data"].append(len(dfs_t_tests_style_classes[style_class][aspect][0][dfs_t_tests_style_classes[style_class][aspect][0]["had_enough_data"] == True]))
            results_df_t_tests_style_classes["had_enough_data_rescaled"].append(len(dfs_t_tests_style_classes[style_class][aspect_rescaled][0][dfs_t_tests_style_classes[style_class][aspect_rescaled][0]["had_enough_data"] == True]))
            results_df_t_tests_style_classes["total_results"].append(len(dfs_t_tests_style_classes[style_class][aspect][0]))
            results_df_t_tests_style_classes["total_results_rescaled"].append(len(dfs_t_tests_style_classes[style_class][aspect_rescaled][0]))

    df_t_tests_style_classes = pd.DataFrame(results_df_t_tests_style_classes)
    return df_t_tests_style_classes

# %%
# For each style_class, subset the data and run the t-test for each aspect
# Create a dictionary to store the results
df_ba_t_tests_style_classes_countries = perform_t_tests_per_class(top_5_style_classes_ba, df_ba, "style_class", "BA", "user_country")
df_ba_t_tests_style_classes_countries

# %%
# Compute the average ratio of significant results per style class
df_ba_t_tests_style_classes_countries["ratio_significant_results"] = df_ba_t_tests_style_classes_countries["significant_results"] / df_ba_t_tests_style_classes_countries["had_enough_data"]
df_ba_t_tests_style_classes_countries["ratio_significant_results_rescaled"] = df_ba_t_tests_style_classes_countries["significant_results_rescaled"] / df_ba_t_tests_style_classes_countries["had_enough_data_rescaled"]
df_ba_t_tests_style_classes_countries
# Average the ratios
print("Proportion of significant results ",df_ba_t_tests_style_classes_countries["ratio_significant_results"].mean())
print("Proportion of significant results rescaled",df_ba_t_tests_style_classes_countries["ratio_significant_results_rescaled"].mean())


# %% [markdown]
# We can still observe a relatively high number of significant results accross countries when we subset to the top 5 style classes.

# %%
df_ba_t_tests_style_classes_states = perform_t_tests_per_class(top_5_style_classes_ba, df_ba, "style_class", "BA", "user_state")
df_ba_t_tests_style_classes_states

# %%
# Compute the average ratio of significant results per style class
df_ba_t_tests_style_classes_states["ratio_significant_results"] = df_ba_t_tests_style_classes_states["significant_results"] / df_ba_t_tests_style_classes_states["had_enough_data"]
df_ba_t_tests_style_classes_states["ratio_significant_results_rescaled"] = df_ba_t_tests_style_classes_states["significant_results_rescaled"] / df_ba_t_tests_style_classes_states["had_enough_data_rescaled"]
df_ba_t_tests_style_classes_states
# Average the ratios
print("Proportion of significant results ",df_ba_t_tests_style_classes_states["ratio_significant_results"].mean())
print("Proportion of significant results rescaled",df_ba_t_tests_style_classes_states["ratio_significant_results_rescaled"].mean())

# %%
# In RB
df_rb["style_class"].value_counts().plot(kind="bar")
plt.title("Number of ratings per style class in RB")
plt.xlabel("Style class")
plt.ylabel("Number of ratings")
plt.show()

# %%
# Create a list with the top 5 style classes
top_5_style_classes_rb = df_rb["style_class"].value_counts().head(5).index.tolist()
top_5_style_classes_rb

# %%
df_rb_t_tests_style_classes_states = perform_t_tests_per_class(top_5_style_classes_rb, df_rb, "style_class", "RB", "user_state")
df_rb_t_tests_style_classes_states

# %%
# Compute the average ratio of significant results per style class
df_rb_t_tests_style_classes_states["ratio_significant_results"] = df_rb_t_tests_style_classes_states["significant_results"] / df_rb_t_tests_style_classes_states["had_enough_data"]
df_rb_t_tests_style_classes_states["ratio_significant_results_rescaled"] = df_rb_t_tests_style_classes_states["significant_results_rescaled"] / df_rb_t_tests_style_classes_states["had_enough_data_rescaled"]
# Average the ratios
print("Proportion of significant results ",df_rb_t_tests_style_classes_states["ratio_significant_results"].mean())
print("Proportion of significant results rescaled",df_rb_t_tests_style_classes_states["ratio_significant_results_rescaled"].mean())

# %%
# How many ratings do we have per style?
# In BA, plot the top 20 styles
df_ba["style"].value_counts().head(20).plot(kind="bar")
plt.title("Number of ratings per style in BA - top 20")
plt.xlabel("Style")
plt.ylabel("Number of ratings")
plt.show()

# %%
# Create a list with the top 5 styles  
top_5_styles_ba = df_ba["style"].value_counts().head(5).index.tolist()
top_5_styles_ba


# %%
df_ba_t_tests_style_countries = perform_t_tests_per_class(top_5_styles_ba, df_ba, "style", "BA", "user_country")
df_ba_t_tests_style_countries

# %%
# Compute the average ratio of significant results per style 
df_ba_t_tests_style_countries["ratio_significant_results"] = df_ba_t_tests_style_countries["significant_results"] / df_ba_t_tests_style_countries["had_enough_data"]
df_ba_t_tests_style_countries["ratio_significant_results_rescaled"] = df_ba_t_tests_style_countries["significant_results_rescaled"] / df_ba_t_tests_style_countries["had_enough_data_rescaled"]
# Average the ratios
print("Proportion of significant results ",df_ba_t_tests_style_countries["ratio_significant_results"].mean())
print("Proportion of significant results rescaled",df_ba_t_tests_style_countries["ratio_significant_results_rescaled"].mean())

# %%
df_ba_t_tests_style_states = perform_t_tests_per_class(top_5_styles_ba, df_ba, "style", "BA", "user_state")
df_ba_t_tests_style_states

# %%
# Compute the average ratio of significant results per style 
df_ba_t_tests_style_states["ratio_significant_results"] = df_ba_t_tests_style_states["significant_results"] / df_ba_t_tests_style_states["had_enough_data"]
df_ba_t_tests_style_states["ratio_significant_results_rescaled"] = df_ba_t_tests_style_states["significant_results_rescaled"] / df_ba_t_tests_style_states["had_enough_data_rescaled"]
# Average the ratios
print("Proportion of significant results ",df_ba_t_tests_style_states["ratio_significant_results"].mean())
print("Proportion of significant results rescaled",df_ba_t_tests_style_states["ratio_significant_results_rescaled"].mean())





# %%
# In RB, plot the top 20 styles
df_rb["style"].value_counts().head(20).plot(kind="bar")
plt.title("Number of ratings per style in RB - top 20")
plt.xlabel("Style")
plt.ylabel("Number of ratings")
plt.show()

# %%
# Create a list with the top 5 styles
top_5_styles_rb = df_rb["style"].value_counts().head(5).index.tolist()
top_5_styles_rb

# %%
df_rb_t_tests_style_states = perform_t_tests_per_class(top_5_styles_rb, df_rb, "style", "RB", "user_state")
df_rb_t_tests_style_states

# %%
# Compute the average ratio of significant results per style
df_rb_t_tests_style_states["ratio_significant_results"] = df_rb_t_tests_style_states["significant_results"] / df_rb_t_tests_style_states["had_enough_data"]
df_rb_t_tests_style_states["ratio_significant_results_rescaled"] = df_rb_t_tests_style_states["significant_results_rescaled"] / df_rb_t_tests_style_states["had_enough_data_rescaled"]
# Average the ratios
print("Proportion of significant results ",df_rb_t_tests_style_states["ratio_significant_results"].mean())
print("Proportion of significant results rescaled",df_rb_t_tests_style_states["ratio_significant_results_rescaled"].mean())

# %% [markdown]
# TODO: discuss results and motivate the following analysis
# Subsetting to most popular style or style_class does not make much difference..

# %%
# Get a list of countries/states that have significant results for a given style
aspect_rescaled = "rating_rescaled"
dfs_t_tests_countries[aspect_rescaled][0]
# Keep only the significant results
dfs_t_tests_countries[aspect_rescaled][0][dfs_t_tests_countries[aspect_rescaled][0]["significant"] == True]


# %%
aspect_rescaled = "aroma_rescaled"
dfs_t_tests_countries[aspect_rescaled][0]
# Keep only the significant results
dfs_t_tests_countries[aspect_rescaled][0][dfs_t_tests_countries[aspect_rescaled][0]["significant"] == True]

# %%
aspect_rescaled = "palate_rescaled"
dfs_t_tests_countries[aspect_rescaled][0]
# Keep only the significant results
dfs_t_tests_countries[aspect_rescaled][0][dfs_t_tests_countries[aspect_rescaled][0]["significant"] == True]

# %%
aspect_rescaled = "appearance_rescaled"
dfs_t_tests_countries[aspect_rescaled][0]
# Keep only the significant results
dfs_t_tests_countries[aspect_rescaled][0][dfs_t_tests_countries[aspect_rescaled][0]["significant"] == True]

# %%
aspect_rescaled = "taste_rescaled"
dfs_t_tests_countries[aspect_rescaled][0]
# Keep only the significant results
dfs_t_tests_countries[aspect_rescaled][0][dfs_t_tests_countries[aspect_rescaled][0]["significant"] == True]

# %%
aspect_rescaled = "overall_rescaled"
dfs_t_tests_countries[aspect_rescaled][0]
# Keep only the significant results
dfs_t_tests_countries[aspect_rescaled][0][dfs_t_tests_countries[aspect_rescaled][0]["significant"] == True]


# %%
print(ba_aspects)

# %% [markdown]
#  ### Evaluate the significant results

# %%
# For Each of the significant results. Are the beer averages significantly different?
# Get a list of all beers rated by the 


# %%
significant_country_pairs = {"countryA":[],"countryB":[],"aspect":[]}
for aspect in ba_aspects:
    aspect_rescaled = aspect + "_rescaled"
    significant = dfs_t_tests_countries[aspect_rescaled][0][dfs_t_tests_countries[aspect_rescaled][0]["significant"] == True]
    for index, row in significant.iterrows():
        significant_country_pairs["countryA"].append(row["user_country1"])
        significant_country_pairs["countryB"].append(row["user_country2"])
        significant_country_pairs["aspect"].append(aspect)


# %%
# Create a dataframe with the significant country pairs
df_significant_country_pairs = pd.DataFrame(significant_country_pairs)
df_significant_country_pairs

# %%
# Get unique country names
unique_countries = df_significant_country_pairs["countryA"].unique().tolist()
unique_countries.extend(df_significant_country_pairs["countryB"].unique().tolist())
unique_countries = list(set(unique_countries))
unique_countries

# %%
# For each of the unique countries, get the plot the number of ratings of the top n beers
top_n = 20
for country in unique_countries:
    df_ba[df_ba["user_country"] == country]["beer_name"].value_counts().head(top_n).plot(kind="bar")
    plt.title("Number of ratings per beer in BA - top " + str(top_n) + " for " + country)
    plt.xlabel("Beer")
    plt.ylabel("Number of ratings")
    plt.show()
# %%
# For each of the unique countries, get all ratings on the top n beers and store them in a dictionary
top_n = 100
dict_top_n_beers = {}
for country in unique_countries:
    # Get the top n beers
    top_n_beers = df_ba[df_ba["user_country"] == country]["beer_name"].value_counts().head(top_n).index.tolist()
    # Get all ratings for the top n beers
    top_n_beers_ratings = df_ba[df_ba["beer_name"].isin(top_n_beers)]
    # Store the ratings in a dictionary
    dict_top_n_beers[country] = top_n_beers_ratings
dict_top_n_beers

# %%
# Get unique beer rating in the dict_top_n_beers
unique_beers = []
for country in unique_countries:
    unique_beers.extend(dict_top_n_beers[country]["beer_name"].unique().tolist())
unique_beers = list(set(unique_beers))
len(unique_beers)

# %%
# Get the number of unique beers in df_ba
len(df_ba["beer_name"].unique().tolist())

# %%
# For each significant country pair, t_test the average ratings of the top n beers
# Create a dictionary to store the results
dict_t_tests_top_n_beers = {}
# compute the sidak correction
sidak_correction = 1 - (1 - 0.05)**(1/len(df_significant_country_pairs))
print("sidak_correction: " + str(sidak_correction))
for index, row in df_significant_country_pairs.iterrows():
    # Get the ratings for the two countries
    ratings_countryA = dict_top_n_beers[row["countryA"]]
    ratings_countryB = dict_top_n_beers[row["countryB"]]
    # Perform a t-test
    t_test = stats.ttest_ind(ratings_countryA[row["aspect"]+"_rescaled"], ratings_countryB[row["aspect"]+"_rescaled"])
    # Check if the t-test is significant
    significant = False
    if t_test[1] < sidak_correction:
        significant = True
    p_value = t_test[1]
    t_statistic = t_test[0]
    # Store the results in a dictionary
    dict_t_tests_top_n_beers[index] = {"countryA":row["countryA"], "countryB":row["countryB"], "aspect":row["aspect"], "t_test_statistic":t_statistic, "p_value":p_value, "significant":significant}
dict_t_tests_top_n_beers
# %%
# Create a dataframe with the results
df_t_tests_top_n_beers = pd.DataFrame(dict_t_tests_top_n_beers).T
df_t_tests_top_n_beers

# %%
# Get the ratio of significant results
df_t_tests_top_n_beers["significant"].value_counts()


# %% [markdown]
#  ### States BA

# %%
dfs_ba_t_tests_states

# %%
significant_state_pairs_BA = {"stateA":[],"stateB":[],"aspect":[]}
for aspect in ba_aspects:
    aspect_rescaled = aspect + "_rescaled"
    significant = dfs_ba_t_tests_states[aspect_rescaled][0][dfs_ba_t_tests_states[aspect_rescaled][0]["significant"] == True]
    for index, row in significant.iterrows():
        significant_state_pairs_BA["stateA"].append(row["user_state1"])
        significant_state_pairs_BA["stateB"].append(row["user_state2"])
        significant_state_pairs_BA["aspect"].append(aspect)


# %%
# Create a dataframe with the significant country pairs
df_significant_state_pairs_ba = pd.DataFrame(significant_state_pairs_BA)
df_significant_state_pairs_ba

# %%
# Get unique country names
unique_states = df_significant_state_pairs_ba["stateA"].unique().tolist()
unique_states.extend(df_significant_state_pairs_ba["stateB"].unique().tolist())
unique_states = list(set(unique_states))
unique_states

# %%
# For each of the unique countries, get the plot the number of ratings of the top n beers
top_n = 20
for state in unique_states[:5]:
    df_ba[df_ba["user_state"] == state]["beer_name"].value_counts().head(top_n).plot(kind="bar")
    plt.title("Number of ratings per beer in BA - top " + str(top_n) + " for " + state)
    plt.xlabel("Beer")
    plt.ylabel("Number of ratings")
    plt.show()
# %%
# For each of the unique countries, get all ratings on the top n beers and store them in a dictionary
top_n = 100
dict_top_n_beers = {}
for state in unique_states:
    # Get the top n beers
    top_n_beers = df_ba[df_ba["user_state"] == state]["beer_name"].value_counts().head(top_n).index.tolist()
    # Get all ratings for the top n beers
    top_n_beers_ratings = df_ba[df_ba["beer_name"].isin(top_n_beers)]
    # Store the ratings in a dictionary
    dict_top_n_beers[state] = top_n_beers_ratings
dict_top_n_beers

# %%
# For each significant country pair, t_test the average ratings of the top n beers
# Create a dictionary to store the results
dict_t_tests_top_n_beers = {}
# compute the sidak correction
sidak_correction = 1 - (1 - 0.05)**(1/len(df_significant_state_pairs_ba))
print("sidak_correction: " + str(sidak_correction))
for index, row in df_significant_state_pairs_ba.iterrows():
    # Get the ratings for the two countries
    ratings_countryA = dict_top_n_beers[row["stateA"]]
    ratings_countryB = dict_top_n_beers[row["stateB"]]
    # Perform a t-test
    t_test = stats.ttest_ind(ratings_countryA[row["aspect"]+"_rescaled"], ratings_countryB[row["aspect"]+"_rescaled"])
    # Check if the t-test is significant
    significant = False
    if t_test[1] < sidak_correction:
        significant = True
    p_value = t_test[1]
    t_statistic = t_test[0]
    # Store the results in a dictionary
    dict_t_tests_top_n_beers[index] = {"stateA":row["stateA"], "stateB":row["stateB"], "aspect":row["aspect"], "t_test_statistic":t_statistic, "p_value":p_value, "significant":significant}
dict_t_tests_top_n_beers
# %%
# Create a dataframe with the results
df_t_tests_top_n_beers_ba = pd.DataFrame(dict_t_tests_top_n_beers).T
df_t_tests_top_n_beers_ba

# %%
# Get the ratio of significant results
df_t_tests_top_n_beers_ba["significant"].value_counts()



# %% [markdown]
#  ### States RB

# %%
# TODO: repeat the same analysis per state...
dfs_rb_t_tests_states

# %%
significant_state_pairs_RB = {"stateA":[],"stateB":[],"aspect":[]}
for aspect in ba_aspects:
    aspect_rescaled = aspect + "_rescaled"
    significant = dfs_rb_t_tests_states[aspect_rescaled][0][dfs_rb_t_tests_states[aspect_rescaled][0]["significant"] == True]
    for index, row in significant.iterrows():
        significant_state_pairs_RB["stateA"].append(row["user_state1"])
        significant_state_pairs_RB["stateB"].append(row["user_state2"])
        significant_state_pairs_RB["aspect"].append(aspect)


# %%
# Create a dataframe with the significant country pairs
df_significant_state_pairs_rb = pd.DataFrame(significant_state_pairs_RB)
df_significant_state_pairs_rb

# %%
# Get unique country names
unique_states = df_significant_state_pairs_rb["stateA"].unique().tolist()
unique_states.extend(df_significant_state_pairs_rb["stateB"].unique().tolist())
unique_states = list(set(unique_states))
unique_states

# %%
# For each of the unique countries, get the plot the number of ratings of the top n beers
top_n = 20
for state in unique_states[:5]:
    df_rb[df_rb["user_state"] == state]["beer_name"].value_counts().head(top_n).plot(kind="bar")
    plt.title("Number of ratings per beer in BA - top " + str(top_n) + " for " + state)
    plt.xlabel("Beer")
    plt.ylabel("Number of ratings")
    plt.show()
# %%
# For each of the unique countries, get all ratings on the top n beers and store them in a dictionary
top_n = 100
dict_top_n_beers = {}
for state in unique_states:
    # Get the top n beers
    top_n_beers = df_rb[df_rb["user_state"] == state]["beer_name"].value_counts().head(top_n).index.tolist()
    # Get all ratings for the top n beers
    top_n_beers_ratings = df_rb[df_rb["beer_name"].isin(top_n_beers)]
    # Store the ratings in a dictionary
    dict_top_n_beers[state] = top_n_beers_ratings
dict_top_n_beers

# %%
# For each significant country pair, t_test the average ratings of the top n beers
# Create a dictionary to store the results
dict_t_tests_top_n_beers = {}
# compute the sidak correction
sidak_correction = 1 - (1 - 0.05)**(1/len(df_significant_state_pairs_rb))
print("sidak_correction: " + str(sidak_correction))
for index, row in df_significant_state_pairs_rb.iterrows():
    # Get the ratings for the two countries
    ratings_countryA = dict_top_n_beers[row["stateA"]]
    ratings_countryB = dict_top_n_beers[row["stateB"]]
    # Perform a t-test
    t_test = stats.ttest_ind(ratings_countryA[row["aspect"]+"_rescaled"], ratings_countryB[row["aspect"]+"_rescaled"])
    # Check if the t-test is significant
    significant = False
    if t_test[1] < sidak_correction:
        significant = True
    p_value = t_test[1]
    t_statistic = t_test[0]
    # Store the results in a dictionary
    dict_t_tests_top_n_beers[index] = {"stateA":row["stateA"], "stateB":row["stateB"], "aspect":row["aspect"], "t_test_statistic":t_statistic, "p_value":p_value, "significant":significant}
dict_t_tests_top_n_beers
# %%
# Create a dataframe with the results
df_t_tests_top_n_beers_rb = pd.DataFrame(dict_t_tests_top_n_beers).T
df_t_tests_top_n_beers_rb

# %%
# Get the ratio of significant results
df_t_tests_top_n_beers_rb["significant"].value_counts()


# %%
