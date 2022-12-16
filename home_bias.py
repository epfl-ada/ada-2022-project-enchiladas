# %%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
# %% [markdown]
# #RQ4: home bias
# %%
# load the data
df = read_pickle(data_folder + "df_ba_ratings_filtered_beers_merged_users.pkl")
# define the treatment variable
df["treatment"] = df.apply(lambda row: 1 if row["user_country"] == row["beer_country"] else 0, axis=1)
# %% [markdown]
# ## data exploration for RQ4
# %%
# feature number of reviews for each beer
df_groupby_beer = df.groupby(by = "beer_id").agg({"beer_id": "count", "treatment": "sum"})
df["nb_reviews_per_beer"] = df.apply(lambda row: df_groupby_beer.loc[row["beer_id"]]["beer_id"], axis=1)
df["nb_reviews_per_beer_local"] = df.apply(lambda row: df_groupby_beer.loc[row["beer_id"]]["treatment"], axis=1)
df["nb_reviews_per_beer_foreign"] = df["nb_reviews_per_beer"] - df["nb_reviews_per_beer_local"]
df["share_local_reviews"] = df["nb_reviews_per_beer_local"] / df["nb_reviews_per_beer"]

# %%
# plot the proportion of local reviews for each beer
groupby_beer = df.groupby(by="beer_id").agg({"share_local_reviews": "mean"})
groupby_beer["share_local_reviews"].hist(bins=50, label = "all reviews")

# removing american reviews
groupby_beer_no_us = df[df["user_country"] != "United States"].groupby(by="beer_id").agg({"share_local_reviews": "mean"})
groupby_beer_no_us["share_local_reviews"].hist(bins=50, label = "non american reviews")
plt.title("share of local reviews per beer")
plt.legend()
# %% [markdown]
# Comment on the figure above: It makes sense. most of the users are concentrated in the us. Therefore a beer from the us is mostly rated by locals and an beer with few user in this country mostly receives foreign reviews.
# %%
# how many beers do have a balance distribution ? (10-90%)
groupby_beer_no_us[(groupby_beer_no_us["share_local_reviews"] > 0.1) & (groupby_beer_no_us["share_local_reviews"] < 0.9)]["share_local_reviews"].hist()
# %% [markdown]
# # analysis without any matching
# %%
# plot the rating distribution of the treatment and control groups prior to matching
df[df["treatment"] == 1]["rating"].hist(bins=20, alpha=0.5, label="local reviews")
df[df["treatment"] == 0]["rating"].hist(bins=20, alpha=0.5, label="foreign reviews")
plt.legend()
plt.title("distribution of local vs. foreign reviews")

# run a t-test to see if there is a significant difference in ratings between reviews with treatment = 1 and reviews with treatment = 0 (prior to matching)
from scipy.stats import ttest_ind
res = ttest_ind(df[df["treatment"] == 1]["rating"], df[df["treatment"] == 0]["rating"], equal_var=False)
print(res)
# print the average difference of mean ratings between treatment = 1 and treatment = 0
print("average difference of mean ratings between treatment = 1 and treatment = 0: ", df[df["treatment"] == 1]["rating"].mean() - df[df["treatment"] == 0]["rating"].mean())

# now if we subset to beer with at least 10% of foreign reviews and 10% of local reviews
df_balanced = df[(df["share_local_reviews"] > 0.1) & (df["share_local_reviews"] < 0.9)]
df_balanced[df_balanced["treatment"] == 1]["rating"].hist(bins=20, alpha=0.5, label="local reviews")
df_balanced[df_balanced["treatment"] == 0]["rating"].hist(bins=20, alpha=0.5, label="foreign reviews")

# run t-test
res = ttest_ind(df_balanced[df_balanced["treatment"] == 1]["rating"], df_balanced[df_balanced["treatment"] == 0]["rating"], equal_var=False)
print(res)
# print the average difference of mean ratings between treatment = 1 and treatment = 0
print("average difference of mean ratings between treatment = 1 and treatment = 0: ", df_balanced[df_balanced["treatment"] == 1]["rating"].mean() - df_balanced[df_balanced["treatment"] == 0]["rating"].mean())
# %% [markdown]
# comments on the analysis: The result seem to show some significance in the bias. However, as seen in the previous, the huge imbalances in the dataset lead to a lot of beer with either only local reviews or only foreign reviews. Therefore, we need to balance the dataset before running the analysis. We will use the propensity score matching method to balance the dataset.
# %% [markdown]
# ## propensity score matching
# %% [markdown]
# ### propensity score calculcation using random forest
# %%
# add feature of average ratings given by the user
df_users = df.groupby(by = "user_id").agg({"rating": "mean"})
df["avg_user_rating"] = df.apply(lambda row: df_users.loc[row["user_id"]]["rating"], axis=1)

# categories the style class
df["style_class_cat"] = df["style_class"].astype("category").cat.codes
# %% [markdown]
# We create a feature vector using the following features:
# - avg user rating
# - number of reviews
# - beer style class
# - beer average rating

# we cannot use user country and beer country beause they are already part of the treatment

# %%
# feature list
feature_list = ["avg_user_rating", "nbr_reviews", "style_class_cat", "avg_beer_rating"]

# why do we now have nans?
df = df.dropna(subset = "avg_beer_rating")

X = df[feature_list].values
# create label vector (treatment column)
y = df["treatment"].values
# %%
# split into train and test set
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

def random_forest_propensity(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train a random forest classifier

    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    # get the probabilities on the test set
    y_pred = clf.predict_proba(X_test)[:,1]
    # predict the treatment with a threshold of 0.5
    y_pred_treated = clf.predict(X_test)

    # measure f1 score
    print("f1 score: ", f1_score(y_test, y_pred_treated))
    # print confusion matrix
    print(confusion_matrix(y_test, y_pred_treated))

    return y_pred

y_pred = random_forest_propensity(X, y)
# %% [markdown]
# comment on the results: Eventhough the f1 score looks high, the predictor is still shit. The predictor is very much biased toward predicting treatment = 1 because the dataset is very unbalanced.
# To try to better assess the quality of the mdel, we filter the review dataset to only keep countries with at least 1000 reviews (top 8 countries). We then balance our dataset by randomly sampling 1000 reviews per country.
# %%
g = df.groupby("user_country").count().sort_values(by="beer_id", ascending=False)
# filter to countries in g with at least 1000 reviews
countries = g[g["beer_id"] >= 1000].index
df_topcountries = df[df["user_country"].isin(countries)] # 8 countries
g = df_topcountries.groupby("user_country")
def sampling_k_elements(group, k=1000):
    if len(group) < k:
        return group
    return group.sample(k)

df_topcountries_balanced = df_topcountries.groupby('user_country').apply(sampling_k_elements).reset_index(drop=True)
# check if we have 1000 reviews per country
df_topcountries_balanced.groupby("user_country").count()

# %%
# check if random forest performance
X = df_topcountries_balanced[["avg_user_rating", "nbr_reviews", "style_class_cat", "avg_beer_rating"]].values
# create label vector (treatment column)
y = df_topcountries_balanced["treatment"].values
random_forest_propensity(X, y)
# random forest is still pretty shit
# %% [markdown]
# ### propensity matching using matrix factorization with biases
# %% [markdown]
# The motivation of the method is the following: some beers are arguably of better quality than others. Similarly some user are more critical in their ratings than others. We can retrieve those bias (for the users and the beers) using matrix factorization with biases. We then use those biases to match pairs of users-beers (aka review) between the treatment and control groups. The matching is done by minimizing the squared distance between the user biases of review (A, B) and the beer biases of review (A, B)
# %%
# reset beer id and user id to be from 0 to nb_beer and nb_user (long, n^2)
def reset_id(df, col):
    df[col] = df[col].astype("category").cat.codes
    return df
# %%    
df_sample = df.sample(1000)
df_sample = reset_id(df_sample, "beer_id")
df_sample = reset_id(df_sample, "user_id")
# %%
# Matrix factorization with biases
import surprise.prediction_algorithms.matrix_factorization as mf
from surprise import Reader, Dataset

def get_biases(df, plot=False, verbose=False):
    algo = mf.SVD(n_factors=100, n_epochs=20, biased=True, lr_all=0.005, reg_all=0.02, verbose=verbose)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "beer_id", "rating"]].rename(columns={"user_id": "userID", "beer_id": "itemID", "rating": "rating"}), reader)
    user_bias = algo.fit(data.build_full_trainset()).bu
    beer_bias = algo.fit(data.build_full_trainset()).bi
    if plot:
        plt.hist(user_bias)
        plt.hist(beer_bias)
        plt.show()
    return user_bias, beer_bias
# %%
user_bais, beer_bias = get_biases(df_sample)
# %%
# add feature user bias and beer bias
df_sample["user_bias"] = df_sample.apply(lambda row: user_bias[row["user_id"]], axis=1)
df_sample["beer_bias"] = df_sample.apply(lambda row: beer_bias[row["beer_id"]], axis=1)

# %%
# reset index reviews
df_sample = df_sample.reset_index()
df_sample["idx"] = df_sample.index

# %%
import networkx as nx
B = nx.Graph()
# Add nodes with the node attribute "bipartite"
B.add_nodes_from(df_sample[df_sample["treatment"] == 0]["idx"], bipartite=0)
B.add_nodes_from(df_sample[df_sample["treatment"] == 1]["idx"], bipartite=1)
print(B)
# Add edge between nodes of opposite node sets. The weights are the difference in user_bias squared plus the difference in beer_bias squared
control_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
treatment_nodes = set(B) - control_nodes
for control in control_nodes:
    for treatment in treatment_nodes:
        B.add_edge(control, treatment, weight=(df_sample.loc[control]["user_bias"] - df_sample.loc[treatment]["user_bias"])**2 + (df_sample.loc[control]["beer_bias"] - df_sample.loc[treatment]["beer_bias"])**2)

# TODO: find an algorithm which is not O(n^2)

# %%
# find maximum matching of the graph
matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(B, top_nodes=control_nodes)
# the key-value pair in matching appear twice, so we can just take the first half of the matching
control_ids = list(matching.keys())[:len(matching)//2]
treatment_ids = list(matching.values())[:len(matching)//2]
# print intersection of matching keys and values
print(set(matching.keys()) & set(matching.values()))
# %%
# create a new df with the matching
df_control = df_sample.loc[control_ids]
df_treatment = df_sample.loc[treatment_ids]

# run a t-test on rating with the matching dataframe
res = ttest_ind(df_control["rating"], df_treatment["rating"])
print(res)
# after matching the result is not significative anymore
# TODO: verify with bigger sample size
# comput the difference of mean rating between control and treatment
print(df_control["rating"].mean() - df_treatment["rating"].mean())

#plot the distribution of rating for the matching
plt.hist(df_control["rating"], alpha=0.5, label="control")
plt.hist(df_treatment["rating"], alpha=0.5, label="treatment")
plt.show()

# %% [markdown]
# ### Matrix factorization (bis): comparison of bias vectors between control and treatment groups
# %%
# reset beer id and user id to be from 0 to nb_beer and nb_user (long, n^2)

df_treatment_sample = df[df["treatment"] == 1]
df_control_sample = df[df["treatment"] == 0]

#df_treatment_sample = df_treatment.sample(1000)
#df_control_sample = df_control.sample(1000)

df_treatment_sample = reset_id(df_treatment_sample, "beer_id")
df_treatment_sample = reset_id(df_treatment_sample, "user_id")
df_control_sample = reset_id(df_control_sample, "beer_id")
df_control_sample = reset_id(df_control_sample, "user_id")

user_bias_treatment, beer_bias_treatment = get_biases(df_treatment_sample)
user_bias_control, beer_bias_control = get_biases(df_control_sample)
# %%
# plot the difference in bias vectors between control and treatment groups
plt.hist(user_bias_treatment, label="treatment", alpha = 0.5, bins = 25)
plt.hist(user_bias_control, label="control", alpha = 0.5, bins = 25)
plt.title("user bias")
plt.legend()
plt.show()
plt.hist(beer_bias_treatment, label="treatment", alpha = 0.5, bins = 25)
plt.hist(beer_bias_control, label="control", alpha = 0.5, bins = 25)
plt.title("beer bias")
plt.legend()
plt.show()
# %%
# run a t-test on the difference in bias vectors
res = ttest_ind(user_bias_treatment, user_bias_control)
print(res)
# print difference in mean
print(np.mean(user_bias_treatment) - np.mean(user_bias_control))
res = ttest_ind(beer_bias_treatment, beer_bias_control)
print(res)
print(np.mean(beer_bias_treatment) - np.mean(beer_bias_control))
# interestingly we seem to always find similar values. So the effect is small and significative or not depending on the method (probably also the sample size)
# %% [markdown]
# ### Matrix factorization (ter): analysis per country:

groupby_country = df.groupby("user_country").count().sort_values(by="beer_id", ascending=False)
topten_country = groupby_country.index[:10]
df_topcountries = df[df["user_country"].isin(topten_country)]

# %%
# boostrapping function to get the confidence interval
# bootstrapping function
# input: dataset, nb of iterations
# output: sorted list of means, overal mean, 95% confidence interval

def bootstrapping_function(treatment, control, level = 0.05, iterations = 1000):
    differences = []
    for i in range(iterations):
        treatment_sample = np.random.choice(treatment, size = len(treatment), replace = True)
        control_sample = np.random.choice(control, size = len(treatment), replace = True)
        differences.append(np.mean(treatment_sample) - np.mean(control_sample))
    
    differences.sort()
    return np.mean(differences), differences[int(np.floor(level/2*iterations))], differences[int(np.ceil(1-(level/2) * iterations))]

N_BOOSTRAP = 1000 #number of time we boostrap each dataset (be careful with runtimes)
# sidak correct:
alpha_1 = 1-(1-0.05)**(1/10)
print(alpha_1)
# %%
list_results = []

from scipy.stats import bootstrap

for country in topten_country:
    df_country = df_topcountries[df_topcountries["user_country"] == country]
    df_country_treatment = df_country[df_country["treatment"] == 1]
    df_country_control = df_country[df_country["treatment"] == 0]
    user_bias_treatment, beer_bias_treatment = get_biases(df_country_treatment)
    user_bias_control, beer_bias_control = get_biases(df_country_control)

    # run a t-test on the difference in bias vectors
    print("country: ", country)
    stat_user, p_user = ttest_ind(user_bias_treatment, user_bias_control)
    diff_user = np.mean(user_bias_treatment) - np.mean(user_bias_control) #positive if treatment is better
    stat_beer, p_beer = ttest_ind(beer_bias_treatment, beer_bias_control)
    diff_beer = np.mean(beer_bias_treatment) - np.mean(beer_bias_control)
    print("user bias: ", stat_user, p_user, diff_user)
    print("beer bias: ", stat_beer, p_beer, diff_beer)

    # compute confidence interval
    diff_user_mean, diff_user_low, diff_user_high = bootstrapping_function(user_bias_treatment, user_bias_control, alpha_1, N_BOOSTRAP)

    #print(f"mean: {diff_user_mean:0.04}, 95%CI: [{diff_user_low:0.04}, {diff_user_high:0.04}]")

    #append results to list
    list_results.append({"country" : country,"user_bias_treatment" : user_bias_treatment, "beer_bias_treatment" : beer_bias_treatment, "user_bias_control" : user_bias_control, "beer_bias_control" : beer_bias_control, "diff_user_mean" : diff_user_mean, "diff_user_low" : diff_user_low, "diff_user_high" : diff_user_high})
# %%
df_results = pd.DataFrame(list_results)
df_results["err_low"] = df_results["diff_user_mean"] - df_results["diff_user_low"]
df_results["err_high"] = df_results["diff_user_high"] - df_results["diff_user_mean"]
fig, ax = plt.subplots(figsize = (10, 5))
plt.errorbar([i for i in range(len(df_results))], df_results["diff_user_mean"].to_numpy(), yerr=df_results[["err_low", "err_high"]].transpose().to_numpy(), fmt = 'o', color = 'b', label = "data")
ax.axhline(0, 0, 1, linestyle = "--", color = "k", label = "reference")
plt.xticks([i for i in range(len(df_results))], topten_country)
plt.ylabel("diference in user bias")
plt.title("diference in user bias confidence interval per country")
plt.legend()
plt.show()