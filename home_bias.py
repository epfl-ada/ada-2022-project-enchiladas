# %% [markdown]
# TODO: abstract of approach and key finidings
# %% [markdown]
# # RQ4: home bias
# %% [markdown]
# The following section details the analysis of the home bias effect in beer reviews, ie. do consummers rate local beers higher or lower than foreign ones. We define a review to be "local" if the beer and the reviewer are from the same country (the country of the beer is taken to be the location of the main brewery). A contrario, the review is "foreign" if the country of the user and the beer differ. The hypothesis of home bias can be formalized as follows:
#
# H0: $\mu_{local} = \mu_{foreign}$, the home bias effect is not present)
#
# H1: $\mu_{local} \neq \mu_{foreign}$, the home bias effect is present, either positively (local > foreign) or negatively (foreign > local)
#
# To test this hypothesis, we divide or dataset into two groups: one with local reviews (treatment) and one with foreign reviews (control). We then try to reject the null-hypothesis using a t-test on the overall rating of the review.

# %%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
import warnings
from scipy.stats import ttest_ind

#from prettify import *
pd.set_option('display.max_columns', 500)
img_path = "./Images/"
# %%
# load the data
df = pickle_load('./Data/' + 'df_ba_ratings_filtered_beers_merged_users')
# define the treatment variable
df["treatment"] = df.apply(lambda row: 1 if row["user_country"] == row["beer_country"] else 0, axis=1)
# %% [markdown]
# ## Preliminary data exploration for RQ4
# %% [markdown]
# In order to assess the effect of home bias, let's first explore the rate of local reviews per beer to se if we can further compare the treatment and control groups.
# %%
# create a feature "number of reviews" for each beer
df_groupby_beer = df.groupby(by = "beer_id").agg({"beer_id": "count", "treatment": "sum"})
df["nb_reviews_per_beer"] = df.apply(lambda row: df_groupby_beer.loc[row["beer_id"]]["beer_id"], axis=1)
# same for number of local and foreign reviews
df["nb_reviews_per_beer_local"] = df.apply(lambda row: df_groupby_beer.loc[row["beer_id"]]["treatment"], axis=1)
df["nb_reviews_per_beer_foreign"] = df["nb_reviews_per_beer"] - df["nb_reviews_per_beer_local"]
# compute the fraction of local reviews
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

# %%
# put the two bar plot on the same figure
groupby_country = df.groupby(by="user_country").agg({"beer_id": "count"}).sort_values(by="beer_id", ascending=False)
x_axis = np.arange(10)
x_label = groupby_country.head(10).index
plt.bar(x_axis-0.2, groupby_country.head(10)["beer_id"], 0.4, label="number of users")
groupby_country = df.groupby(by="beer_country").agg({"beer_id": "count"}).reindex(x_label)
plt.bar(x_axis+0.2, groupby_country.head(10)["beer_id"], 0.4, label="number of reviews on a beer of that country")
plt.title("number of users and beer reviews per country")
plt.xticks(x_axis, x_label, rotation=90)
plt.legend()
plt.show()
# %% [markdown]
# The distribution of local reviews makes sense. most of the users are concentrated in the us. Therefore a beer from the us is mostly rated by locals and an beer with fewer user in its country of origin (for example Belgium or Germany) mostly receives foreign reviews (as seen on the second plot).
#
# Looking at this very skewed distribution, it might make sense to subset the data to beers with a balanced distribution of local and foreign reviews (at least 10% of each). Let's see how many beers we would have to remove to get a balanced distribution.
# %%
# how many beers do have a somewhat balance distribution ? (20-80%)
groupby_beer[(groupby_beer["share_local_reviews"] > 0.2) & (groupby_beer["share_local_reviews"] < 0.8)]["share_local_reviews"].hist()
plt.title("zoom on beers with balanced distribution of local and foreign reviews")
plt.show()
# what is the fraction of beers with balanced distribution of local and foreign reviews ?
df_balanced = df[df["beer_id"].isin(groupby_beer[(groupby_beer["share_local_reviews"] > 0.2) & (groupby_beer["share_local_reviews"] < 0.8)].index)]

print("fraction of beers with balanced distribution of local and foreign reviews: ", len(df_balanced) / len(df))

# %% [markdown]
# subset the data to beers with balanced distribution of local and foreign reviews leaves only about 3% of the number of total reviews, which will make it difficult to draw any conclusions. However, this primary analysis shows that there is a need to find a way to balance the treatment and control groups, using for example propensity score matching.
# %% [markdown]
# ## Baseline: home bias analysis without any matching
# %% [markdown]
# Eventhough the data is not balanced, let's first run a simple t-test to see if there is a significant difference in ratings between local and foreign reviews (prior to matching).
# %%
# plot the rating distribution of the treatment and control groups prior to matching
df[df["treatment"] == 1]["rating"].hist(bins=20, alpha=0.5, label="local reviews")
df[df["treatment"] == 0]["rating"].hist(bins=20, alpha=0.5, label="foreign reviews")
plt.legend()
plt.title("distribution of local vs. foreign reviews")
plt.show()
# run a t-test to see if there is a significant difference in ratings between reviews with treatment = 1 and reviews with treatment = 0 (prior to matching)
res = ttest_ind(df[df["treatment"] == 1]["rating"], df[df["treatment"] == 0]["rating"], equal_var=False)
# print results of the t-test
print("t-test on the mean ratings of local and foreign reviews:")
print("t-statistic: ", res[0])
print("p-value: ", res[1])
print("average difference of mean ratings between treatment and control", df[df["treatment"] == 1]["rating"].mean() - df[df["treatment"] == 0]["rating"].mean())

# now if we subset to beer with at least 10% of foreign reviews and 10% of local reviews
df_balanced = df[(df["share_local_reviews"] > 0.2) & (df["share_local_reviews"] < 0.8)]
df_balanced[df_balanced["treatment"] == 1]["rating"].hist(bins=20, alpha=0.5, label="local reviews")
df_balanced[df_balanced["treatment"] == 0]["rating"].hist(bins=20, alpha=0.5, label="foreign reviews")
plt.title("distribution of reviews for beer with at least 20% of local and 20% of foreign reviews")
plt.show()

# run t-test
res = ttest_ind(df_balanced[df_balanced["treatment"] == 1]["rating"], df_balanced[df_balanced["treatment"] == 0]["rating"], equal_var=False)
print("t-test on the mean ratings of local and foreign reviews:")
print("t-statistic: ", res[0])
print("p-value: ", res[1])
print("average difference of mean ratings between treatment and control", df_balanced[df_balanced["treatment"] == 1]["rating"].mean() - df_balanced[df_balanced["treatment"] == 0]["rating"].mean())
# %% [markdown]
# The results seem to show some significance in the bias. However, as seen in the previous, the huge imbalances in the dataset lead to a lot of beer with either only local reviews or only foreign reviews. Because of that the result we see might be caused by other factors such as the country of origine of the beer or its quality. Therefore, we need to mitigate this imbalance in the dataset before running the analysis. However, if we subset only to beers with comparable amount of local and foreign reviews, the dataset will be too small to run a meaningful analysis. Therefor, we will try propensity score matching method to balance the dataset.
# %% [markdown]
# ## propensity score matching
# %% [markdown]
# ### propensity score calculcation using random forest
# %% [markdown]
# Firstly, we will try a random forest classifier to calculate the propensity score. Random forest is chosen over logistic regression because it can better deals with categorical features (such as country or style class). The classifier will be trained on the following features:

# - avg user rating
# - number of reviews
# - beer style class
# - beer average rating
# - beer country

# we cannot use user country and beer country  simultaneously beause they are already part of the treatment. Indeed, the classifier would then just have to check if they are equal or not to decide on the treatment/control group probabilities.

# %%
# add feature of average ratings given by the user
df_users = df.groupby(by = "user_id").agg({"rating": "mean"})
df["avg_user_rating"] = df.apply(lambda row: df_users.loc[row["user_id"]]["rating"], axis=1)

# categories the style class
df["style_class_cat"] = df["style_class"].astype("category").cat.codes
# categories the country
df["beer_country_cat"] = df["beer_country"].astype("category").cat.codes
# %%
# feature list
feature_list = ["avg_user_rating", "nbr_reviews", "style_class_cat", "avg_beer_rating", "beer_country_cat"]
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
# Eventhough the f1 score looks high, the predictor is still shit. The predictor is very much biased toward predicting treatment because the dataset is very unbalanced. In the case of predicting control, the model is wrong nearly 50% of the time.
# To better assess the quality of the mdel, we filter the review dataset to only keep countries with at least 1000 reviews (top 8 countries). We then balance our dataset by randomly sampling 1000 reviews per country.
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
y_pred = random_forest_propensity(X, y)
# %% [markdown]
# This analysis confirms than when the dataset is balanced, the random forest classifier is bad at predicting treatment and control group probabilities (f-score = 0.65). This is because the features available are not good enough to predict the treatment/control group. We will try to use matrix factorization with biases as an alternative method to balance our dataset.
# %% [markdown]
# ### matching using matrix factorization with biases vectors
# %% [markdown]
# The motivation of the method is the following: some beers are arguably of better quality than others. Similarly some user are more critical in their ratings than others.
#  If we want to isolate the effect of home bias on the rating, we should try to match reviews from the treatment and control groups which have similar beer quality and user criticism. We can retrieve those bias (for the users and the beers) using matrix factorization with biases, as demonstrated in the [Netflix Prize competition](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf).
# We then use those biases to match reviews between the treatment and control groups. The matching is done by minimizing the squared distance between the user biases of review (A, B) and the beer biases of review (A, B).


# %%
def reset_id(df, col):
    # reset beer id and user id to be from 0 to nb_beer and nb_user
    df[col] = df[col].astype("category").cat.codes
    return df

# Matrix factorization with biases
import surprise.prediction_algorithms.matrix_factorization as mf
from surprise import Reader, Dataset

def get_biases(df, plot=False, verbose=False):
    # returns the biases vectors for the beers and the users as two new columns in the dataframe
    df = reset_id(df, "beer_id")
    df = reset_id(df, "user_id")
    algo = mf.SVD(n_factors=100, n_epochs=20, biased=True, lr_all=0.005, reg_all=0.02, verbose=verbose) #default parameters
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "beer_id", "rating"]].rename(columns={"user_id": "userID", "beer_id": "itemID", "rating": "rating"}), reader)
    user_bias = algo.fit(data.build_full_trainset()).bu
    beer_bias = algo.fit(data.build_full_trainset()).bi
    if plot:
        plt.hist(user_bias, alpha = 0.5, bins = 100, label="user bias", density=True)
        plt.hist(beer_bias, alpha = 0.5, bins = 100, label="beer bias", density=True)
        plt.xlabel("Bias")
        plt.ylabel("Density")
        plt.title("Histogram of user and beer biases")
        plt.legend()
        plt.show()
    df["user_bias"] = df.apply(lambda row: user_bias[row["user_id"]], axis=1)
    df["beer_bias"] = df.apply(lambda row: beer_bias[row["beer_id"]], axis=1)
    return df
# %%
df = get_biases(df, plot=True)
# %% [markdown]
# The following algorithm creates a bipartite graph where each node of the control group is connected to each node of the treatment group. The weight of the edge is the squared distance between the user bias and the beer bias of the two reviews. The algorithm then finds the minimum weight matching between the two groups. The matching is done using the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm).
# The main issue of this approach is that we need to create 3.1e11 ($O(n^2)$) connections between the nodes of the two groups, which would simply take to loong. We therefore take a sample of the reviews to try the algorithm.
# %%
import networkx as nx

def graph_matching(df, concat = False):
    # reset index reviews
    df = df.reset_index()
    df["idx"] = df.index
    B = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    B.add_nodes_from(df[df["treatment"] == 0]["idx"], bipartite=0)
    B.add_nodes_from(df[df["treatment"] == 1]["idx"], bipartite=1)
    # Add edge between nodes of opposite node sets. The weights are the difference in user_bias squared plus the difference in beer_bias squared
    control_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
    treatment_nodes = set(B) - control_nodes
    for control in control_nodes:
        for treatment in treatment_nodes:
            B.add_edge(control, treatment, weight=(df.loc[control]["user_bias"] - df.loc[treatment]["user_bias"])**2 + (df.loc[control]["beer_bias"] - df.loc[treatment]["beer_bias"])**2)
    # find minimum weight matching of the graph
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        matching_double = nx.bipartite.minimum_weight_full_matching(B, top_nodes = control_nodes)
    
    # the key-value pair in matching appear twice, so we remove from matching all keys which are in the treatment group
    matching = {k: v for k, v in matching_double.items() if k in control_nodes}

    # get the ids of the reviews in the treatment and control groups
    control_ids = [k for k, v in matching.items()]
    treatment_ids = [v for k, v in matching.items()]
    # create a new df with the matching
    df_control = df.loc[control_ids]
    df_treatment = df.loc[treatment_ids]

    # add weight to the dataframe
    df_control["distance"] = df_control.apply(lambda row: np.sqrt(B[row["idx"]][matching_double[row["idx"]]]["weight"]), axis=1)
    df_treatment["weight"] = df_treatment.apply(lambda row: np.sqrt(B[matching_double[row["idx"]]][row["idx"]]["weight"]), axis=1)

    if concat:
        return pd.concat([df_control, df_treatment])
    else:
        return df_control, df_treatment
# %%
# run the graph matching algorithm (slow)
df_sample = df.sample(frac = 0.001)

df_control, df_treatment = graph_matching(df_sample)

# run a t-test on rating with the matching dataframe
res = ttest_ind(df_control["rating"], df_treatment["rating"])
print(res)
# comput the difference of mean rating between control and treatment
print(df_treatment["rating"].mean() - df_control["rating"].mean())

#plot the distribution of rating for the matching
plt.hist(df_treatment["rating"], alpha=0.5, label="treatment", bins=25, density=True)
plt.hist(df_control["rating"], alpha=0.5, label="control", bins=25, density=True)
plt.legend()
plt.show()

# %% [markdown]
# After propensity matching, we still find some difference in the mean rating between the control and treatment groups. However, the result is not significant because the sample size is too small. Since it is impossible to run the algorithm on the whole dataset, we will try to develop an approximate matching approach.

# %% [markdown]
# ### Matrix factorization: nearest neighbour
# create control treatment pairs where the user bias are close (within a bin of width epsilon)

# %% [markdown]
# The idea of this method is the following: after computing the user and beer biases, we match each local review 1-1 with another foreign review where the user bias and beer bias are close, ie within the same bin. This minimize the number of possible connections to $O(m^2)$ where m is the number of points in the bin. For example, in 1D, if we bin the data with 10 equal frequency bins, the amount of edges for each graph has effectively been divided by $10^2 = 100$ (but as counter-effect, there are now 10 problems to solve). In the case where bin size increase to infinity (only 1 bin), we recover the optimal solution. The bins can be visualized on the following plot:
# %%
# 2D hexagonal heat map plot of user bias and beer bias for treatment and control groups
df = get_biases(df, plot=False)
plt.hexbin(df[df["treatment"] == 0]["user_bias"], df[df["treatment"] == 0]["beer_bias"], bins = 'log', gridsize=25, label="control", alpha=0.5)
plt.xlabel("user bias")
plt.ylabel("beer bias")
plt.legend()
plt.show()
plt.hexbin(df[df["treatment"] == 1]["user_bias"], df[df["treatment"] == 1]["beer_bias"], bins = 'log', gridsize=25, label="treatment", alpha=0.5)
plt.xlabel("user bias")
plt.ylabel("beer bias")
plt.legend()
plt.show()
# %% [markdown]
# The idea behind this method is that it is very unlikely for a point to be matched with another point outside its neighbourhood (bin) since the euclidiean distance would be to big.
# When binning the user and beer bias, we have 4 different options:
# - equal width: the bins have the same width
# - equal width log: the bins have the same width in log scale
# - equal frequency: we bin each axis (user_bias, beer_bias) so that each bin has the same number of points. However, in practice the 2D bins are not equal in size. Bins on the extrem corners of the space have less points than bins in the middle.
# - equal frequency log: same idea but for log re-scaled biases.
#
# To choose which binning method to use, we can compare the average pair euclidean distance for each method.
# The implementation of this discretized approximate graph matching algorithm is based on the previous function graph_matching and is included in the following code. However, in practice, we found that the algorithm is still too slow to run on the whole dataset on our machine. A test on a sample of the dataset is shown below.

# %%

def log_bias(df):
    # create a new column with the log of bias (we add one to deal with the case where the bias is 0)
    df["user_bias_log"] = df["user_bias"].apply(lambda x: np.log(x+1))
    df["beer_bias_log"] = df["beer_bias"].apply(lambda x: np.log(x+1))
    return df

def discretized_graph_matching(df):
    if len(df[df["treatment"] == 1]) == 0 or len(df[df["treatment"] == 0]) == 0:
        return None
    else:
        return graph_matching(df, concat=True)

def bin_data(df, bin_method, bins=20, column1="user_bias", column2="beer_bias"):
    # create a new column with the discretized bias

    pd.options.mode.chained_assignment = None
    if bin_method == "equal_frequency":
        df["user_bias_discretized"] = pd.qcut(df[column1], bins, labels=False, duplicates='drop')
        df["beer_bias_discretized"] = pd.qcut(df[column2], bins, labels=False, duplicates='drop')
    elif bin_method == "equal_width":
        df["user_bias_discretized"] = pd.cut(df[column1], bins, labels=False)
        df["beer_bias_discretized"] = pd.cut(df[column2], bins, labels=False)
    elif bin_method == "log_equal_frequency":
        df = log_bias(df)
        df["user_bias_discretized"] = pd.qcut(df[column1 + "_log"], bins, labels=False)
        df["beer_bias_discretized"] = pd.qcut(df[column2 + "_log"], bins, labels=False)
    elif bin_method == "log_equal_width":
        df = log_bias(df)
        df["user_bias_discretized"] = pd.cut(df[column1 + "_log"], bins, labels=False)
        df["beer_bias_discretized"] = pd.cut(df[column2 + "_log"], bins, labels=False)
    else:
        raise ValueError("bininng method not supported")
    return df


def discretized_matching(df, bin_method = "equal_frequency", bins=20, column1="user_bias", column2="beer_bias"):
    #approximately matches each review to a review in the other group by binning the user and beer bias and then balancing each bin
    df = bin_data(df, bin_method, bins, column1, column2)
    
    # performs approximate graph matching within each bin
    df_balanced = df.groupby(["user_bias_discretized", "beer_bias_discretized"]).apply(lambda x: discretized_graph_matching(x)).reset_index(drop=True)

    return df_balanced
# %%
# run-test with exact matching and equal frequency
df_balanced = discretized_matching(df.sample(frac = 0.001, random_state=42), bin_method="equal_frequency", bins = 20)
# sanity check
# sanity check if there are the same number of reviews in treatment and control groups
df_balanced.groupby("treatment").count()

#print average distance
print("average distance between the matching reviews")
print(df_balanced["distance"].mean())

# %% [markdown]
# ### stochastic method
# In order to perform the matching of the whole dataset in reasonnable time, we finnally decided to use a stochastic relaxation of the algorithm above. The idea is to bin the user and beer bias and then to balance each bin by sampling the majoritarian group (treatment or control) without replacement. The number of samples is the number of reviews in the minoritarian group. Altough this algorithm will result in a less precise matching than the exact or discretized graph matching presented above, it will still give a reasonnable approximation of the perfect matching. Once each bin is balanced, and given that bins are sufficiently small, we can expect that the average distance between the potential matching reviews will be small. This solution doesn't require to find exact pairs and is thus linear in the number of reviews. The implementation of this algorithm is included in the following code.

# %%

def random_balance(row, compute_distance=True):
    treatment = row[row["treatment"] == 1]
    control = row[row["treatment"] == 0]
    if len(treatment) == 0 or len(control) == 0:
        return None
    if len(treatment) > len(control):
        treatment = treatment.sample(len(control), random_state=42)
    else:
        control = control.sample(len(treatment), random_state=42)
    # compute the distance of each review to the mean of the treatment and control groups
    if compute_distance:
        treatment["distance"] = treatment.apply(lambda x: np.sqrt((x["user_bias"] - treatment["user_bias"].mean())**2 + (x["beer_bias"] - treatment["beer_bias"].mean())**2), axis=1)
        control["distance"] = control.apply(lambda x: np.sqrt((x["user_bias"] - control["user_bias"].mean())**2 + (x["beer_bias"] - control["beer_bias"].mean())**2), axis=1)
    return pd.concat([treatment, control])

# redefine discretized_matching to include the random balancing method
def discretized_matching_updated(df, bin_method = "equal_frequency", match_method = "random_matching", bins=20, column1="user_bias", column2="beer_bias", compute_distance=True):
    #approximately matches each review to a review in the other group by binning the user and beer bias and then balancing each bin
    df = bin_data(df, bin_method, bins, column1, column2)

    # choose matching method
    if match_method == "random_matching":
        df_balanced = df.groupby(["user_bias_discretized", "beer_bias_discretized"]).apply(lambda x: random_balance(x, compute_distance)).reset_index(drop=True)
    elif match_method == "graph_matching":
        df_balanced = df.groupby(["user_bias_discretized", "beer_bias_discretized"]).apply(lambda x: discretized_graph_matching(x)).reset_index(drop=True)
    else:
        raise ValueError("matching method not supported")
    return df_balanced

# %%
# To choose the binning method, we run the algorithm with different binning methods and different number of bins and then compare the performance of the matching. Since the stochastic algorithm doesn't provide us the pairs of the 1-1 matching, we use as a proxy the average distance of each point to its bin center. We then select the binning method that gives the smallest average distance.
# %%
# assessing the quality of the matching for each binning option
df_sample = df
bin_methods = ["equal_frequency", "equal_width", "log_equal_frequency", "log_equal_width"]
for method in bin_methods:
    print(method)
    df_balanced = discretized_matching_updated(df_sample, bin_method=method, match_method="random_matching", bins = 20)
    # total number of reviews in the balanced dataframe
    print(len(df_balanced))
    # average distance between the matching reviews
    print("average distance between the matching reviews")
    print(df_balanced["distance"].mean())
    # run a t-test on rating with the balanced dataframe
    res = ttest_ind(df_balanced[df_balanced["treatment"] == 0]["rating"], df_balanced[df_balanced["treatment"] == 1]["rating"])
    print(res)
    print(df_balanced[df_balanced["treatment"] == 1]["rating"].mean() - df_balanced[df_balanced["treatment"] == 0]["rating"].mean())

# %% [markdown]
#We can see that the log_equal_frequency method gives the best results. We will use this method for the rest of the analysis. Also, even though the binning reduces the number of possible edges between control and treatment groups, the number of edges is still $O(n^2)$ which makes it impossible to run on the full dataset. We will therefore use the random method to balance the dataset, which we suppose would give a similar approximation to the graph matching method.

# %% [markdown]
# Finally, we can run the matching on the full dataset and perform a t-test between the rating of the two groups in the matched reviews dataset.
# %%
df_balanced = discretized_matching_updated(df, bin_method="log_equal_frequency", match_method = "random_matching", bins = 20, compute_distance = False)

# %%
# run a t-test on rating with the balanced dataframe
res = ttest_ind(df_balanced[df_balanced["treatment"] == 0]["rating"], df_balanced[df_balanced["treatment"] == 1]["rating"])
print(res)
# print difference of mean rating between control and treatment
print(df_balanced[df_balanced["treatment"] == 1]["rating"].mean() - df_balanced[df_balanced["treatment"] == 0]["rating"].mean())

# plot histogram of rating for treatment and control groups
plt.hist(df_balanced[df_balanced["treatment"] == 1]["rating"], label="local reviews", alpha = 0.5, bins = 25, density = True)
plt.hist(df_balanced[df_balanced["treatment"] == 0]["rating"], label="foreign reviews", alpha = 0.5, bins = 25, density = True)
plt.xlabel("rating")
plt.ylabel("density")
# add a vertical line at the mean rating
plt.axvline(df_balanced[df_balanced["treatment"] == 1]["rating"].mean(), color='b', linestyle='dashed', linewidth=1, label="mean rating local")
plt.axvline(df_balanced[df_balanced["treatment"] == 0]["rating"].mean(), color='orangered', linestyle='dashed', linewidth=1, label="mean rating foreign")
plt.legend()
plt.savefig("Images/rating_distribution_homebias.jpg", dpi = 500)
plt.title("Rating distribution")
plt.show()
# %%
def bootstrapping_function(treatment, control, level = 0.05, iterations = 1000):
    # boostrapping function to get the confidence interval
    # bootstrapping function
    # input: dataset, nb of iterations
    # output: overal mean, 95% confidence interval
    np.random.seed(42)
    differences = []
    for i in range(iterations):
        treatment_sample = np.random.choice(treatment, size = len(treatment), replace = True)
        control_sample = np.random.choice(control, size = len(treatment), replace = True)
        differences.append(np.mean(treatment_sample) - np.mean(control_sample))
    
    differences.sort()
    return np.mean(differences), differences[int(np.ceil(level/2*iterations))], differences[int(np.floor(1-(level/2) * iterations))]

N_BOOSTRAP = 1000 #number of time we boostrap each dataset (be careful with runtimes)
# %%
diff_user_mean, diff_user_low, diff_user_high = bootstrapping_function(df_balanced["rating"].loc[df_balanced["treatment"] == 1], df_balanced["rating"].loc[df_balanced["treatment"] == 0], level = 0.05)

df_results = pd.DataFrame({"diff_user_mean": [diff_user_mean], "diff_user_low": [diff_user_low], "diff_user_high": [diff_user_high]})

# %%
df_results["err_low"] = df_results["diff_user_mean"] - df_results["diff_user_low"]
df_results["err_high"] = df_results["diff_user_high"] - df_results["diff_user_mean"]
fig, ax = plt.subplots()
plt.errorbar([i for i in range(len(df_results))], df_results["diff_user_mean"].to_numpy(), yerr=df_results[["err_low", "err_high"]].transpose().to_numpy(), barsabove=True, capsize = 5, fmt = '.b', label = "mean (95% CI)")
ax.axhline(0, 0, 1, linestyle = "--", color = "k", label = "no difference")
plt.legend()
plt.xticks([i for i in range(len(df_results))], ["full dataset"])
plt.ylabel("user local - foreign rating difference")
plt.title("diference in user rating between local and foreign reviews")
plt.show()
# %% [markdown]
# The difference of distribution of rating between local and foreign reviews is almost indistinguishable. However, looking at the 95% confidence intervall, the effect is still statistically significant. We can conclude that the local beers are rated higher than the foreign beers, but not by much.
# %% [markdown]
# ### Matrix factorization: analysis per country:
# %% [markdown]
# We can now perform the same analysis on the dataset per country. We will use the same discretized stochastic matching method as before and perform matching for each country separately. We will only consider the top 10 countries of the dataset to ensure that we have enough reviews for each country.
# since we are testing results for 10 countries, we need to correct for multiple testing. We will use the Sidak correction. The alpha value for the Sidak correction is $\alpha = 1 - (1-0.05)^\frac{1}{10} = 0.005 \%$, which will use to correct for multiple testing.

# %%
# get the top 10 countries
groupby_country = df.groupby("user_country").count().sort_values(by="beer_id", ascending=False)
topten_country = groupby_country.index[:10]
print("The top 10 countries are: ", topten_country)
df_topcountries = df[df["user_country"].isin(topten_country)]
# %%
# sidak correction
alpha_1 = 1-(1-0.05)**(1/10)
print("The alpha value for the sidak correction is: ", alpha_1)
# %%
# get the results for each country
list_results = []
for country in topten_country:
    df_country = df_topcountries[df_topcountries["user_country"] == country].copy()
    print(f"country: {country}, number of reviews: {len(df_country)}")
    df_country = get_biases(df_country)
    # perform matching of the country
    df_country = discretized_matching_updated(df_country, bin_method = "equal_frequency", match_method = "random_matching", bins = 10)
    # run a t-test on rating with the balanced dataframe
    res = ttest_ind(df_country[df_country["treatment"] == 1]["rating"], df_country[df_country["treatment"] == 0]["rating"])
    print(f"{country}: {res}")
    # print difference of mean rating between control and treatment
    print("country: ", country)
    print(df_country[df_country["treatment"] == 1]["rating"].mean() - df_country[df_country["treatment"] == 0]["rating"].mean())

    # compute confidence interval
    diff_user_mean, diff_user_low, diff_user_high = bootstrapping_function(df_country["rating"].loc[df_country["treatment"] == 1], df_country["rating"].loc[df_country["treatment"] == 0], level = alpha_1, iterations = 1000)

    #append results to list in key-value pairs
    list_results.append({"country": country, "diff_user_mean": diff_user_mean, "diff_user_low": diff_user_low, "diff_user_high": diff_user_high})
# %%
# plotting of the results
df_results = pd.DataFrame(list_results)
df_results["idx"] = [i for i in range(len(df_results))]
# computing error bars
df_results["err_low"] = df_results["diff_user_mean"] - df_results["diff_user_low"]
df_results["err_high"] = df_results["diff_user_high"] - df_results["diff_user_mean"]

# plotting
fig, ax = plt.subplots(figsize = (10, 5))
plt.errorbar([i for i in range(len(df_results))], df_results["diff_user_mean"].to_numpy(), yerr=df_results[["err_low", "err_high"]].transpose().to_numpy(), fmt = 'o', color = 'chocolate', label = f"mean ({(1 - alpha_1)*100:.3f}% CI)")
ax.axhline(0, 0, 1, linestyle = "--", color = "gray", label = "no difference")
plt.xticks([i for i in range(len(df_results))], topten_country)
plt.ylabel("diference in user average rating for local and foreign beers")
plt.title("user local - foreign rating difference")
plt.legend()
plt.savefig("Images/homebias_confidence_intervall_countries.jpg", dpi = 500)
plt.show()
# %% [markdown]
# Developping the analysis per country revealed a form of simpson's paradoxe. Altough the general dataset seem to be biased towards local beers, when analysing per country bias, we notice that 8 of the top 10 countries are biased towards foreign beers. This effect was hidden in the global analysis because the dataset mostly consists of reviews from the US, which is slightly positively biased towards local beers.
# 
# While the best beer country is subjective, Belgium and US are widely considered as major beer countries according to this [top 13 best beer countries in the world](https://www.thrillist.com/drink/nation/the-best-beer-countries-in-the-world). Therefor, it makes sense that user from those countries are biased toward rating their local beer higher. However, this argument is limited because other countries such as England or the Netherlands are also famous beer countries but have a bias towards foreign beers.
# 
# Bias towards foreign products might also be a result of consumer cosmopolitanism, an effect which described by  “the extent to which a consumer (1) exhibits an open-mindedness towards foreign countries and cultures, (2) appreciates the diversity brought about by the availability of products from different national and cultural origins, and (3) is positively disposed towards consuming products from foreign countries.” This effect also contributs to self identity, ie. “[the] frame of reference by which individuals evaluate their self-worth.” according to [Balabanis, G., Stathopoulou, A., & Qiao, J. (2019). Favoritism Toward Foreign and Domestic Brands: A Comparison of Different Theoretical Explanations. Journal of International Marketing](https://openaccess.city.ac.uk/id/eprint/23521/)

# %% [markdown]
# ### (discarded) Matrix factorization: comparison of bias vectors between control and treatment groups
# The idea of this method is to perform two separate matrix factorization on the control and treatment groups, and compare the bias vectors. This method has been discarded for two reasons:
# - It didn't help to rebalance the dataset because the method didn't involve some kind of matching between the groups. Therefor, it didn't add much compared to the initial naive approach.
# - It wasn't clear if the biases from matrix factorization where scaled to the rating and therefor the result couldn't be compared with the other methods.
# the code is left here for reference.
# %%
df_treatment = df_sample[df_sample["treatment"] == 1]
df_control = df_sample[df_sample["treatment"] == 0]

# balance the number of reviews in the control and treatment groups
df_treatment = df_treatment.sample(n = 924)
df_control = df_control.sample(n = 924)

user_bias_treatment, beer_bias_treatment = get_biases(df_treatment)
user_bias_control, beer_bias_control = get_biases(df_control)
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

# %% [markdown]
# ## Creating datapane for the website
# %%
# %%
import datapane as dp
# create datapane:

method_distribution = dp.Text('The ratings of each dataset are split into two groups: local when the user rates a beer from its own counry and foreign. To accounts for difference in users critic level bias and beer quality , we compute user and beer bias vectors by performing [matrix factorization with biases](https://surprise.readthedocs.io/en/stable/matrix_factorization.html?highlight=matrix%20factorization) on the user-beer review matrix. Each rating is approximated by $\hat r_{user,beer} = \mu + b_{user} + b_{beer} + q_{beer}^T p_{user}$, from wich we recover the biases $b_{user}$ and $b_{beer}$. The reviews are then put into bins of equal frequency in user and beer biases, then approximaely matched by resampling the majoritarian group within each bin. Once the dataset is balanced, we run a t-test to compare the mean rating of local and foreign reviews. We find a small, but significant difference.')

app = dp.App(
    dp.Page(title= "Home bias", blocks=["# BeerAdvocate", dp.Media(file="Images/rating_distribution_homebias.jpg", name="home_bias_distribution", caption="distribution of rating for local and foreign")]),
    dp.Page(title="Method", blocks=["# Method", method_distribution]))

app.save(path="Images/home_bias.html")

method_country = dp.Text('We group our review by user country and repeat the matching for each country in the top 10 number of reviews in the dataset. We compute the mean difference between the two groups and confidence intervals using bootstraping and sidak-corrected significance level.')

app = dp.App(
    dp.Page(title= "Home bias", blocks=["# BeerAdvocate", dp.Media(file="Images/homebias_confidence_intervall_countries.jpg", name="home_bias_confidence_countries", caption="Rating difference between local and foreign beers per country")]),
    dp.Page(title="Method", blocks=["# Method", method_country]))

app.save(path="Images/home_bias_countries.html")