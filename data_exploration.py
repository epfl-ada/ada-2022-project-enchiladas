# %% [markdown]
# TODO: choose what to do with this file for the final report
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