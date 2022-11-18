## Title
“The Irish all love Guinness, right?”

## Abstract
Countries have (perhaps stereotypical) associations with different beer preferences. For example, one might think the Germans all love Helles but judge Guinness with the highest contempt. We aim to investigate if there truly is such a geographical and cultural influence on beer preferences by looking at differences in consumer ratings, and in preferences in beer styles and beer characteristics. By utilizing the written reviews data from two of the biggest beer-ranking websites, we will be able to investigate if there are regional variations in language usage for beer reviews.

Finally, it is also often said that beers from home taste better. Leveraging the geographical information for users and beers, will be able to investigate whether there exists a "home bias" (preference for local produce) when users rate beers.

## Research Questions
1. Are beer style preferences influenced by geography/culture?
2. Do different cultures prioritise/prefer different aspects of beers such as palate? Are some cultures more critical of beer?
3. Do different cultures have stylistically different ways of writing reviews and discussing beer? Do users talk about foreign beers differently than they talk about their local ones?
4. Is there a "home bias" for reviewers? I.e. do users rate local beers higher than their foreign counterparts? Vice-versa, is there a "home advantage" for local beers, i.e. is the average rating from local reviewers higher than from foreign reviewers?

_Alternatives_:
The following ideas were considered but didn't fit well with the direction of our data story.

5. What other factors influence consumer ratings based on location? In particular, in the US we can look at the following per state:
    - population density (from the [US Census Bureau](https://data.census.gov/cedsci/))
    - wealth (from the [US Census Bureau](https://data.census.gov/cedsci/))

## Proposed Additional Datasets
-

## Methods (Pipeline/Tasks)
1. [ ] Initial data cleaning/wrangling. Missing values are marginal and are mostly dropped. We also address the coherence of the merged dataset (naming, filtering beers without review) and issues regarding country names (matching to ISO-codes). 
2. [ ] Filtered and transformed datasets:
    - df_beers: matched beers from RateBeer and BeerAdvocate
    - df_brew: matched breweries from RateBeer and BeerAdvocate
    - df_ba: BeerAdvocate reviews for all of the matched beers including user data (pickled)
    - df_rb: RateBeer reviews for all of the matched beers including user data (pickled)
    By working on the pre-filtered merged dataset from 1., the sizes are reduced to 200mb and fit easily into RAM.
3. [ ] From the pickled datasets, filter again depending on the research question (e.g. by minimum number of reviews or countries of interest), address similarity/differences between the two rating dataframes (min-max scaling), visualize the basic data (rating distributions for the two userbase, geographical distribution), enrich the beer styles by grouping similar beer styles together
4. [ ] We conduct the analysis for RateBeer and BeerAdvocate separately, and then compare the results. This allows us to check if our findings are robust between the two platforms, controlling for an extra source of variation due to differing user bases.
    - [ ] Investigate RQ1 by viewing ratings for different beer styles for each country. Specifically, we want conduct t-tests investigating if there are differences in this between countries, using the Sidak correction.
    - [ ] Investigate RQ2 by seeing if general ratings differ per country. As in the previous point, we can do this using t-tests. Then furthermore we can see if certain countries are more critical on certain rating aspects.
    - [ ] Investigate RQ3 by finding word frequencies and computing distances between the language corpuses i.e. reviews for each country. To achieve this, we will leverage existing methods and research such as [Ruzicka](https://github.com/mikekestemont/ruzicka), [PyStyl](https://github.com/mikekestemont/pystyl) or [word-cloud](https://github.com/amueller/word_cloud). Furthermore, we will investigate if there are differences for foreign and local reviews.
    - [ ] Investigate RQ4 (the home bias/home advantage). Using a t-test, we compare the average user ratings given to local vs. foreign beers to see if they are different. To control the effect of cofounders, we match users based on propensity. Propensity score measures the probability of a user to rate beer from his own country/state vs. from a foreign one given observed covariates. Is is learned using a logistic regression with labels being 1 if the reviewed beer is local, 0 if it is foreign. Some features considered are for example user's avg ratings, number of ratings, user "taste" (ratings per style), country, etc...
5. [ ] The data story first gives an overall feeling on the country-based beer preferences visualizing the world/USA on a map. It then investigates textual geographical variation and visualises the linguistic differences between countries via wordmaps, and heatmaps showing the linguistic distances between countries. The textual analysis will finish with a dendrogram showing which countries are stylistically closest in terms of their review content. Finally, the results of the home bias analyses will be visualized using confidence intervals for each country.

## Proposed timeline
1. 18/11/2022: Tasks 1 (data wrangling), 2 (pickling), and 3 (data pipeline). We have also already begun T4 RQ1, (looking at variations in ratings, beer styles etc. per country).
2. 18/11/2022-02/12/2022: Dedicated to homework 2.
4. 09/12/2022: RQ1 and RQ2 are finished (simpler), start working on RQ3 and RQ4.
5. 16/12/2022: Begin working on visualisation and finish RQ3 and RQ4.
6. 23/12/2022: Finalise visualisation.

## Organization within the team:
- Matthieu: Home bias and map visualisation
- Oisín: Visualisation and textual analysis
- Kasimir: Visualisation and textual analysis
- Andrea: Home bias and textual analysis

## Questions for TAs (optional):
-
