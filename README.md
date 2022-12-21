## Title
“The Irish all love Guinness, right?”

## Our Work
[Website](https://epfl-ada.github.io/ada-2022-project-enchiladas/) | [Code](https://github.com/epfl-ada/ada-2022-project-enchiladas) | [Milestone2](https://github.com/epfl-ada/ada-2022-project-enchiladas/tree/fe78b2989b4e2ac2de91e916df7d9e12c9a344b8)

## Code Structure
```
.
├── Data/
│   ├── file11.ext
│   └── file12.ext
├── models/
│   ├── file21.ext
│   ├── file22.ext
│   └── file23.ext
├── states_ba/
├── states_rb/
├── country/
├── helpers.py
├── ldist.ipynb
├── nlp_viz.ipynb
├── RQ1_and_2.py
├── home_bias.ipynb
├── home_bias.py
├── nlp.py
├── nlp.py
└── README.md
```

## Abstract
Countries have (perhaps stereotypical) associations with different beer preferences. For example, one might think the Germans all love Helles but judge Guinness with the highest contempt. We aim to investigate if there truly is such a geographical and cultural influence on beer preferences by looking at differences in consumer ratings, and in preferences in beer styles and beer characteristics. By utilizing the written reviews data from two of the biggest beer-ranking websites, we will be able to investigate if there are regional variations in language usage for beer reviews.

Finally, it is also often said that beers from home taste better. Leveraging the geographical information for users and beers, will be able to investigate whether there exists a "home bias" (preference for local produce) when users rate beers.

## Research Questions
In this work, we tackle our central research question:
_How are beer preferences affected by geography or culture?_

To understand this, we look at a number of important sub-questions
1. Are beer ratings influenced by geography/culture?
2. Do different cultures prioritise/prefer different aspects of beers such as palate? Are some cultures more critical of beer?
3. Do different cultures have stylistically different ways of writing reviews and discussing beer? Do users talk about foreign beers differently than they talk about their local ones?
4. Is there a "home bias" for reviewers? I.e. do users rate local beers higher than their foreign counterparts? Vice-versa, is there a "home advantage" for local beers, i.e. is the average rating from local reviewers higher than from foreign reviewers?

## Methods (Pipeline/Tasks)
1. [X] Initial data cleaning/wrangling. Missing values are marginal and are mostly dropped. We also address the coherence of the merged dataset (naming, filtering beers without review) and issues regarding country names (matching to ISO-codes). 
2. [X] Filtered and transformed datasets:
    - df_beers: matched beers from RateBeer and BeerAdvocate
    - df_brew: matched breweries from RateBeer and BeerAdvocate
    - df_ba: BeerAdvocate reviews for all of the matched beers including user data (pickled)
    - df_rb: RateBeer reviews for all of the matched beers including user data (pickled)
    By working on the pre-filtered merged dataset from 1., the sizes are reduced to 200mb and fit easily into RAM.
3. [X] From the pickled datasets, filter again depending on the research question (e.g. by minimum number of reviews or countries of interest), address similarity/differences between the two rating dataframes (min-max scaling), visualize the basic data (rating distributions for the two userbase, geographical distribution), enrich the beer styles by grouping similar beer styles together
4. [X] Conduct preliminary analysis by investigating mean differences in ratings for palette, aroma, overall etc. per country and per state. Conduct t-tests with Sidak correction to investigate if this is significant.
5. [X] See if one can explain these differences by removing any user-bias i.e. it could be that some users are more negative in general.
6. [X] Next, go about explaining all of the difference by looking simultaneously at both beer and user biases. To do this, employ a matrix factorisation.
7. 
8. 
9. 
10.
11. 
4. [ ] We conduct the analysis for RateBeer and BeerAdvocate separately, and then compare the results. This allows us to check if our findings are robust between the two platforms, controlling for an extra source of variation due to differing user bases.
    - [ ] Investigate RQ1 by viewing ratings for different beer styles for each country. Specifically, we want conduct t-tests investigating if there are differences in this between countries, using the Sidak correction.
    - [ ] Investigate RQ2 by seeing if general ratings differ per country. As in the previous point, we can do this using t-tests. Then furthermore we can see if certain countries are more critical on certain rating aspects.
    - [ ] Investigate RQ3 by finding word frequencies and computing distances between the language corpuses i.e. reviews for each country. To achieve this, we will leverage existing methods and research such as [Ruzicka](https://github.com/mikekestemont/ruzicka), [PyStyl](https://github.com/mikekestemont/pystyl) or [word-cloud](https://github.com/amueller/word_cloud). Furthermore, we will investigate if there are differences for foreign and local reviews.
    - [ ] Investigate RQ4 (the home bias/home advantage). Using a t-test, we compare the average user ratings given to local vs. foreign beers to see if they are different. To control the effect of cofounders, we match users based on propensity. Propensity score measures the probability of a user to rate beer from his own country/state vs. from a foreign one given observed covariates. Is is learned using a logistic regression with labels being 1 if the reviewed beer is local, 0 if it is foreign. Some features considered are for example user's avg ratings, number of ratings, user "taste" (ratings per style), country, etc...

## Requirements

## Organization within the team:
- Matthieu: Home bias (RQ4) and map visualisation (RQ 1,2,4)
- Oisín: Textual analysis distance and dendogram (RQ3)
- Kasimir: Bag of word for textual analysis, heat maps (RQ3)
- Andrea: Style preferences and aspects analysis (RQ1,2)
- Everyone: wrap-up of data story and visualization