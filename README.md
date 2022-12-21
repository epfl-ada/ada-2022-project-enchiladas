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
4. [X] Conduct preliminary analysis by investigating mean differences in ratings for palette, aroma, overall etc. per country and per state. Conduct independent t-tests in order to test if this difference is significant. Specifically, the test statistic is then constructed as follows
    $$ t = \frac{\mu_1 - \mu_2}{\sqrt{\frac{s_1^2}{N_1}+\frac{s_2^2}{N_2}}}$$
    where $\mu_i$ is the sample mean for the sample $i$ and $s_i$ is the _corrected_ sample standard deviation.
    Since we are conducting multiple t-tests, use the Sidak correction. Specifically, if we want our final test to be equivalent to a signifance level of $\alpha$ and we have $m$ independent hypothesis tests, then we conduct each individual hypothesis test at a significance level of
    $$ \alpha_1 = 1 - (1-\alpha)^{\frac{1}{m}}$$
5. [X] See if one can explain these differences by removing any user-bias i.e. it could be that some users are more negative in general. To do this, rescale the ratings such that the mean corresponds to a value of 0, the max a value of 1 and the min a value of -1.
$$ GIVE FORMULA $$
6. [X] Move investigation onto checking if the difference may be because users are reviewing different beers. [EXPLAIN MORE]
7. [X] Next, go about explaining all of the difference by looking simultaneously at both beer and user biases. To do this, employ a matrix factorisation.
    $$ GIVE FORMULA $$
    and then furthermore propensity analyses.
8. [X] Then look per country.
9. [X] Begin to look at language. Ergo, construct an NLP pipeline as follows:
    [GIVE DETAILS]
9. [X] Check if this is also reflected by language. Specifically, is there a difference in the language used for foreign vs local.
10. [X] Look more generally to see if we can pinpoint cultural differences between countries. Start just by looking at wordclouds
11. [X] Quantify the distance per country and state See if we can reconstruct geographical proximity.

_Note_: RateBeer only have reviews from the US, so we conduct the analysis for RateBeer and BeerAdvocate separately. Furthermore, the rating systems on the websites are different also. When looking at the US, we compare results.

## Limitations
- Lack of data for some regions
- Dataset is skewed
- RB and BA are not directly comparable
- NLP models do not always have all words and also words can have different meanings in beer context to general English

## Organization within the team:
Andrea: 
- Create data visualisation pipeline using datapane
- Set-up website
- Front end development
Kasimir:
- Initial investigation into differences in averages per country
- Investigation using t-tests
- Investigation if user scaling explains the bias
- Investigation if beers rated are different
Matthieu:
- Create matrix factorisation
- Propensity matching
- Conduct analysis per country
Oisín:
- Create NLP pipeline

## Requirements
[ADD AT END]

## References
[1] blah