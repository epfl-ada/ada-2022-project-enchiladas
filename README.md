# “The Irish all love Guinness, right?”

Countries have (perhaps stereotypical) associations with different beer preferences. For example, one might think the Germans all love Helles but judge Guinness with the highest contempt. We aim to investigate if there truly is such a geographical and cultural influence on beer preferences by looking at differences in consumer ratings, and in preferences in beer styles and beer characteristics.

We then conduct an observational study to evaluate if consummers are biased towards rating beers from their own country higher (or lower) than foreign ones.

Finaly, utilizing the written reviews data from two of the biggest beer-ranking websites, we will be able to investigate if there are regional variations in language usage for beer reviews.

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
├── RQ12.ipynb
├── home_bias.ipynb
├── home_bias.py
├── nlp.py
└── README.md
```
The files/folders contain the following information:
- `Data/` contains the raw beer data sourced from [here](https://drive.google.com/drive/folders/1Wz6D2FM25ydFw_-41I9uTwG9uNsN4TCF)
- `models/` contains 7 downloaded NLP models:
    - `fasttext-wiki-news-subwords-300`
    - `conceptnet-numberbatch-17-06-300`
    - `word2vec-ruscorpora-300`
    - `word2vec-google-news-300`
    - `glove-wiki-gigaword-300`
    - `glove-twitter-200`
    - `tf-idf`
- `states_ba/`, `states_rb/` and `country/` contain vectorised representations of the corpuses obtained from the NLP pipeline
- `helpers.py` contains some useful general functions
- `prettify.py` contains the plot style definitions
- `nlp.py` contains NLP-specific functions used in the NLP pipeline
- `RQ12.ipynb` contains the code for the analysis of RQ1 and RQ2
- [FINISH]


## Research Question
_How are beer preferences affected by geography or culture?_

To understand this, we look at 4 key sub-questions:
1. Are beer ratings different by country or state? If so, also by aspect?
2. Why are beer ratings different by country and state? 
3. Is there a "home bias" for reviewers? I.e. do users rate local beers higher than their foreign counterparts? Vice-versa, is there a "home advantage" for local beers, i.e. is the average rating from local reviewers higher than from foreign reviewers?
4. Do different cultures have stylistically different ways of writing reviews and discussing beer? Do users talk about foreign beers differently than they talk about their local ones?

## Methods and results
_For further details, please see our notebooks_

1. [X] Initial data cleaning/wrangling. Missing values are marginal and are mostly dropped. We also address the coherence of the merged dataset (naming, filtering beers without review) and issues regarding country names (matching to ISO-codes). 
2. [X] Filtered and transformed datasets:
    - `df_beers`: matched beers from RateBeer and BeerAdvocate
    - `df_brew`: matched breweries from RateBeer and BeerAdvocate
    - `df_ba`: BeerAdvocate reviews for all of the matched beers including user data (pickled)
    - `df_rb`: RateBeer reviews for all of the matched beers including user data (pickled)
    By working on the pre-filtered merged dataset from 1., the sizes are reduced to 200MB and fit easily into RAM.
3. [X] From the pickled datasets, filter again depending on the research question (e.g. by minimum number of reviews or countries of interest), address similarity/differences between the two rating dataframes (min-max scaling), visualize the basic data (rating distributions for the two userbase, geographical distribution), enrich the beer styles by grouping similar beer styles together
4. [X] Conduct preliminary analysis by investigating mean differences in ratings for palette, aroma, overall etc. per country and per state. Conduct independent t-tests in order to test if this difference is significant. Specifically, the test statistic is then constructed as follows
    $$ t = \frac{\mu_1 - \mu_2}{\sqrt{\frac{s_1^2}{N_1}+\frac{s_2^2}{N_2}}}$$
    where $\mu_i$ is the sample mean for the sample $i$ and $s_i$ is the _corrected_ sample standard deviation.
    Since we are conducting multiple t-tests, use the Šidák correction. Specifically, if we want our final test to be equivalent to a signifance level of $\alpha$ and we have $m$ independent hypothesis tests, then we conduct each individual hypothesis test at a significance level of
    $$ \alpha_1 = 1 - (1-\alpha)^{\frac{1}{m}}$$
5. [X] See if one can explain these differences by removing any user-bias - it could be that some users are more negative in general. To do this, rescale the ratings for each user such that the mean corresponds to a value of 0, the max corresponds to a value of 1 and the min correponds to a value of -1.
$$ GIVE FORMULA $$
6. [X] Check if the difference may be because users are reviewing different beers. Visualise the main beers per country and also conduct a statistical test to determine [FINISH]
7. [X] Investigate wether consummers have a preference for local beers over foreign ones. To accounts for difference in users critic level bias and beer quality , we compute user and beer bias vectors by performing [matrix factorization with biases](https://surprise.readthedocs.io/en/stable/matrix_factorization.html?highlight=matrix%20factorization) on the user-beer review matrix. Each rating is approximated by $\hat r_{user,beer} = \mu + b_{user} + b_{beer} + q_{beer}^T p_{user}$, from wich we recover the biases $b_{user}$ and $b_{beer}$.
The matching is initially done by computing the minimum weight matching bi-partite graph between local and foreign group (weight is the euclidean distance of user and beer biases difference between subjects). However, as we scale up to the full dataset, the number of possible connection grows with $O(n^2)$. To speed up the process, we use a stochastic approximation where we randomly match each users within a discretized equal frequency bining of the user and beer biases.
Once the dataset matched, we run a t-test to compare the mean rating of local and foreign reviews. We find a small, but significant difference.
9. [X] We group our review by user country and repeat the matching for the top 10 countries present in the dataset. We compute the mean difference between the two groups and confidence intervals using bootstraping and sidak-corrected significance level. The results show that most country actually rate foreign beers higher than local ones. This simpson paradox was caused by the imbalance towards american reviews in the initial dataset.
10. [X] Investigate the language data. Check the distribution of reviews per countries and filter to countries where we have enough data.
11. [X] Construct an NLP pipeline so one can begin to look at language. To do this, we first preprocess the dataset by casefolding, tokenising and removing any non-alphabetic data. For each state and country, vectorise the corpus using the following models:
    - `fasttext-wiki-news-subwords-300`
    - `conceptnet-numberbatch-17-06-300`
    - `word2vec-ruscorpora-300`
    - `word2vec-google-news-300`
    - `glove-wiki-gigaword-300`
    - `glove-twitter-200`
    - `tf-idf`

    To make all the vectorisations comparable, standardise by row.
    Then compute distances from the vectorised corpuses using the following metrics:
    - `euclidean`
    - `cityblock`
    - `cosine`
4.  [X] Run the NLP pipeline for 4 cases: on the raw data per country and state, on beer-specific words per country and state, ignoring the most common words per country and state and lastly on foreign versus local reviews.
5.  [X] Check if there is a difference in the language used for foreign versus local reviews. Investigate specifically if non-American reviewers sound more American when reviewing American beer.
6.  [X] Look more generally to see if we can pinpoint cultural differences between countries. Look at wordclouds to check whether or not countries use the same data.
7.  [X] Quantify the distance per country and state See if we can reconstruct geographical proximities or not.

_Note_: RateBeer only have reviews from the US, so we conduct the analysis for RateBeer and BeerAdvocate separately. Furthermore, the rating systems on the websites are different also. When looking at the US, we compare results. ?????

## Limitations
- Lack of data for some regions
- Dataset is heavily skewed towards the US
- RateBeer and BeerAdvocate are not directly comparable since they have different rating systems
- NLP models do not always have all words in vocabularly and also words can have different meanings in beer context to general usage
- Implemented NLP approach only works on English language data

## Contributions:
Andrea: 
- Create data visualisation pipeline using datapane
- Set-up website
- Front end development
- Documentation of code

Kasimir:
- Initial investigation into differences in averages per country
- Investigation using t-tests
- Investigation if user scaling explains the bias
- Investigation of ratings per style class and style
- Investigation if some countries rate beers that are just better overall
- Documentation of code

Matthieu:
- Propensity and matrix factorization matching methods
- Home bias study

Oisín:
- Research NLP methods
- Vectorise corpuses using 7 state-of-the-art NLP methods
- Compute distances using an ensemble model
- Documentation of code

## Requirements
[ADD AT END]

<!-- ## References
[1] blah -->