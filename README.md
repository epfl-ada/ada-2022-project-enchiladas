# ada_beer

## Title
“The Irish all love Guinness, right?”

## Abstract
### A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why?
Many countries have (perhaps stereotypical) associations with different beer preferences. For example, one might think the Germans all love Helles but judge Guinness with the highest contempt. We aim to investigate if there truly is such a geographical and cultural influence on beer preferences by looking at differences in consumer ratings, ratings criterion and language. Furthermore, it is also often said that beers taste better at home. Certain countries also have different ways of discussing and critiquing beers. We are also interested in seeing if this can be linked to how international and open a country is to foreign beers. By utilizing the data from two of the biggest beer ranking websites, this project will be able to investigate whether there exist a "home bias" (preference for local produce) when users rate beers. Moreover, we will also look at other geographical-based factor that could bias reviews.
<!-- Mat, for me this abstract is a bit vague and goes in too many direction. Try to focus more towards our research questions>

<!-- Why ? How?


Since geographical data is available for both breweries and users, it should be possible to test  whether there is substantive geographical variation in beer preferences and in the type of language used in reviews. It would be very interesting to test rural vs urban differences by incorporating census data e.g. US census data https://data.census.gov/cedsci/. This immediately lends itself to further extensions with other variables of interest e.g. one can investigate the relationship with political leanings, average wealth, alcohol laws etc. -->

## Research Questions
### A list of research questions you would like to address during the project.
1. Are beer preferences influenced by geography/culture?
2. Do different cultures prioritise/prefer different aspects of beers such as feel? Are some cultures more critical of beer?
3. Do different cultures have stylistically different ways of writing reviews and discussing beer?
4. Is there a "home bias" for reviewers? I.e. do users rate local beers higher than their foreign counterparts? Can we say something about how open different cultures are to foreign beers?
5. Do we also see an effect if we look in the opposite direction i.e. is there a "home advantage" for local beers? Specifically, is a beer likely to have a higher rating from local users than from foreign ones?
6. Do users talk about foreign beers differently than they talk about their local ones?
7. What are the other factor that influence consumer ratings based on their location? In particular, in the US we can look at:
    - urban vs. rural
    - wealth
    - politics
<!-- is this too much? Actually all questions are interesting.-->

## Proposed Additional Datasets
### (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible.
- Per-state population, wealth (median income) is available from the [US Census Bureau](https://data.census.gov/cedsci/). Data size is negligible. For each state, we will collect two numbers: the population and the median income and trivially include it with our existing datasets.
- Per-state voting in presidental elections is available from the [Harvard Election Data Science Lab](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/42MVDX). Again, the data size is minimal and for each state we will just collect the votes for Republicans and votes for Democrats and trivially include it with our existing datasets.
<!-- maps dataset?: visualisation -->

## Methods (Tasks)
1. [ ] Investigate the existing data and check for any problems with the data. Clean/wrangle the data where necessary.
2. [ ] Collect the data from our additional datasets.
3. [ ] Create pickled files containing the _cleaned_ data we will need to conduct our experiments. Namely, we will have the following datasets
    - df_beers: matched beers from RateBeer and BeerAdvocate
    - df_brew: matched breweries from RateBeer and BeerAdvocate
    - df_ba: BeerAdvocate reviews for all of the matched beers
    - df_rb: RateBeer reviews for all of the matched beers
    - DS5: Dataset containing extra information per US state about the wealth (median income), politics (votes in recent presidential elections) and population
4. [ ] Create a pipeline for using the datasets, ensuring that we can handle the datasize. We also need to have filtering where appropriate e.g. if a beer has too few reviews it should be dropped.
5. [ ] Investigate the research questions (RQs). We have on purpose created many RQs since we do not know results ahead of time, and this will thus allow us to focus on areas where we find interesting results and dropping RQs where we don't have interesting findings to discuss. We conduct the analysis for RateBeer and BeerAdvocate separately, and then compare the results. This allows us to check if our findings are robust between the two platforms, controlling for an extra source of variation due to differing user bases. We can also use Sidak correction to adjust our p-values.
    - [ ] Investigate RQ1 by viewing ratings for different beer styles for each country
    - [ ] Investigate RQ2 by seeing if general ratings differ per country. Then furthermore seeing if certain countries are more critical on certain rating aspects.
    - [ ] Investigate RQ3 by computing distances between the language corpuses i.e. reviews for each country. To achieve this, we will leverage existing methods and research such as [Ruzicka](https://github.com/mikekestemont/ruzicka) or [PyStyl](https://github.com/mikekestemont/pystyl).
    - [ ] Investigate RQ4. To achieve this, we need to use a propensity analysis to obtain a local and foreign beer of comparable quality. To do this, we match using share of foreign reviews, share of local reviews, avg review. We end up with a table like so afterward:

        | User | Local Beer | Foreign Beer |
        | --- | --- | --- |
        | User1 | Rating1 | Rating2 |
        | User2 | Rating1 | Rating2 |
        | ... | ... | ... |
        
        We can then conduct a paired t-test to investigate if there exists such a "home bias".
        <!-- This I'll change to explicitely describe H0 and p = P(user's country == beer's country | covariates)-->
    - [ ] Investigate RQ5 using the same approach by now using propensity analysis to match users with similar rating patterns. Specifically, we match on share of foreign reviews, share of local reviews, average rating, and also match on style preference, and obtain:

        | Beer | Local User | Foreign User |
        | --- | --- | --- |
        | Beer1 | Rating1 | Rating2 |
        | Beer2 | Rating1 | Rating2 |
        | ... | ... | ... |

        As before, we conduct a paired t-test to investigate if there a "home advantage".
    - [ ] Investigate RQ6 by analysing differences in the stylistic language for local vs foreign beers. Will likely need to do this on a per-country basis since we expect countries to have different stylistic choices in general.
    - [ ] Investigate RQ7 by looking at ratings with respect to the following variables:
        - population
        - wealth
        - politics
6. [ ] Create a datastory and visualisations to communicate effectively our results.
        

## Proposed timeline
1. Already for Project Milestone 2, we have completed T1, T2, T3, T4 [FINISH LIST]
2. 

## Organization within the team: A list of internal milestones up until project Milestone P3.
- 

## Questions for TAs (optional): Add here any questions you have for us related to the proposed project.
