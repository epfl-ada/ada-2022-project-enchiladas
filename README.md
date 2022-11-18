## Title
“The Irish all love Guinness, right?”

## Abstract
### A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why?
Many countries have (perhaps stereotypical) associations with different beer preferences. For example, one might think the Germans all love Helles but judge Guinness with the highest contempt. We aim to investigate if there truly is such a geographical and cultural influence on beer preferences by looking at differences in consumer ratings, ratings criterion and language. Furthermore, it is also often said that beers taste better at home. Certain countries also have different ways of discussing and critiquing beers. We are also interested in seeing if this can be linked to how international and open a country is to foreign beers. By utilizing the data from two of the biggest beer ranking websites, this project will be able to investigate whether there exist a "home bias" (preference for local produce) when users rate beers. Moreover, we will also look at other geographical-based factor that could bias reviews.
<!-- Mat, for me this abstract is a bit vague and goes in too many direction. Try to focus more towards our research questions>

<!-- Why ? How?


Since geographical data is available for both breweries and users, it should be possible to test  whether there is substantive geographical variation in beer preferences and in the type of language used in reviews. It would be very interesting to test rural vs urban differences by incorporating census data e.g. US census data https://data.census.gov/cedsci/. This immediately lends itself to further extensions with other variables of interest e.g. one can investigate the relationship with political leanings, average wealth, alcohol laws etc. -->

## Research Questions
1. Are beer preferences influenced by geography/culture?
2. Do different cultures prioritise/prefer different aspects of beers such as feel? Are some cultures more critical of beer?
3. Do different cultures have stylistically different ways of writing reviews and discussing beer? Do users talk about foreign beers differently than they talk about their local ones?
4. _(Bonus)_ Is there a "home bias" for reviewers? I.e. do users rate local beers higher than their foreign counterparts? Vice-versa, is there a "home advantage" for local beers, ie. is the average rating from local reviewers higher than from foreign reviewers?

_Alternatives_:
The following ideas were considered but didn't fit well with the direction of our data story.

5. What other factors influence consumer ratings based on location? In particular, in the US we can look at the following per state:
    - population density (from the [US Census Bureau](https://data.census.gov/cedsci/))
    - wealth (from the [US Census Bureau](https://data.census.gov/cedsci/))

## Proposed Additional Datasets
### (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible.

## Methods (Tasks)
1. [ ] Investigate the existing data and check for any problems with the data. Clean/wrangle the data where necessary.
2. [ ] Collect the data from additional datasets.
3. [ ] Build the following pickled datasets from the _cleaned_ data:
    - df_beers: matched beers from RateBeer and BeerAdvocate
    - df_brew: matched breweries from RateBeer and BeerAdvocate
    - df_ba: BeerAdvocate reviews for all of the matched beers
    - df_rb: RateBeer reviews for all of the matched beers
    - one for maps stuff
(No problem with data size since each filtered are under 200mb and fit into RAM.)
4. [ ] Create pipeline to filter relevant datasets, e.g. by minimum number of reviews or countries of interest.
5. [ ] Investigate the research questions (RQs). We conduct the analysis for RateBeer and BeerAdvocate separately, and then compare the results. This allows us to check if our findings are robust between the two platforms, controlling for an extra source of variation due to differing user bases.
    - [ ] Investigate RQ1 by viewing ratings for different beer styles for each country
    - [ ] Investigate RQ2 by seeing if general ratings differ per country. Then furthermore seeing if certain countries are more critical on certain rating aspects.
    - [ ] Investigate RQ3 by finding word frequencies and computing distances between the language corpuses i.e. reviews for each country. To achieve this, we will leverage existing methods and research such as [Ruzicka](https://github.com/mikekestemont/ruzicka), [PyStyl](https://github.com/mikekestemont/pystyl) or [word-cloud](https://github.com/amueller/word_cloud). Also investigate if there are differences for foreign and local reviews.
    - [ ] RQ4

6. [ ] Create a datastory and visualisations to communicate effectively our results. more idea
        

## Proposed timeline
1. Already for Project Milestone 2, we have completed T1, T2, T3, T4 [FINISH LIST]
2. 

## Organization within the team: A list of internal milestones up until project Milestone P3.
- 

## Questions for TAs (optional): Add here any questions you have for us related to the proposed project.
