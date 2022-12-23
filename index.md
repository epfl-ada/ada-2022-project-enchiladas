---
layout: default
title: Applied Beer Analysis
---

<!-- # TODO
- Motivate story - Interesting intro - blabla
- Once attention is grabbed, detail the research questions -->


<!-- # Introduction (Better title?) -->
## Doesn't everyone like beer?!
Beer is the world's most widely consumed alcoholic drink [2] and is the third-most consumed drink overall, after only water and tea. It is enormously popular - but not everybody likes beer the same way!

Users from beer rating websites (such as [BeerAdvocate](https://www.beeradvocate.com/) or [RateBeer](https://www.ratebeer.com/ratebeerbest/)) are certainly beer enthusiasts, but do their preferences vary from state to state or country to country?
To begin with our investigation, let's look at average ratings across the US states. You can use the menue to select which aspects of the beer is rated:

<!-- Map of states and their average rating -->
<iframe src="./Pages/States.html" title="States - Means" width="100%" frameborder="0" scrolling="no" height="800"></iframe>

They do differ quite a lot! Indeed, we observe that about **50% of the states pairs** actually have a significantly different rating distribution among all aspects.

## How about other countries?
The figure below shows the distribution of ratings among all aspects for the most occuring countries in the BeerAdvocate dataset.

<!-- Ratings per Country -->
<iframe src="./Pages/boxplots_of_aspects_for_all_countries.html" title="Ratings per Country" width="100%" frameborder="0" scrolling="no" height="800"></iframe>

Here, the results vary even more! We do observe that **80% of the country pairs** actually have a different rating distribution among all aspects.

_Can we trust this result?_
The limits of this initial naive analysis quickly show up. Availability of beers heavily depends on the geographical location. The following section assesses if this might be problematic to the quality of our analysis.


## Differences in beer quality between regions ?

The answer seems obvious! Although user might have personal preferences or different levels of criticism, beers also have intrisinc qualities which would be recognised amongst a majority of enthusiasts.

To illustrate this point, let's subset our reviews to the most rated beers in each country (and also states). They do differ quite a bit between each region. Can you recognise popular names in your country of origin?

<!-- Top beers per state -->
<iframe src="./Pages/states_beer_app.html" title="Top Beers per State and Country" width="100%" height=800 frameborder="0" scrolling="no"></iframe>

As expected, beer quality does influence the ratings a lot! Indeed, we found that **84% of the pairs** within the top beers have significant differences in their average rating. Similarly, when looking at most popular beers within each US states, a **vast majority** of the pairs do show significant differences in rating distribution (92% in the BeerAdvocate dataset and 75% in the RateBeer dataset).
These numbers have to be taken with a grain of salt. We are not able to estimate a good global average for beers as a huge amount of ratings are from the US. However, it is now clear that both user's biases and differences in beer quality must be taken into account when comparing ratings behaviours for users of different regions.

<!-- Matthieus part  -->
## We need matching between reviews!
To allow for apple to apple comparisons, the ideal controlled experiment would ask consumers to rate a set of beers, randomly changing the labels to have isolate the effect of the user's behaviour. To mimic this in an observational study, we instead match reviews with a similar beer quality and a similar user's level of criticism (more details can be found in the "method" tab of the next figure, or in our [github repo](https://github.com/epfl-ada/ada-2022-project-enchiladas)).

Now that we have established a better method, let's apply it to answer some interesting questions!

## Does beer from home always taste better?

We have already seen different countries rate beers differently. If each country has its own way of enjoying beer, it is also interesting to see how they perceive foreign beverages. In particular, can we observe some consumer preferences towards local products as compared to foreign ones?

To isolate this effect, we match each local review to a similar foreign review. The following figure observes the distribution of the two groups:
<!-- Plot with distributions (Rating distribution per group - foreign vs local) -->

<iframe src="./Pages/home_bias.html" title="Distribution of local and foreign reviews" width="100%" height="800" frameBorder="0" scrolling="no"></iframe>

Once matched, the difference of distribution of rating between local and foreign reviews is almost visually indistinguishable. Indeed, the users only seem to give on average 0.018 (between countries) or 0.014 (between states), meaning they **prefer local beers compared to foreign ones**. However, despite being small, the difference is still significant as shown by the small p-value (1.9e-12 and 3.9e-9).

## Simpson's paradox?

Since our dataset has a majority of reviews written by Americans, the observations might not be representative of users in other countries. To investigate if there is more information hidden under the hood, let's have a look at the home bias when grouping users by country for the top 10 countries.

<!-- Plot with confidence intervals  -->

<iframe src="./Pages/home_bias_countries.html" title="Distribution of local and foreign reviews" width="100%" height="650" frameBorder="0" scrolling="no"></iframe>

Interestingly, there now seem to be a **majority of countries showing a more pronounced negative home bias,** meaning that user would actually **rate foreign beers higher** than local beers. This contradiction is a manifestation of [Simpson's paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox): when users are combined, the majoritarian country of the dataset (US) masks the effect of all others. Let's also note that Belgium, which is quite renowned for its beer [3], seem to still prefer their local beers. For US states, we also have disparate results with some states showing positive and others negative biases towards their home beers. However, here we don't see any global trend.

These results are interesting but it is hard to find a good foolproof explanation behind each user's behaviour. Indeed, favoritism towards local or foreign brands has been extensively studied in social sciences. For example, Balabanis _et al._ [4] summarise the possible explanations to five effects, which will either bias the consumer toward choosing local brands or, conversely, toward foreign ones. Given our lack of data on consumer profile, it is not possible to further quantify each effect. Our analysis can therefore concludes that the composition of all these effects leads to reviewers of certain country having a more or less pronouced preference for local (the US or Belgium) or foreign beers (England, Australia, Spain, etc.).

<!-- Oisins part -->
<!-- Motivate look at language:
Is this bias reflected in language?
Foreign vs local beers (Oisin will do) [give numbers]
-> Move to more general language analysis
Do countries talk differently? -->

## Do people also talk about foreign and local beer differently?

Clearly, a difference exists between foreign and local beers in terms of their ratings. But does this also manifest itself in the language used in these beer reviews?

Firstly, we need to tackle a simpler question: how can we even quantify whether or not people talk differently?

To do this, we leverage state-of-the-art machine learning methods which have been specifically developed for textual analyses like this (word2vec [5], GloVe [6] and FastTest [7, 8]). Using these methods, this allows us to find how close two texts are - which is precisely what we need!
Also, we use an ensemble model, meaning we can even track how much variation we have in our predictions.

Applying this to foreign versus local beer, we can find that there is a difference of μ ± σ = 0.38 ± 0.17. This agrees with the previous result i.e. **people do treat foreign and local beers differently**, but it is very uncertain.

_What if we just look at the US?_
Could it be that the non-Americans sound more American when talking about American beer rather than their local beers? The answer surprisingly is no! In fact, reviews by non-Americans of American beers are **even further away** (18.5% further) from reviews of Americans than reviews of non-Americans of their own local beers.

Why is this? First of all, our results are quite uncertain - meaning that we cannot rule out the opposite trend. _BUT,_ the current trend can nonetheless be explained:

- It may be that users which review foreign beers constitute an entirely different social class to other users. Although the website itself attracts a very specific type of person, it is obvious that within this there are also wildly differing demographics. For example, perhaps it may be that users who review foreign beers are richer and travel more, constituting an upper class. Via education differences etc., this class likely also have a different means of talking as compared to the general populations. Thus, this may be a possible explanation.
- There may be a psychological effect. When talking about foreign beers, it may that users exaggerate their own speaking mannerisms in order to distinguish themselves. For some, their language can be a mark of pride. As an example, the English are unlikely to ever use the word "soccer" in any case, but they may make a point of explicitly using or exaggerating the word "football" when the conversation is about American culture/people/things.

## Do beer talk also differ from region to region?

Language and culture are interlinked, so we know **different cultures have different ways of expressing their preferences** for beers. Let's look at an example!

<!-- the beer wordcloud  -->
<iframe src="./Pages/wordcloud.html" title="Wordcloud" width="100%" height="800" frameBorder="0" scrolling="no"></iframe>

A canonical example of differences between English usage among countries is American versus British English. Even at a cursory glance, we can already see the spelling differences occurring - American spellings of "color" and "flavor" versus British spellings of "colour" and "flavour".

However, we also see some cultural difference arising too!

There is also more prominence in the American wordcloud for positive words - we can see "great", "good", "really", "nice", "much", "good", "well", "nice" all occuring very often whereas in the English wordcloud these occur less frequently. This is a known difference between British and American culture - the British are known for having a "stiff upper lip" and Americans are known for being very optimistic and upbeat (e.g. see [9] and [10]).

## Can we group countries that talk beer in a similar way?

<!-- Some differences, can we cluster countries or states based on language? -->

Since users of different countries write reviews differently, we can try to use those variation in language to cluster them together. With this approach, do we find that geographically similar countries also talk similarly?

<!-- dendrogram across countries -->
<iframe src="./Pages/dendrogram.html" title="Dendrogram" width="100%" height="700" frameBorder="0" scrolling="no"></iframe>

We can see that in general, **geographic proximity does not entirely determine how similarly countries talk**. This does make sense, since geographic proximity also does not determine how similarly two people sound either. For example, it is known that the United Kingdom has the largest variation of accents of any country in the world [11, 12]. In fact, we see exactly this effect, with the UK nations not being particularly close. On the other hand, Canada and the United States are very similar as we can imagine they share a lot of common culture.

## How about for states?

We can conduct the same analysis for US states to find the linguistic similarity between them. We plot these on a 2D grid, with a marker size indicating the number of reviews of the state.

<!-- network of states  -->
<iframe src="./Pages/network.html" title="Network" width="100%" height="770" frameBorder="0" scrolling="no"></iframe>

We can also see again that geographic proximity does not entirely predict linguistic similarity. Nonetheless, we do see some clusters of states that occur. For example, Pennsylvania and New York are extremely close in both the RateBeer and BeerAdvocate datasets.

We also note that states with more reviews are quite similar, with most of the outliers being states with few reviews - for example, South Dakota is an outlier in both datasets due to having very few reviews. We do note that peripheral states such as Hawaii are less linguistically similar to others, but in general states are similar. This points to the fact that United States is relatively homogeneous in general in its language usage.

<!-- Final parts and takeaways -->
<!-- Key bullet points and takeaways (Table) -->
# So, what insights have we garnered?
- Users of different regions rate beers differently.
- Those variations are however also influenced by the user criticism levels and the beer quality.
- There is therefore a need to account for differences in user and beer biases when comparing ratings, using 1-to-1 review matching.
- Once those differences are accounted, there is a small but significant home bias amongst beer reviews.
- The trend is reversed if we look at home bias on a per country basis (example of Simpson's paradox).
- Countries talk differently about beers.
- Cultural differences are complicated and are not entirely determined by geographical proximity.

# Where can I find out more?
Our code is publicly available [here](https://github.com/epfl-ada/ada-2022-project-enchiladas), and detailed explanations of the methods employed can be found in the notebooks. We have also included a summarised methods tab for each figure to explain our approach.

# References

[1] "Psychology and culture", Annual Review of Psychology, [Lehman, D. R., Chiu, C. Y., & Schaller, M. (2004)](https://www2.psych.ubc.ca/~schaller/LehmanChiuSchaller2004.pdf)

[2] "European Beer Statistics", European Beer Guide, [Pattinson, R. (2005)](http://www.europeanbeerguide.net/eustats.htm#production)

[3] "The 13 Best Beer Countries in the World, Ranked", Thrillist, [Mack, Z. (2015)](https://www.thrillist.com/drink/nation/the-best-beer-countries-in-the-world)

[4] "Favoritism Toward Foreign and Domestic Brands: A Comparison of Different Theoretical Explanations", Journal of International Marketing, [Balabanis, G., Stathopoulou, A., & Qiao, J. (2019)](https://openaccess.city.ac.uk/id/eprint/23521/)

[5] "Efficient estimation of word representations in vector space", Arxiv, [Mikolov, T., Chen, K., Corrado, G. and Dean, J. (2013)](https://arxiv.org/abs/1301.3781)

[6] "GloVe: Global Vectors for Word Representation", Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, [Pennington, J., Socher, R. and Manning, C.D. (2014)](https://nlp.stanford.edu/pubs/glove.pdf)

[7] "Enriching Word Vectors with Subword Information", Transactions of the Association for Computational Linguistics , [Bojanowski, P., Grave, E., Joulin, A. and Mikolov, T. (2017)](https://arxiv.org/abs/1607.04606v1)

[8] "Bag of Tricks for Efficient Text Classification", Arxiv, [Joulin, A., Grave, E., Bojanowski, P. and Mikolov, T. (2016)](https://arxiv.org/abs/1607.01759)

[9] "Brits DO have a stiff upper lip: Americans more optimistic and romantic than Britons", MailOnline, [Bond, A. (2013)](https://www.dailymail.co.uk/news/article-2346351/Brits-DO-stiff-upper-lip-Americans-optimistic-romantic-Britons.html)

[10] "What Makes Americans So Optimistic?", The Atlantic, [Keller, J. (2015)](https://www.theatlantic.com/politics/archive/2015/03/the-american-ethic-and-the-spirit-of-optimism/388538/)

[11] "Why are there so many regional accents in the UK, in comparison to other English-speaking countries?", The Guardian, [Various Contributors](https://www.theguardian.com/notesandqueries/query/0,5753,-18336,00.html#:~:text=SEMANTIC%20ENIGMAS-,Why%20are%20there%20so%20many%20regional%20accents%20in%20the%20UK,to%20other%20English%2Dspeaking%20countries%3F&text=Until%20the%201930s%20communication%20between,means%20of%20retaining%20an%20identity)

[12] "Accents in Britain", AccentBiasBritain, [Unknown](https://accentbiasbritain.org/accents-in-britain/)