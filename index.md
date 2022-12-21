---
layout: default
title: Applied Beer Analysis
---


# TODO
- Motivate story - Interesting intro - blabla
- Once attention is grabbed, detail the research questions


# Introduction (Better title?)
First have a look at means per state... Some preliminary geographical difference visible?

<!-- Map of states and their average rating -->
<iframe src="./Pages/States.html" title="States - Means" width="100%" height="800" frameBorder="0"></iframe>

How about per country?

<!-- Distribution of ratings per country -->
<iframe src="./Pages/ratings_countries_app.html" title="Countries - Distribution" width="100%" height="800" frameBorder="0"></iframe>

Report some numbers and significance ratios.. (Kasimir will do)

Some differences occur! Why?
Attempt to remove user bias by rescaling the ratings per user. (Kasimir will do) TODO: method for Countries-dist -> describe rescaling
Somewhat fails (again report numbers using a table!)
Maybe due to the beers rated by country/state? 
Next idea: look at most rated beers per state and country:

<!-- Top beers per state -->
<iframe src="./Pages/states_beer_app.html" title="Top beers per state" width="100%" height="800" frameBorder="0"></iframe>

<!-- Top beers per country -->
<iframe src="./Pages/boxplots_of_aspects_for_all_countries.html" title="Top beers per country" width="100%" height="800" frameBorder="0"></iframe>

Our data is skewed. Global average is not representative for the actual quality of beers.
Indeed, very different beers rated per country
Significant in global ratings for top beers versus countries/states (Kasimir will do) (Add a table!)

<!-- Matthieus part  -->

Motivate propensity analysis. 
We need to match on beers and users at the same time.
(Matrix factorization! - we'll see how detailed...)

<!-- Plot with distributions (Rating distribution per group - foreign vs local) -->

<iframe src="./Pages/home_bias.html" title="Distribution of local and foreign reviews" width="100%" height="800" frameBorder="0"></iframe>

The difference of distribution of rating between local and foreign reviews is almost indistinguishable. Indeed, the users only seem to give on average (MEAN DIFF) points more to local beers compared to reviews. However, despite being small, the difference is still significant as shown by the small p-value (P VALUE).

Since our dataset has a majority of reviews written by americans, the observation might not be representative of every country. In the following plot, let's look at the home bias when grouping user by country for the top 10 countries on the website to investigate if there are more information hidden under the hood.

<!-- Plot with confidence intervals  -->

<iframe src="./Pages/home_bias_countries.html" title="Distribution of local and foreign reviews" width="100%" height="800" frameBorder="0"></iframe>

Interestingly, there now seem to be a majority of countries showing negative home bias, meaning that user would actually rate higher foreign beers compared to local beers. This contradiction is a manifestation of Simpson's paradoxe: when users are combined, the majoritarian country (US) mask the effect of all other. Let's also note that Belgium, which is quite renowned for its beer, seem to still prefer their local beers.

Those result are interesting but it is hard to find a good explanation behind each user's behaviour. Indeed, favoritism towards local or foreign brands has been extensively studied in social sciences. In "Favoritism Toward Foreign and Domestic Brands: A Comparison of Different Theoretical Explanations. Journal of International Marketing", [Balabanis, G., Stathopoulou, A., & Qiao, J. (2019)](https://openaccess.city.ac.uk/id/eprint/23521/) summarize the possible explanation to five effects, which will either bias the consummer toward choosing local brands or , conversely, foreign ones. Given our lack of data on consummer profile, it seems however impossible to further quantify each effect. Our analysis therefor only render the composition of all effect.

<!-- Oisins part -->
Motivate look at language:
Is this bias reflected in language?
Foreign vs local beers (Oisin will do) [give numbers]
-> Move to more general language analysis
Do countries talk differently?

<!-- the beer wordcloud  -->
<iframe src="./Pages/wordcloud.html" title="Wordcloud" width="100%" height="1000" frameBorder="0"></iframe>

Discuss: why different...

Some differences, can we cluster countries or states based on language?

<!-- dendogram across countries -->
<iframe src="./Pages/dendogram.html" title="Dendogram" width="100%" height="800" frameBorder="0"></iframe>

Discuss

<!-- network of states  -->
<iframe src="./Pages/network.html" title="Network" width="100%" height="800" frameBorder="0"></iframe>

Discuss

<!-- Final parts and takeaways -->
Key bullet points and takeaways (Table)
- Countries rate differently (a lot of it is due to user bias)
- Countries rate different beers (they have different quality)
- Propensity matching (Matthieu: is there more to say?) 
- Countries talk differently about beer
- Cultural difference is not entirely geographical (network)






























Text can be **bold**, _italic_, or ~~strikethrough~~.

[Link to another page](./another-page.html).

```{python}
print("Hello Python!")
```

<iframe src="./pages/bignumber.html" title="Tutorials" width="100%" height="350" frameBorder="0"></iframe>

There should be whitespace between paragraphs.


There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.

# Header 1

This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.


<iframe src="./pages/new_rep.html" title="Tutorials" width="100%" height="800" frameBorder="0"></iframe>



## Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.



##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

![Regression](./images/discount_regression.PNG)


###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
