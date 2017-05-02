---
layout: post
image: '/img/renthop.png'
title: "Predicting Interest Levels in NYC Rental Properties"
subtitle: "Category: Kaggle Competition"
---


Finding a new NYC apartment: it's one of the most daunting parts of living in NYC.  And there are few things New Yorkers cherish more than their little shoebox they call home.  Although NYC is one of the priciest cities to live in, it certainly doesn't lack options.  For this week's project I'll turn my attention to a Kaggle competition where I'll be predicting the popularity of various NYC rental apartments based on listing information found on [Renthop.com](https://www.renthop.com/).

This post gives an outline of my approach to solving the [Two Sigma Connect: Rental Listing Inquiries](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries) Kaggle competition.  This competition was as rewarding as it was challenging.  It featured a very diverse dataset (including geospatial, image, text, and temporal data) and was co-sponsored by Two Sigma Investments and Renthop.com.  The goal of the competition was to predict the popularity of various NYC rental properties based on listing information found on Renthop.com.  The problem was a multi-class problem in which we were tasked with predicting the associated probabilities of an apartment listing receiving "low", "medium" or "high" levels of interest.  Submissions were evaluated using a multi-class logarithmic loss metric.

My final model consisted of an ensemble of gradient boosting (XGBoost), extremely randomized trees (ExtraTrees) and a two-level feedforward stacked metamodel (StackNet) that featured a wide variety of base learners.  In the end, I ranked 214 out of 2,489 competitors (top 9%).  You can find the final leaderboard [here](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/leaderboard).  

![png](/img/renthop_project/renthop_screenshot.png)

## About the Data

The target variable was called *interest_level* and consisted of three categories: *low*, *medium*, and *high*.  There were also 14 original input features: *bathrooms*, *bedrooms*, *building_id*, *created*, *description*, *display_address*, *features*, *latitude*, *listing_id*, *longitude*, *manager_id*, *photos*, *price*, and *street_address*.  Below is a depiction of the original input features.

## Feature Engineering

Because of the diversity of the dataset I thought feature engineering was going to be the key to differentiating my submissions.  My thinking was something along the lines of: there are only a finite number of models in a data scientist's toolbox, so the best way to achieve better performance is to input better data (aka spend a lot of time feature engineering).  More than half my time was spent feature engineering, in essence trying to tease out additional insights from the data that would hopefully add predictive power to my models.  The other half of my time was divided between the exploratory data analysis and the model selection/tuning phases.  

Below is an overview of some of the feature engineering techniques I used along with a brief discussion of their relative feature importance:

- **Geospatial density**: For every apartment listing I calculated the number of other apartment listings within a 5 block radius. My intuition was that denser areas will have more competition and thus lower overall interest. This was mostly true, but the feature importance wasn't as good as I hoped it would be.
- **Distance to city center**: I calculated the distance of each apartment to the center coordinates of NYC, which ended up being somewhere inside Central Park.  This feature was useful for the same reason listed above: the farther you get from the center of the city the less populated the area tends to become.  The data showed that less population 
- **Neighborhood clustering**: I used unsupervised clustering (KMeans) to create 40 different neighborhoods across NYC. I clustered solely on latitude and longitude and it worked fairly well actually. Using this I could compare apartment prices to the median apartment price for each neighborhood. These were in the top 15% of feature importance.  Prior attempts to use the Google API resulted in me getting kicked off (too many API calls in a single day) and I found that the Python Reverse Geocoder package was not accurate enough to discern neighborhood information within NYC.  See below for the KMeans clustering that was used as a proxy for neighborhood.  Spurious geospatial coordinates (e.g. latitude/longitude = 0) were imputed by calling the Google API with the street address and then using the latitude and longitude outputs to reclassify the geo-coordinates.  I constrained all remaining outliers to be within the 1st and 99th latitude/longitude percentiles, which essentially formed a box around NYC.


- **Datetime objects**: I broke the created column down into smaller, more relevant time objects such as week, day of week, month, hour, etc.  Minutes and year were dropped as they added no value.
- **Normalized parts of speech**: I looked at 18 different parts of speech (e.g., nouns, adverbs, conjuctions, etc.) on the description column and then normalized these by the total number of description words per row. Many of these were surprisingly high on my feature importance ranking, but adding them actually lowered my overall score. Perhaps this meant these features were highly correlated with other features. Anyway, I didn't explore that, I just dropped them and moved on. These were not included in my final model.
- **Sentiment analysis**: I calculated the negative, neutral, positive and compound sentiment analysis scores for each row in the description column. These gave decent feature importance, but not as high as I hoped for.
- **Tf-idf analysis**: I performed term frequency-inverse document frequency analysis on the description column to collect the most popular 100 words (using a minimum usage threshold of 20 and English stopping words).
- **Word counts**: This was a simple feature that I applied to both the description and features columns.  The number of words in the apartment description column as well as the number of features an apartment had were actually pretty good indicators of people's interest in a rental apartment.
- **Spam filter**: As part of an attempt to filter out bogus or spammy entries I implemented a spam filter that calculated the percentage of words in the description that were captitalized as well as a counter that tracked the number of exclamation points found in the description.  These features had average feature importance.
- **Homemade dictionaries**: After doing CountVectorizer on the features column to obtain the 200 most popular words, I used a homemade dictionary to try to group similar words/phrases (e.g., "doorman" and "door man"). I also created a dictionary of luxury words that were meant to flag luxury apartments. Both dictionaries made my score worse, so I dropped them.
- **Street-display address similarities**: I calculated similarity scores between the street address and display address. What I found was interesting: the more dissimilar the addresses were the more likely they would received "high" interest.
- **Flagging street directions and type**: I created flags for "East","West","North" and "South" directions as well as "Street" and "Avenue" from the street address column.  These flags added very little value to the model.
- **Flagging "bad" photos**: I noticed each photo link had the listing_id in it. However, several (maybe 500 or so) of the photos actually referenced different listings, which means the photos were pointing to the wrong apartment. This ranked poorly in feature importance because of the relatively rare occurrences of "bad" photos.
- **Flagging "weird" prices**: I defined a "weird" price as anything that didn't end in a 0, 5 or 9. My intuition here was that people may be turned off to an apartment if the monthly rent is somewhat random, for example, \$2171 as opposed to \$2100. This feature ended up having basically no importance to the model.
- **"Price per" features**: Features derived from the price (e.g., price per bedroom, price to median price of a neighborhood, etc.) really moved the needle in terms of improving model performance.  The trick was using the price feature to create valuable insights.  For example, if we calculate the ratio of the price of an apartment to the median price of all other apartments in the building we can create a sense of whether or not that particular listing is a "good deal".  We can also use the ratio of price to number of bedrooms or bathrooms as a proxy for estimating the price per square foot of the apartment.
- **Photo metadata**: I extracted simple 
- **Manager and building "skill"**: 


In the final analysis I was able to transform the original 14 features into 300+ new features that I input into my models.

## Model Selection

## Conclusion