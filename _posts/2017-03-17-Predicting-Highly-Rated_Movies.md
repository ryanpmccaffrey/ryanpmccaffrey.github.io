---
layout: post
image: '/img/imdb.png'
title: "Is This Movie in the Top 250?"
subtitle: "Category: APIs + PCA + Random Forest (10 min read)"
---

Everybody loves movies, but what makes a truly great one?  In this week's project I will explore exactly that.  I'll look at things like leading actors, directors, plot summary, genre, title, runtime, etc., and see if these features can be used to predict whether or not a movie is in the all-time top 250.  To do this I'll be using the Internet Movie Database (IMDB), the go-to online database of movies, television, and videogames. 

This week we will use two IMDB APIs ([ImdbPie](https://github.com/richardasaurus/imdb-pie) and [OMDB](http://www.omdbapi.com/)) to access movie data to predict whether or not a movie is in the top 250 of all-time movies.  We will build a random forest model and examine its feature importance to determine what factors contribute to a movie becoming highly rated.  As part of our data cleaning process we will also perform textual analysis on the movie plot summaries and use principal component analysis (PCA) to reduce the number of plot summary feature dimensions. 


![jpg](/img/project-6_files/imdb_top_250.jpg)
*Figure 1 - Collection of posters of all-time IMDB top 250 films. ([Source](http://pre07.deviantart.net/6cd1/th/pre/f/2011/035/6/4/imdb_top_250_movie_poster_by_saxon1964-d38rnod.jpg))*

This project will focus around the following sections and is intended for a technical audience:

- APIs and data collection
- Data cleaning
- PCA on plot summary
- Random forest and other models
- Conclusion


We make the following risks and assumptions about the data and our model:

- IMDB rating methodology has not changed since 1990
- Model only applies to feature films from 1990 to 2016

## APIs and data collection


```python
# Import packages
import pandas as pd
import numpy as np
import random
from imdbpie import Imdb
import matplotlib.pyplot as plt
import pickle
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
import re
%matplotlib inline

# Instantiate ImdbPie API to proxy requests
imdb = Imdb()
imdb = Imdb(anonymize=True) 

# Retrieve top 250 movie data and plot ratings
tmp = imdb.top_250()
top250 = pd.DataFrame(tmp)
top250['rating'].hist();
```


![png](/img/project-6_files/project-6_2_0.png)


The ImdbPie API conveniently let's us easily access movie data of the top 250 all-time movies as rated by IMDB users (ranking can be found [here](http://www.imdb.com/chart/top)).  The above figure is a rating histogram of the top 250 movies.  Since we want to compare (and predict) movies from both inside and outside the top 250 we need a way to acquire more movie data.  One approach would be to webscrape IMDB's website, retrieve a large list of movie IDs and then query the IMDB Pie API with those IDs.  However, that approach might be flawed in the sense that the data we collect on movies outside the top 250 may not represent a random subset of movies outside the top 250 (it would be biased, for example, based on the particular pages we webscraped).  To ensure we generate a truly random movie subset we will instead ping the ImdbPie with randomly generated IMDB IDs.  Then we will compare those observations with the top 250 movies to see if we can establish accurate predictions to differentiate between the two classes (i.e., movies inside and outside the top 250 ranking).

Because people's perception of movies and how they rate them sometimes change over time, we will limit our comparison to films released in 1990 or later. This will give us a more consistent basis of comparison.


```python
# Initialize empty lists
rating = []
imdb_id = []

# Loop through randomly generated ids and ping API
count = 0
for i in range(15000):
    # Generate random id
    id = 'tt' + "%07d" % (random.randint(1,5999999),)

    # Query imdb database to retrieve data
    try:
        title_id = imdb.get_title_by_id(id)
        if (title_id.type == 'feature') and (title_id.rating >= 0 and title_id.rating <= 10) and (title_id.year >=1990) and (len(title_id.plots)>0):
            rating.append(title_id.rating)
            imdb_id.append(title_id.imdb_id)
        else:
            pass
    except:
        pass
    # Print progress
    if count%10 == 0:
        print i,
    count += 1
    
# Convert lists to dataframe
df = pd.DataFrame({'rating': rating, 'imdb_id': imdb_id})
```

After each session of say 5,000 pings to the ImdbPie API we should save the dataframe as a pickle object so we can use it later.


```python
# Save dataframe for later use
pd.to_pickle(df,'ping')
```


```python
# Create large list of imdb ids
movie_ids = list(pd.read_pickle('ping1')['imdb_id']) + list(pd.read_pickle('ping2')['imdb_id']) + list(pd.read_pickle('ping3')['imdb_id']) + list(top250[top250['year'].astype(int)>=1990]['tconst'])

# See if there are any duplicate movies
print len(movie_ids)
print len(pd.Series(movie_ids).drop_duplicates())
```

    341
    341


Once we have a list of all the IMDB movie IDs we can then use it to call the OMDB API to obtain nicely formatted movie data.


```python
# Set OMDB API url
api_url = "http://www.omdbapi.com/?i={}&plot=full&r=json"

# Function to access movie data with get request
def get_content(movie_ids):
    r = requests.get(api_url.format(movie_ids))
    tmp = json.loads(r.text)
    return tmp

# Create dataframe to capture movie data
df = pd.DataFrame([get_content(i) for i in movie_ids])

# Function to drop row with text value 'N/A'
def drop_na(x):
    if x == 'N/A':
        return None
    else:
        return x

# Drop row with value 'N/A' and plot histogram
df['imdbRating'] = df['imdbRating'].apply(drop_na)
df['imdbRating'].astype(float).hist();
```


![png](/img/project-6_files/project-6_10_0.png)


We find that the randomly generated movie rating data is normally distributed with a mean of around 6.5, whereas the spike above 8 is attributed to the rating data of movies in the top 250.

## Data cleaning

Once we have the data, we need to clean it up for our model.  We do things like: (1) dropping irrelevant, redundant or highly collinear columns, (2) creating a binary target variable for whether or not a movie is in the top 250, (3) reformating the 'Runtime' column, (4) creating new features for Oscar wins and nominations, (5) creating dummy variables for the 'Rated' column, (6) converting the 'imdbVotes' column to float, and (7) applying a term frequency–inverse document frequency (tf-idf) vectorizer to the text columns.


```python
df.columns
```




    Index([u'Actors', u'Awards', u'Country', u'Director', u'Genre', u'Language',
           u'Metascore', u'Plot', u'Poster', u'Rated', u'Released', u'Response',
           u'Runtime', u'Title', u'Type', u'Writer', u'Year', u'imdbID',
           u'imdbRating', u'imdbVotes'],
          dtype='object')




```python
# Function to create binary variable of whether or not movie is in top 250
def top_250(x):
    if x in list(top250['tconst']):
        return 1
    else:
        return 0

df['Top250'] = df['imdbID'].apply(top_250)
```


```python
# Drop columns that are either irrelevant, redundant or highly collinear with rating score
df = df.drop(['Metascore','Poster','Response','Released','Type','imdbID','imdbRating'], axis=1)
```


```python
# Function to reformat runtime column
def convert_runtime(x):
    if x == 'N/A':
        return 0
    else:
        return int(x.split(' ')[0])

df['Runtime'] = df['Runtime'].apply(convert_runtime)

# Create column for number of Oscar nominations
def noms(x):
    regex = re.compile('Nominated for ([0-9][0-9]?) Oscar')
    if regex.findall(x):
        return regex.findall(x)[0]
    else:
        return 0

df['Nominations'] = df['Awards'].apply(noms)

# Create column for number of Oscar nominations
def wins(x):
    regex = re.compile('Won ([0-9][0-9]?) Oscar')
    if regex.findall(x):
        return regex.findall(x)[0]
    else:
        return 0

df['Oscars'] = df['Awards'].apply(wins)

rated_df = pd.get_dummies(df['Rated'].astype('category'), drop_first=True)
```

For text columns such as 'Actors', 'Country', 'Director', 'Genre', 'Language', 'Title' and 'Plot' we used a term frequency–inverse document frequency (tf-idf) vectorizer to extract the most important words or terms.  For all columns except for 'Plot' we sorted the words by their tf-idf score and then selected the highest 5-10 words per column.  For the 'Plot' column we found that approach not to be suitable due to the vastness of the number of words (over 5,000 words) so we decided to reduce the number of features by applying PCA to the tf-idf scores of the 'Plot' column.


```python
# Vectorize column using tf-idf
cvec = TfidfVectorizer(ngram_range=(2,2),stop_words='english')
cvec.fit(df['Actors'])
text_df  = pd.DataFrame(cvec.transform(df['Actors']).todense(),
             columns=cvec.get_feature_names())
cols = text_df.sum().sort_values(ascending=False).head(5).index.values
actors_df = text_df[cols]
text_df.sum().sort_values(ascending=False).head(5)
```




    aamir khan           1.896803
    leonardo dicaprio    1.849741
    tom hanks            1.836838
    christian bale       1.339540
    kevin spacey         1.314475
    dtype: float64




```python
# Vectorize column using tf-idf
cvec = TfidfVectorizer(stop_words='english')
cvec.fit(df['Country'])
text_df  = pd.DataFrame(cvec.transform(df['Country']).todense(),
             columns=cvec.get_feature_names())
cols = text_df.sum().sort_values(ascending=False).head(5).index.values
country_df = text_df[cols]
text_df.sum().sort_values(ascending=False).head(5)
```




    usa       149.955674
    uk         31.901690
    india      23.318646
    canada     17.991814
    france     17.819811
    dtype: float64




```python
# Vectorize column using tf-idf
cvec = TfidfVectorizer(ngram_range=(2,2),stop_words='english')
cvec.fit(df['Director'])
text_df  = pd.DataFrame(cvec.transform(df['Director']).todense(),
             columns=cvec.get_feature_names())
cols = text_df.sum().sort_values(ascending=False).head(10).index.values
director_df = text_df[cols]
text_df.sum().sort_values(ascending=False).head(10)
```




    christopher nolan    7.000000
    martin scorsese      5.000000
    quentin tarantino    4.860305
    steven spielberg     4.000000
    peter jackson        3.000000
    david fincher        3.000000
    clint eastwood       3.000000
    hayao miyazaki       3.000000
    frank darabont       2.000000
    guy ritchie          2.000000
    dtype: float64




```python
# Vectorize column using tf-idf
cvec = TfidfVectorizer(stop_words='english')
cvec.fit(df['Genre'])
text_df  = pd.DataFrame(cvec.transform(df['Genre']).todense(),
             columns=cvec.get_feature_names())
cols = text_df.sum().sort_values(ascending=False).head(6).index.values
genre_df = text_df[cols]
text_df.sum().sort_values(ascending=False).head(6)
```




    drama        107.904601
    comedy        56.821567
    thriller      36.720006
    crime         35.093919
    action        26.496740
    adventure     25.953811
    dtype: float64




```python
# Vectorize column using tf-idf
cvec = TfidfVectorizer(stop_words='english')
cvec.fit(df['Language'])
text_df  = pd.DataFrame(cvec.transform(df['Language']).todense(),
             columns=cvec.get_feature_names())
cols = text_df.sum().sort_values(ascending=False).head(5).index.values
lang_df = text_df[cols]
text_df.sum().sort_values(ascending=False).head(5)
```




    english     182.025720
    french       23.667448
    spanish      21.226335
    japanese     17.362435
    hindi        15.957968
    dtype: float64




```python
# Vectorize column using tf-idf
cvec = TfidfVectorizer(stop_words='english')
cvec.fit(df['Title'])
text_df  = pd.DataFrame(cvec.transform(df['Title']).todense(),
             columns=cvec.get_feature_names())
cols = text_df.sum().sort_values(ascending=False).head(5).index.values
title_df = text_df[cols]
text_df.sum().sort_values(ascending=False).head(5)
```




    story     2.642579
    day       2.484945
    father    2.352324
    love      2.097245
    city      2.015957
    dtype: float64




```python
df['imdbVotes'] = df['imdbVotes'].str.replace(',','').replace('N/A',0).astype(int)
```

## PCA on plot summary

The reason we perform PCA only on the 'Plot' summary column is because we want to preserve the interpretability of our model.  If we apply PCA to the entire dataframe then we lose any possible insights into which features actually impact whether or not a movie is in the top 250.  By applying PCA only to the 'Plot' summary column, we only lose the interpretation of which plot keywords most affect movie ranking.  This approach is especially preferred if intuitively we believe the text of the plot summary will have little impact in predicting whether or not a movie is in the top 250 of all-time movies.


```python
# Vectorize column using tf-idf
cvec = TfidfVectorizer(stop_words='english')
cvec.fit(df['Plot'])
text_df  = pd.DataFrame(cvec.transform(df['Plot']).todense(),
             columns=cvec.get_feature_names())

# Center and scale the feature data
x_standard = StandardScaler().fit_transform(text_df)

# Instantiate and fit PCA model
pca = PCA(n_components=40)
X_pca = pca.fit_transform(x_standard)

# Convert output to dataframe
plot_df = pd.DataFrame(X_pca)

# Print percentage of variance explained
pca.explained_variance_ratio_.sum()
```




    0.25209239865526256



Performing tf-idf on the plot summary column yields scores for 5,587 unique words.  Using PCA we reduced that feature space down to 40 dimensions, however, we were only able to capture 25% of the variance explained.  Because we don't have that many observations (341) we want to make sure that we limit the size of our feature space.

Now that we sufficiently reduced the number of features related to our 'Plot' summary tf-idf analysis we can then concatenate and normalize that dataframe with the other features to create a large dataframe for input into our model.


```python
X = pd.concat([rated_df,actors_df,country_df,director_df,genre_df,lang_df,title_df,df[['Runtime','Year']]], axis=1)
Xt = StandardScaler().fit_transform(X)
y = df['Top250']
```

## Random forest and other models

The binary target variable of our model is whether or not a movie is in the all-time top 250 ranking.  The feature inputs to our model are all the cleaned up dataframes we put together in the previous sections (including things like director, actors, runtime, year, etc.).


```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

cv = StratifiedKFold(n_splits=5, random_state=21, shuffle=True)

dt = DecisionTreeClassifier(class_weight='balanced')
s = cross_val_score(dt, X, y, cv=cv)
print("{} Score:\t{:0.3} ± {:0.3}".format("Decision Tree", s.mean().round(3), s.std().round(3)))

rf = RandomForestClassifier(n_estimators=1000,class_weight='balanced')
rf.fit(Xt,y)
s = cross_val_score(rf, Xt, y, cv=cv)
print("{} Score:\t{:0.3} ± {:0.3}".format("Random Forest", s.mean().round(3), s.std().round(3)))

et = ExtraTreesClassifier(n_estimators=1000,class_weight='balanced')
s = cross_val_score(et, Xt, y, cv=cv)
print("{} Score:\t{:0.3} ± {:0.3}".format("Extra Trees", s.mean().round(3), s.std().round(3)))

gb = GradientBoostingClassifier(n_estimators=1000)
s = cross_val_score(gb, Xt, y, cv=cv)
print("{} Score:\t{:0.3} ± {:0.3}".format("Gradient Boosting", s.mean().round(3), s.std().round(3)))
```

    Decision Tree Score:	0.88 ± 0.034
    Random Forest Score:	0.906 ± 0.02
    Extra Trees Score:	0.892 ± 0.03
    Gradient Boosting Score:	0.883 ± 0.031


After applying a stratified 5-fold cross validation to four different models (i.e., decision tree, random forest, extra trees and gradient boosting) we find that the random forest and gradient boosting models perform the best based on their mean accuracy scores.  We are able to achieve accuracy scores of around 0.89 with both models.  This is a significant improvement over the baseline accuracy score of 0.62, which would be achieved if we predicted that none of the movies were in the top 250 (recall: only 130 of the 341 observations were actually movies from the top 250).  

You may have noticed during the data cleaning phase we took time to reformat the 'imdbVote' column and developed a method to extract from the 'Awards' column data pertaining to both Oscar wins and nominations.  However, these features were not passed into our model because they would not be known prior to a movie's release.  If we perform an ex posto facto analysis and include these features in our model we find that the model accuracy jumps to around 0.98 accuracy with a significant reduction in the variance.  Lastly, you will also notice that the 40 features we obtained by applying PCA to the plot summary column were not included in the final model.  We chose to remove these features as they did not improve the accuracy of the model. This was not a surpise given that the 40 features only accounted for 25% of the variance explained in the plot summary tf-idf vectorizer scores.  


```python
# Print feature importance of random forest model
pd.DataFrame(zip(X.columns,rf.feature_importances_), columns=['feature','importance']).sort_values('importance', ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44</th>
      <td>Runtime</td>
      <td>0.248592</td>
    </tr>
    <tr>
      <th>0</th>
      <td>N/A</td>
      <td>0.219319</td>
    </tr>
    <tr>
      <th>4</th>
      <td>R</td>
      <td>0.058621</td>
    </tr>
    <tr>
      <th>28</th>
      <td>drama</td>
      <td>0.055154</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Year</td>
      <td>0.049357</td>
    </tr>
    <tr>
      <th>29</th>
      <td>comedy</td>
      <td>0.042571</td>
    </tr>
    <tr>
      <th>34</th>
      <td>english</td>
      <td>0.042436</td>
    </tr>
    <tr>
      <th>33</th>
      <td>adventure</td>
      <td>0.033271</td>
    </tr>
    <tr>
      <th>13</th>
      <td>usa</td>
      <td>0.031426</td>
    </tr>
    <tr>
      <th>31</th>
      <td>crime</td>
      <td>0.031318</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PG-13</td>
      <td>0.028411</td>
    </tr>
    <tr>
      <th>30</th>
      <td>thriller</td>
      <td>0.015834</td>
    </tr>
    <tr>
      <th>35</th>
      <td>french</td>
      <td>0.015021</td>
    </tr>
    <tr>
      <th>14</th>
      <td>uk</td>
      <td>0.014816</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PG</td>
      <td>0.012046</td>
    </tr>
    <tr>
      <th>32</th>
      <td>action</td>
      <td>0.010096</td>
    </tr>
    <tr>
      <th>41</th>
      <td>father</td>
      <td>0.009308</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NOT RATED</td>
      <td>0.008818</td>
    </tr>
    <tr>
      <th>17</th>
      <td>france</td>
      <td>0.008022</td>
    </tr>
    <tr>
      <th>38</th>
      <td>hindi</td>
      <td>0.007184</td>
    </tr>
    <tr>
      <th>16</th>
      <td>canada</td>
      <td>0.005716</td>
    </tr>
    <tr>
      <th>36</th>
      <td>spanish</td>
      <td>0.005669</td>
    </tr>
    <tr>
      <th>8</th>
      <td>aamir khan</td>
      <td>0.005539</td>
    </tr>
    <tr>
      <th>15</th>
      <td>india</td>
      <td>0.005187</td>
    </tr>
    <tr>
      <th>37</th>
      <td>japanese</td>
      <td>0.004137</td>
    </tr>
    <tr>
      <th>10</th>
      <td>tom hanks</td>
      <td>0.003431</td>
    </tr>
    <tr>
      <th>39</th>
      <td>story</td>
      <td>0.003225</td>
    </tr>
    <tr>
      <th>18</th>
      <td>christopher nolan</td>
      <td>0.003176</td>
    </tr>
    <tr>
      <th>7</th>
      <td>UNRATED</td>
      <td>0.003038</td>
    </tr>
    <tr>
      <th>27</th>
      <td>guy ritchie</td>
      <td>0.002562</td>
    </tr>
    <tr>
      <th>20</th>
      <td>quentin tarantino</td>
      <td>0.002423</td>
    </tr>
    <tr>
      <th>25</th>
      <td>hayao miyazaki</td>
      <td>0.002081</td>
    </tr>
    <tr>
      <th>12</th>
      <td>kevin spacey</td>
      <td>0.001696</td>
    </tr>
    <tr>
      <th>19</th>
      <td>martin scorsese</td>
      <td>0.001691</td>
    </tr>
    <tr>
      <th>9</th>
      <td>leonardo dicaprio</td>
      <td>0.001413</td>
    </tr>
    <tr>
      <th>24</th>
      <td>clint eastwood</td>
      <td>0.001095</td>
    </tr>
    <tr>
      <th>40</th>
      <td>day</td>
      <td>0.000932</td>
    </tr>
    <tr>
      <th>11</th>
      <td>christian bale</td>
      <td>0.000837</td>
    </tr>
    <tr>
      <th>21</th>
      <td>steven spielberg</td>
      <td>0.000821</td>
    </tr>
    <tr>
      <th>23</th>
      <td>david fincher</td>
      <td>0.000821</td>
    </tr>
    <tr>
      <th>42</th>
      <td>love</td>
      <td>0.000714</td>
    </tr>
    <tr>
      <th>22</th>
      <td>peter jackson</td>
      <td>0.000591</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TV-14</td>
      <td>0.000531</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TV-PG</td>
      <td>0.000443</td>
    </tr>
    <tr>
      <th>43</th>
      <td>city</td>
      <td>0.000389</td>
    </tr>
    <tr>
      <th>26</th>
      <td>frank darabont</td>
      <td>0.000221</td>
    </tr>
  </tbody>
</table>
</div>



The above figure shows the relative importance of all features included in our random forest model.  One of the key takeaways is that actors and directors may not be as important to creating all-time hit movies as we might think since we find them concentrated toward the middle and bottom of the feature importance ranking. Another key observation is that drama and rated R movies are high on the list of feature importance.  This makes intuitive sense as we see many Oscar nominee films come from the drama genre and are often rated R.  What's most interesting about the feature importance ranking is that it has highlighted many shortcomings in our data.  The two most important features are 'Runtime' and a movies rated 'N/A'.  In general, all of the movies in the top 250 had "good" data.  Good in the sense that all the fields were appropriately filled out and the data seemed to be accurate.  The same was not the case with the randomly collected movie data.  In some cases entries in the runtime column were either not input or input as zero and entries in the rated column were input as 'N/A'.  Because these entries were only lacking for movies not included in the top 250 (perhaps the data for some of the less famous / more obscure films was scarce) it was easy for the random forest model to use these features as key differentiators.  Whereas features like 'drama' and 'R' were likely indicative of movies in the top 250, features like 'N/A' and 'Runtime' were likely indicative of movie not in the top 250.

## Conclusion

In summary, we used two IMDB APIs to succesfully collect data for 341 films, of which 130 were ranked in the all-time top 250.  We performed feature engineering and cleaning on the data, which included textual analysis using tf-idf as well as PCA to reduce the number of features. The random forest and gradient boosting models achieved the highest accuracy scores using 5-fold cross validation (0.89).  We did not optimize our models, but in the future one could apply GridSearchCV to hyper-tune them to achieve greater accuracies. By including additional features not known a priori, such as Oscar wins or nominations, we demonstrated even higher accuracies of around 0.98 can be achieved.  Keywords from the plot summaries were not included in the final model as they did not improve accuracy.  Key features obtained from the random forest model were whether or not a movie had an audience rating and the runtime of the film.  Very short or long films tended not to be included in the top 250.  We also found that films without a proper audience rating were a dead giveaway that they were not a top 250 movie.
