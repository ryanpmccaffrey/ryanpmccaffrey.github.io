---
layout: post
image: '/img/indeed.png'
title: "Predicting High/Low Data Scientist Salaries"
subtitle: "Category: Webscraping + Logistic Regression"
---


In this project we will webscrape indeed.com for data scientist job postings.  We will collect salary information, job summary, keywords, company name, location, number of reviews, etc.  Then using a logistic regression classification model we will predict whether or not the job is a high or low paying job.  While more precision might be better, there is a fair amount of natural variance in job salaries, which precludes us from applying regression techniques to predict individual salaries; however, using a binary predictor (e.g. high or low) may still be useful. The first part of the project will focus on scraping data, and the second will focus on using the listings with salary information to build a model to predict high/low salaries.   

We will collect salary information on data science jobs across the following 14 major US cities: *New York, Chicago, San Francisco, Austin, Atlanta, Boston, Washington, Seattle, Philadelphia, Los Angeles, Denver, San Diego, San Jose and Dallas.*  We would also like to understand which factors most directly impact salaries (e.g., location, summary keywords, etc.).  

This project will be focused around the following sections and is intended for a technical audience:


- Webscraping
- Logistic regression
- Data cleaning
- Conclusion

Here are some of the risks and assumptions we make with respect to the data and our model:


- We collect enough salary data for each city such that it is appropriately representative of the true salary distribution of that city
- All duplicate/redundant job postings are removed before modeling
- If a salary is stated as a range, we assume that by taking the mean of the upper and lower bounds it properly represents the salary information
- That the indeed search engine will only return job postings that are truly related to data scientist positions
- The model we build to predict high/low data scientist salaries is only relevant at the current point in time and should not be used in the future as the model will be outdated based on changes in job markets and salary structures 
- The sample size of job postings with listed salaries will be statistically significant


## Webscraping

When it comes to webscraping our general approach is to set up the 'get request' and then use a for loop to loop over all of the pages and cities.  Using the Python library Beautiful Soup is essential here as it allows us to extract important textual information from the HTML code.  We will first store the results in a list called 'results' and then convert that list to a dataframe called 'df'.  Then we will use the apply method on 'df' to call functions that will allow us to extract things like company name, location, summary snippet, etc. from the results column.

![png](/img/project-4-logreg_files/indeed_screen_shot.png)


```python
# Load packages
import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
```


```python
# Loop through cities and indeed pages to get job posting results
url_template = "http://www.indeed.com/jobs?q=data+scientist+%2420%2C000&l={}&start={}"
max_results_per_city = 3000

results = []

for city in set(['New+York', 'Chicago', 'San+Francisco', 'Austin', 'Atlanta', 'Boston', 'Washington', 'Seattle', 'Philadelphia', 'Los+Angeles', 'Denver', 'San+Diego', 'San+Jose', 'Dallas']):
    print 'Completed: ', city # Print out city just to let us know what city we are currently scraping
    for start in range(0, max_results_per_city, 10):
        # Grab the results from the request
        # Append to the full set of results
        r = requests.get(url_template.format(city,start))
        soup = BeautifulSoup(r.content.decode('utf-8'), "lxml")
        for i in soup.findAll(class_='result'):
            results.append([i,city]) 

df = pd.DataFrame(results,columns=['results','city'])
```




```python
# Print total number of job postings scraped
len(df)
```




    62855




```python
# Function to get company name from results
def get_company(result):
    if result.find('span', class_='company') is None:
        pass
    else:
        return filter(None,result.find('span', class_='company').text.split('\n'))[0].lstrip(' ')
```


```python
# Function to get location from results
def get_location(result):
    if result.find('span', class_='location') is None:
        pass
    else:
        return result.find('span', class_='location').text
```


```python
# Function to get snippet from results
def get_snippet(result):
    if result.find('span', class_='summary') is None:
        pass
    else:
        return filter(None,result.find('span', class_='summary').text.split('\n'))[0].lstrip(' ')
```


```python
# Function to get salary from results
# If salary is stated as a range, take average of lower and upper bounds
# Make sure salary is stated in annual terms
def get_salary(result):
    if result.find('nobr') is None:
        pass
    else: 
        pay = np.mean([int(s) for s in result.find('nobr').text.replace(',','').replace('$','').split() if s.isdigit()])
        if 'year' in result.find('nobr').text.replace(',','').replace('$','').split():
            return pay
        elif 'month' in result.find('nobr').text.replace(',','').replace('$','').split():
            return pay*12
        elif 'week' in result.find('nobr').text.replace(',','').replace('$','').split():
            return pay*52
        elif 'day' in result.find('nobr').text.replace(',','').replace('$','').split():
            return pay*5*52
        elif 'hour' in result.find('nobr').text.replace(',','').replace('$','').split():
            return pay*40*52
        else:
            pass
```


```python
# Function to get number of reviews from results
def get_review(result):
    if result.find('span', class_='slNoUnderline') == None:
        return 0
    else:
        return int(result.find('span', class_='slNoUnderline').text.split()[0].replace(',',''))
```


```python
# Function to get job post link from results
# We will call this later to obtain full job description
def get_job_link(result):
    return result.find('a', rel='nofollow')['href']
```


```python
## Call all functions with apply method
df['company'] = df['results'].apply(get_company)
df['location'] = df['results'].apply(get_location)
df['snippet'] = df['results'].apply(get_snippet)
df['salary'] = df['results'].apply(get_salary)
df['reviews'] = df['results'].apply(get_review)
df['job_link'] = df['results'].apply(get_job_link)
```

Let's check out how many results we got for each city.


```python
df['city'].value_counts()
```




    San+Jose         4500
    Boston           4500
    New+York         4500
    Washington       4500
    San+Francisco    4500
    Philadelphia     4491
    Seattle          4490
    Chicago          4487
    Los+Angeles      4486
    Atlanta          4484
    Austin           4480
    San+Diego        4480
    Denver           4479
    Dallas           4478
    Name: city, dtype: int64




```python
df.count()
```




    results     62855
    city        62855
    company     62854
    location    62855
    snippet     62855
    salary       2229
    reviews     46273
    job_link    62855
    dtype: int64


It looks like most of the job postings don't have a salary listed, so let's filter those jobs out.


```python
# Filter out rows that don't have a salary listed
df = df[df['salary'].notnull()]
```


```python
df.count()
```




    results     2229
    city        2229
    company     2229
    location    2229
    snippet     2229
    salary      2229
    reviews      736
    job_link    2229
    dtype: int64




```python
df.shape
```




    (2229, 8)




```python
# Remove duplicate rows to obtain only unique, non-redundant job postings
df_uniq = df.iloc[:,1:].drop_duplicates()
```


```python
df_uniq.shape
```




    (203, 7)



To recap so far, we webscraped 62,855 data science jobs from 14 different cities across the US.  That yielded us 2,229 jobs that had salaries posted, but when we dropped duplicate job postings it yielded us only 203 unique jobs.  This is a lot less than we expected and we'll need to keep that number of observations in mind when we put together the logistic regression model. Also, given the low number of observations we probably need to double check some of our risks and assumptions.


```python
df_uniq.reset_index(inplace=True)
```


```python
# Slice out index
df_uniq = df_uniq.iloc[:,1:]
```

## Data cleaning

Now here's where things get tricky and a litle unorthodox.  We want to access the full job descriptions to search for certain keywords and the 'job_link' column in the our dataframe will allow us to do that.  However, there are multiple instances where the links are broken.  Anytime we perform a 'get request' we will get an error.  If we use the apply method (like we did above) it will break for the entire column.  

In order to get around this we will use a for loop to loop through the links.  We will append the job summaries to a list called 'summary'.  Anytime the loop breaks due to a broken link, we will append a blank entry to the 'summary' list and restart the for loop after adjusting the starting point of the range function.  In the end we will have two separate dataframes, one that contains the job posting on the main results page and one that contains the full job description on the job-specific page.  We will save each dataframe separately and then merge them when it's time to build the model.


```python
summary = []
```


```python
for i in range(len(df_uniq['job_link'].tolist())):
    print i
    r = requests.get('https://www.indeed.com' + df_uniq['job_link'][i])
    soup = BeautifulSoup(r.content.decode('utf-8'), "lxml")
    summary.append(soup.text)
```



```python
# Run this once after each time the above for loop breaks
summary.append('0')
```


```python
# Convert summary list to a dataframe
df_summary = pd.DataFrame(summary,columns=['summary'])
```


```python
# Convert entire dataframe to lowercase
df_summary = df_summary.apply(lambda x: x.str.lower())
```

Now that we have all the relevant job posting data, let's search the job summaries for certain keywords that might given us insight into whether the job is a low or high paying job.


```python
# Function to search junior keywords
def get_junior(summary):
    a = ['junior','entry level', 'entry-level']
    if any(x in summary for x in a):
        return 1
    else:
        return 0
```


```python
df_summary['junior'] = df_summary['summary'].apply(get_junior)
```


```python
# Function to search senior keywords
def get_senior(summary):
    a = ['senior','very experienced','mid level','mid-level','manager','director']
    if any(x in summary for x in a):
        return 1
    else:
        return 0
```


```python
df_summary['senior'] = df_summary['summary'].apply(get_senior)
```


```python
# Function to search python and R keywords
def get_python(summary):
    a = ['pandas','numpy', 'scipy','python',' r ', 'r/','/r/']
    if any(x in summary for x in a):
        return 1
    else:
        return 0
```


```python
df_summary['python'] = df_summary['summary'].apply(get_python)
```


```python
# Function to search Excel keywords
def get_excel(summary):
    a = [' excel.','excel/',' excel ']
    if any(x in summary for x in a):
        return 1
    else:
        return 0
```


```python
df_summary['excel '] = df_summary['summary'].apply(get_excel)
```


```python
# Function to search big data keywords
def get_big_data(summary):
    a = ['big data','hive','mapreduce','spark','hadoop']
    if any(x in summary for x in a):
        return 1
    else:
        return 0
```


```python
df_summary['big_data'] = df_summary['summary'].apply(get_big_data)
```


```python
# Function to search advanced degree keywords
def get_phd_ms(summary):
    a = ['phd','doctorate','ms','ma','masters','master\'s']
    if any(x in summary for x in a):
        return 1
    else:
        return 0
```


```python
df_summary['phd'] = df_summary['summary'].apply(get_phd_ms)
```


```python
# Function to search machine learning keywords
def get_ml(summary):
    a = ['machine learning','scikit learn','sklearn', 'scikit-learn', 'sci-kit learn','regression','classification','logistic','knn','xgboost','pca','nlp']
    if any(x in summary for x in a):
        return 1
    else:
        return 0
```


```python
df_summary['ml'] = df_summary['summary'].apply(get_ml)
```


```python
# Function to search deep learning keywords
def get_nn(summary):
    a = ['neural nets','deep learning','tensorflow', 'tensor flow', 'neural networks','computer vision']
    if any(x in summary for x in a):
        return 1
    else:
        return 0
```


```python
df_summary['nn'] = df_summary['summary'].apply(get_nn)
```





```python
# Slice out 'summary' column since it's no longer needed
df_summary = df_summary.iloc[:,1:]
```



```python
# Save df_summary dataframe to csv
df_summary.to_csv('summary.csv')
```


```python
# Slice out 'location', 'snippet' and 'job_link' columns since they are no longer needed
df_uniq = df_uniq.iloc[:,[0,1,4,5]]
```


```python
# Save df_uniq dataframe to csv
df_uniq.to_csv('jobs.csv')
```

We have successfully removed duplicate job postings and saved both the summary and unique jobs dataframes to csv files.  Now it's time to start building our logistic regression model.


## Logistic Regression

We start by importing the jobs and summary csv files and merging the two dataframes.  Then we convert the salary column to a binary variable.  



```python
# Import jobs csv
jobs = pd.read_csv('jobs.csv')
```


```python
# Import job summary csv
summary = pd.read_csv('summary.csv')
```


```python
# Merge dataframes
df = jobs.merge(summary)
```


```python
# Slice out unnamed index column
df = df.iloc[:,1:]
```


```python
# Function to create binary target variable
mid_sal = df['salary'].quantile(.5)
def high_low(x):
    if x > mid_sal:
        return 1
    else:
        return 0
```


```python
# Apply high_low function to create new binary column
df['high_low'] = df['salary'].apply(high_low)
```



```python
ax = df['salary'].hist(bins=15);
ax.set_xlabel('Annual Salary ($)')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Data Scientist Salaries in US');
```


![png](/img/project-4-logreg_files/project-4-logreg_10_0.png)






```python
# Slice out 'company' and 'salary' columns and dummify 'city' column
# Set feature and target variable dataframes
X = pd.get_dummies(df.iloc[:,[0,3,4,5,6,7,8,9,10,11]])
y = df['high_low']
```

We scale the features (namely the 'reviews' column) before passing them into the logistic regression model. After that we will use GridSearchCV to optimize the model over an array of C values (the inverse of regularization strength) and compare between the l1 and l2 penalties.  Then we will fit the model with the optimized parameters and compute our cross validation accuracy score.  Because we have so many features, 23, compared to the total number of observations, 203, we want to make sure our C range goes very low in order to encourage a high regularization strength.  This will enable us to constrain the beta coefficients of the logistic regression model and reduce the feature importance of some of the lesser significant features (and in the case of the l1 norm it will outright eliminate some features). Also, because of the limited number of datapoints, we opt to use a high k value for the k-fold cross validation.  By using a high k-fold (in our case we chose cv=30) it allows us to train on more datapoints when predicting on the left out fold.   


```python
# Load sklearn packages
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
```



```python
# Scale input variables
standard_scaler = StandardScaler()
X_stand = standard_scaler.fit_transform(X)
```


```python
# Apply GridSearchCV to find optimal hyper-tuned parameters for logistic regression
logreg = LogisticRegression(solver='liblinear')
C_vals = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, .1, .15, .25, .275, .33, 0.5, .66, 0.75, 1.0, 10.0]
penalties = ['l1','l2']

gs = GridSearchCV(logreg, {'penalty': penalties, 'C': C_vals},\
                  verbose=False, cv=30)
gs.fit(X_stand, y)
```




    GridSearchCV(cv=30, error_score='raise',
           estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'penalty': ['l1', 'l2'], 'C': [1e-06, 5e-06, 1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.275, 0.33, 0.5, 0.66, 0.75, 1.0, 10.0]},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=False)




```python
# Display best parameters
gs.best_params_
```




    {'C': 0.05, 'penalty': 'l1'}




```python
# Set optimized parameters in logistic regression model
logreg = LogisticRegression(C=gs.best_params_['C'],\
                            penalty=gs.best_params_['penalty'])
```


```python
# Get cross validation score
cross_val_score(logreg, X_stand, y, n_jobs=1, cv=30).mean()
```




    0.75714285714285734




```python
# Fit logistic regression model and obtain feature ranking based on coefficient magnitude
cv_model = logreg.fit(X_stand, y)
feature_rank = pd.concat([pd.DataFrame(X.columns).T,pd.DataFrame(abs(cv_model.coef_))], axis =0).T
feature_rank.columns = ['feature','coef']
feature_rank.sort('coef',ascending=False)
```






<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>big_data</td>
      <td>0.27284</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ml</td>
      <td>0.188572</td>
    </tr>
    <tr>
      <th>21</th>
      <td>city_Seattle</td>
      <td>0.173555</td>
    </tr>
    <tr>
      <th>2</th>
      <td>senior</td>
      <td>0.135572</td>
    </tr>
    <tr>
      <th>0</th>
      <td>reviews</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>city_Dallas</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>city_San+Jose</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>city_San+Francisco</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>city_San+Diego</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>city_Philadelphia</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>city_New+York</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>city_Los+Angeles</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>city_Denver</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>city_Boston</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>city_Chicago</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>junior</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>city_Austin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>city_Atlanta</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nn</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>phd</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>excel</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>python</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>city_Washington</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Here we find that the optimal parameters for the logistic regression model are C = 0.05 and penalty = l1.  The model yields a cross validation accuracy score of 0.757, which - considering the variance in the salary data - isn't too bad.  When looking at the ranking of the features (based on the magnitude of the coefficients) we find that the most important factors when predicting high/low salaries of data scientists are: keywords related to 'big data', 'machine learning' and 'senior', and whether or not the job is located in Seattle. The l1 regularization eliminated all other features.

The ROC curve below illustrates the performance of classifying high paying data scientist jobs as the discrimination threshold is varied.  In our case we calculate an area under of curve of 0.81.  In essence the curve demonstrates that if we want to increase our accuracy in predicting high paying jobs when the jobs are in fact high paying (true positive rate) we would have to necessarily increase the number of instances where we are predicting a high paying job when in fact the job is not a high paying job (false positive rate).


```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
%matplotlib inline

Y_test = y
Y_score = logreg.decision_function(X_stand)

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(Y_test, Y_score)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve
plt.figure(figsize=[11,9])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver operating characteristic for predicting high/low data scientist salaries', fontsize=16)
plt.legend(loc="lower right")
plt.show()
```


![png](/img/project-4-logreg_files/project-4-logreg_25_0.png)


## Conclusion

We successfully scraped over 62,000 data science job postings across 14 major cities in the US from indeed.com.  When we filtered out the postings that did not contain salary information and removed duplicate rows we were left with roughly 203 salary observations.  For each job posting we were able to webscrape full job summaries and parse the text to measure word counts of many popular data science keywords.  With our array of features (including location, company name, number of reviews, keyword frequencies, etc.) we fit a logistic regression model to predict whether the job paid a high or low salary.  After optimizing the logistic regression model using GridSearchCV we achieved an accuracy of 0.76.  The high variance in the salary data made it difficult to achieve high accuracy scores.  

The low number of unique job postings with salaries (203) compared to the number of features (23) forced us to impose a high regularization strength on the logistic regression model to constrain the beta coefficients.  We found that the l1 penalty eliminated all but four features and the most important feature was big data related keywords.  We acknowledge that some of our assumptions about the data are likely incorrect due to the low number of salary datapoints we acquired.  For example, salary data for some cities may not be representative of the real salary distribution of that city and may be skewed based on small sample sizes or having few employers post many jobs with similar salaries in that city. Future work could include adding to this dataset with supplemental job postings from other career websites such as Glassdoor.com.  By increasing the number of datapoints we expect the variance to go down as currently several job markets feature very high variances that we do not believe is representative of the real salary distribution.

