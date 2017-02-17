---
layout: post
image: '/img/house_prices.png'
title: "Predicting Home Prices in Ames, Iowa"
subtitle: "Category: Regularized Linear Regression"
---



Our task for this project is to predict home prices in Ames, Iowa, using regularized linear regression.  The dataset comes from a [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), with one twist: we will only be using a subset of the features (19 of 82 features).  Of the 82 available features in the dataset we will limit our analysis and modeling to the following: *LotArea, Utilities, Neighborhood, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt, YearRemodAdd, RoofStyle, RoofMatl, GrLivArea, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, MoSold, YrSold, SalePrice*. A description from Kaggle of all the variables can be found [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).  A more complete data dictionary that includes descriptions of each of the categories can be found [here](https://ww2.amstat.org/publications/jse/v19n3/decock/datadocumentation.txt).

We will use a regularized linear regression approach to modeling home prices and compare both the Lasso and Ridge regularization methods.  In addition to predicting house prices we will also explore things like: where are most sales taking place, where are the most expensive houses located, and whether or not this is changing over time.  The underlying goal of this project is to apply Lasso and Ridge linear regression models to the Ames home price dataset (with explicit steps on how to accomplish this), and provide an $R^{2}$ comparison and discussion of the two models. We will carry out our analysis and modeling only on the training dataset provided by Kaggle and use Python pandas for the exploratory analysis and scikit-learn for the regression modeling.

This project will be organized around the following sections:


- Exploratory analysis
- Linear regression modeling


Here are our assumptions about the data:


- We assume all values in the dataset are correctly categorized and assigned



## Exploratory analysis

Let's start by reading in the data and plotting a histogram of home sale prices ('SalePrice'), which is the target variable we will be trying to predict.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

fields = ['LotArea', 'Utilities', 'Neighborhood', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'MoSold', 'YrSold', 'SalePrice']
df = pd.read_csv('train.csv', usecols = fields)
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotArea</th>
      <th>Utilities</th>
      <th>Neighborhood</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8450</td>
      <td>AllPub</td>
      <td>CollgCr</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1710</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2008</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9600</td>
      <td>AllPub</td>
      <td>Veenker</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1262</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>2007</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11250</td>
      <td>AllPub</td>
      <td>CollgCr</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1786</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>2008</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9550</td>
      <td>AllPub</td>
      <td>Crawfor</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>1717</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2006</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14260</td>
      <td>AllPub</td>
      <td>NoRidge</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>2198</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>12</td>
      <td>2008</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
</div>




```python
nans = pd.isnull(df).sum()
nans[nans>0]
```




    Series([], dtype: int64)



Wow, looks like we're working with a pretty clean dataset -- no NaNs were found anywhere in our data.


```python
# ax = df['SalePrice'].hist(color='r')
# ax.set_xlabel('Sale Price (USD)')
# ax.set_title('Sale Price Histogram');
```


```python
ax = sns.distplot(df['SalePrice'], color='red', bins=10, hist_kws=dict(alpha=1), kde_kws={"color": "darkred"})
ax.set_xlabel('Sale Price (USD)')
ax.set_title('Sale Price Histogram')
plt.show()
```


![png](/img/Project-3_files/Project-3_8_0.png)


We see that the distribution is positively skewed.  Let's apply a logarithmic transformation to the 'SalePrice' variable to see if we can transform it to become more normally distributed. 


```python
# ax = df['SalePrice'].hist(color='r')
# ax.set_xlabel('Log Sale Price (USD)')
# ax.set_title('Log Sale Price Histogram');
```


```python
df['SalePrice'] = np.log1p(df['SalePrice'])
ax = sns.distplot(df['SalePrice'], color='red', bins=10, hist_kws=dict(alpha=1), kde_kws={"color": "darkred"})
ax.set_xlabel('Log Sale Price (USD)')
ax.set_title('Log Sale Price Histogram')
plt.show()
```


![png](/img/Project-3_files/Project-3_11_0.png)


This transformation creates a much more normally distributed histogram, which will come in handy later during the linear regression phase.

Now let's plot the number of houses sold by neighborhood in Ames, Iowa, and see how they stack up.  We can use [this](https://ww2.amstat.org/publications/jse/v19n3/decock/AmesResidential.pdf) Ames residential neighborhood map to try to identify patterns in the location data.

![png](/img/Project-3_files/ames_neighborhoods.png)


```python
ax = df.groupby('Neighborhood')['SalePrice'].count().to_frame().sort_values('SalePrice', ascending = False).plot(kind = 'bar', rot=65, figsize=(16,8), width=.8, color='r', legend=False)
ax.set_xlabel('Neighborhood', fontsize=16)
ax.set_title('Number of Homes Sold, 2006-2010', fontsize=16);
```


![png](/img/Project-3_files/Project-3_15_0.png)


Based on the map it looks like the neighborhoods with the most sales (e.g., North Ames, College Creek, Old Town, Edwards, Somerset) tend to be clustered in the northern and western parts of the city.  Let's take a look at home sales as a function of year.


```python
df['YrSold'].value_counts().sort_index().plot(kind = 'bar', width=.8, color='r', rot=45).set_title('Number of Homes Sold in Ames by Year');
```


![png](/img/Project-3_files/Project-3_17_0.png)



```python
df['YrSold'].value_counts().sort_index()
```




    2006    314
    2007    329
    2008    304
    2009    338
    2010    175
    Name: YrSold, dtype: int64




```python
df[df['SalePrice']>df['SalePrice'].quantile(.8)]['YrSold'].value_counts().sort_index()
```




    2006    65
    2007    71
    2008    56
    2009    66
    2010    31
    Name: YrSold, dtype: int64



We observe a steady trend with the notable exception of 2010, which drops off precipitously.  This could be due to one of the following reasons: (1) the dataset only captures part of 2010, which would therefore suppress total home sales for that year or, (2) the financial housing crisis of 2009 finally caught up to Ames, which led to a significant downturn in home sales activity.  Let's go into more detail by looking at the number of homes sold per year by neighborhood.  We find that 2010 was a down year for almost all Ames neighborhoods.  The most frequently sold neighborhood, North Ames, peaked in 2007 in what appears to be the only time more than 50 homes were sold anywhere in Ames in any year.


```python
#plt.style.use('fivethirtyeight')
ax = pd.pivot_table(df, index=['Neighborhood'], values=['SalePrice'], columns = ['YrSold'], aggfunc=[len]).plot(kind='bar', rot=65, figsize=(16,8), width=.8)
ax.set_xlabel('Neighborhood', fontsize=14)
ax.set_title('Number of Homes Sold by Year', fontsize=14)
ax.legend(labels=['2006','2007','2008','2009','2010']);
```


![png](/img/Project-3_files/Project-3_21_0.png)


Let's now take a look at the most expensive home sales in Ames.  We define the most expensive homes as those in the top 20% percent of home sale prices.  


```python
ax = df[df['SalePrice']>df['SalePrice'].quantile(.8)]['Neighborhood'].value_counts().plot(kind = 'bar', figsize=(16,8), width=.8, color='r')
ax.set_xlabel('Neighborhood', fontsize=16)
ax.set_title('Number of Expensive Homes Sold, 2006-2010', fontsize=16);
```


![png](/img/Project-3_files/Project-3_23_0.png)


Here we see that Somerset and College Creek seem to be toward the top of both charts, which means not only do they rank well in overall number of homes sold but they also rank well for most expensive homes sold.  Northridge Heights, Northridge and Somerset all seem to be clustered around the same northern part of Ames, whereas College Creek is on the western edge of the city.


```python
df[df['SalePrice']>df['SalePrice'].quantile(.8)]['YrSold'].value_counts().sort_index().plot(kind = 'bar', color='r', rot=65).set_title('Number of Homes Sold in Ames by Year');
```


![png](/img/Project-3_files/Project-3_25_0.png)


Here we find the drop off in number of homes sold from 2009 to 2010 was greater than 50% among the most expensive homes.  This drop off is more pronounced than that observed on the overall population of homes sold over that same time.  We also note that though home sales overall peaked in 2009, expensive home sales actually peaked in 2007.  Slight dips were also observed in 2008 for both expensive homes and overall homes.


```python
ax = pd.pivot_table(df[df['SalePrice']>df['SalePrice'].quantile(.8)] , index=['Neighborhood'], values=['SalePrice'], columns = ['YrSold']).plot(kind='bar', figsize=(18,8), width=.8)
ax.set_xlabel('Neighborhood', fontsize=14)
ax.set_ylim([8, 14])
ax.set_title('Number of Expensive Homes Sold by Year', fontsize=14)
ax.legend(labels=['2006','2007','2008','2009','2010']);
```


![png](/img/Project-3_files/Project-3_27_0.png)


Some of the numerical features have outliers, which we need to deal with before we start modeling.  Let's calculate the skew of our numeric columns.


```python
from scipy.stats import skew
```


```python
num_cols = ['LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','GrLivArea','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','MoSold','YrSold']
for i in num_cols:
    print i, skew(df[i])
```

    LotArea 12.1951421251
    OverallQual 0.216720976526
    OverallCond 0.692355213552
    YearBuilt -0.612830724203
    YearRemodAdd -0.50304449676
    GrLivArea 1.36515595477
    FullBath 0.0365239844325
    HalfBath 0.675202834774
    BedroomAbvGr 0.211572441582
    KitchenAbvGr 4.48378409386
    MoSold 0.21183506019
    YrSold 0.0961695796181


Here we find large positive skews in 'LotArea', 'GrLivArea' and 'KitchenAbvGr', so we apply a logarithmic transformation to these features.


```python
df['LotArea'] = np.log1p(df['LotArea'])
df['GrLivArea'] = np.log1p(df['GrLivArea'])
df['KitchenAbvGr'] = np.log1p(df['KitchenAbvGr'])
```

Now we need to convert our categorical variables to dummy variables so that we can use them appropriately in our linear regression model.  Note that by converting the categorical variables to dummy variables we are increasing the size of our dateframe from 19 columns to 61 columns.


```python
df = pd.get_dummies(df, drop_first=True)
```

Let's create a correlation heatmap so that we can quickly visualize the interdependencies among the features.


```python
# Move 'SalePrice' column to end of dataframe
cols = list(df)
cols.insert(len(cols), cols.pop(cols.index('SalePrice')))
df = df.ix[:, cols]
# Double check shape of dataframe
df.shape
```




    (1460, 61)




```python
sns.set(style="white")

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 16))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.2,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.1, cbar_kws={"shrink": .7}, ax=ax);
```


![png](/img/Project-3_files/Project-3_37_0.png)


By looking at the bottom row of the correlation triangle we identify that the highest correlations with 'SalePrice' seem to be 'OverallQual' and 'GrLivArea'.



## Linear regression modeling

This Kaggle competition uses the root-mean-squared-error metric to evaluate model submissions.  Specificaly, here's what the competition rules say:

*Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)*

We will use the RMSE metric to optimize and tune our model. But because the goal of this project isn't necessarily to submit to Kaggle, we will opt to use $R^{2}$ values (percentage of variance explained) to compare predictions among our models to determine which one performs best.

We will focus on the Lasso (L1 norm) and Ridge (L2 norm) regularization methods as they help promote simple models by penalizing complexity (as measured by the &beta; magnitudes).  First, however, we will apply linear regression without any regularization in order to establish a baseline for comparison.


```python
# Set X and y
X = df.iloc[:,:60]
y = df.iloc[:,60]
```


```python
# Import model and estimator packages
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
```


```python
# Instantiate linear model
lr = linear_model.LinearRegression()

# Fit our model using our data and target
lr_model = lr.fit(X, y)

# We pass in our estimator, lr, and our data and target
print 'lr rmse: ', np.sqrt(-cross_val_score(lr, X, y, n_jobs=1, cv=5, scoring='neg_mean_squared_error')).mean()
```

    lr rmse:  0.153464472135


The RMSE for our linear regression model is 0.153.  Now let's try Ridge regression with leave-one-out cross validation (LOOCV) to see how it compares.


```python
# Import model and estimator packages
from sklearn.linear_model import RidgeCV
```


```python
# Set regularization strength
alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20]
```


```python
# Instantiate RidgeCV model
rcv = linear_model.RidgeCV(alphas=alphas, store_cv_values=True)

# Fit our model using our data and target
rcv_model = rcv.fit(X, y)

# Create dataframe of rmse with alpha as column header
np.sqrt(pd.DataFrame(rcv_model.cv_values_, columns=rcv.alphas)).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.0001</th>
      <th>0.0005</th>
      <th>0.001</th>
      <th>0.005</th>
      <th>0.01</th>
      <th>0.1</th>
      <th>0.25</th>
      <th>0.5</th>
      <th>1.0</th>
      <th>2.0</th>
      <th>5.0</th>
      <th>10.0</th>
      <th>20.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.002275</td>
      <td>0.002291</td>
      <td>0.002311</td>
      <td>0.002463</td>
      <td>0.002638</td>
      <td>0.004317</td>
      <td>0.005130</td>
      <td>0.005393</td>
      <td>0.005152</td>
      <td>0.004209</td>
      <td>0.001416</td>
      <td>0.002238</td>
      <td>0.007401</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.089771</td>
      <td>0.089818</td>
      <td>0.089877</td>
      <td>0.090330</td>
      <td>0.090858</td>
      <td>0.096414</td>
      <td>0.099738</td>
      <td>0.101271</td>
      <td>0.100880</td>
      <td>0.097593</td>
      <td>0.087381</td>
      <td>0.075302</td>
      <td>0.061085</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.035669</td>
      <td>0.035678</td>
      <td>0.035689</td>
      <td>0.035776</td>
      <td>0.035877</td>
      <td>0.036904</td>
      <td>0.037410</td>
      <td>0.037434</td>
      <td>0.036811</td>
      <td>0.035220</td>
      <td>0.031098</td>
      <td>0.026313</td>
      <td>0.020519</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.178595</td>
      <td>0.178593</td>
      <td>0.178591</td>
      <td>0.178574</td>
      <td>0.178552</td>
      <td>0.177984</td>
      <td>0.176931</td>
      <td>0.175326</td>
      <td>0.172618</td>
      <td>0.168280</td>
      <td>0.158649</td>
      <td>0.147203</td>
      <td>0.132031</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.159036</td>
      <td>0.159061</td>
      <td>0.159091</td>
      <td>0.159326</td>
      <td>0.159597</td>
      <td>0.162288</td>
      <td>0.163968</td>
      <td>0.165351</td>
      <td>0.167053</td>
      <td>0.169248</td>
      <td>0.172352</td>
      <td>0.173261</td>
      <td>0.170605</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Take mean of all the rmse values for each alpha
rcv_rmse = np.sqrt(pd.DataFrame(rcv_model.cv_values_, columns=rcv.alphas)).mean(axis=0)

# Return optimal alpha value
rcv_model.alpha_
```




    2.0




```python
print 'rcv rmse: ', rcv_rmse.min()
```

    rcv rmse:  0.0977133322139


The RMSE of our RidgeCV model is 0.097, which is a significant improvement over the linear regression model.


```python
ax = rcv_rmse.plot(figsize=(16,8), linewidth=3, fontsize=16)
ax.set_ylabel('RMSE', fontsize=16)
ax.set_xlabel(r'$\alpha$', fontsize=16)
ax.set_title('Ridge Model', fontsize=16);
```


![png](/img/Project-3_files/Project-3_52_0.png)


Here &alpha; is the tuning parameter that controls the strength of the regularization.  Higher &alpha; values will contrain more the &beta; coefficients of the linear model.  When &alpha; is zero we get back a non-regularized linear model. If we take a look at the above plot first we see the error improve as &alpha; increases, then it hits a minimum and then it starts to increase.  This is an example of the bias-variance tradeoff.  At first, the low &alpha; values show that the model is overfit with a high variance (i.e., our model is too tightly fit and is starting to pick up the noise in the data, which leads to a lot of variance).  As &alpha; reaches the higher end we find that the model is underfit (i.e., our model is too simplified for the data and we are therefore biasing our predictions to be poorly representative of the shape or complexity of the data). The higher the &alpha; value the less prone our model will be to overfitting.  Here we find that the RMSE reaches a minimum at &alpha; = 2 on our RidgeCV regularization model, which represents the minimum sum of the bias and variance errors.


```python
# Let's look at the beta coefficients
rcv_coef = pd.Series(rcv_model.coef_, index = X.columns)
```


```python
rcv_coef.plot(kind = "barh", color='r', figsize=(8,16))
plt.title("Coefficients in the RidgeCV Model");
```


![png](/img/Project-3_files/Project-3_55_0.png)


Although its difficult to interpret exactly how important each feature is since we have already transformed several numeric variables, if we look at the magnitude of the &beta; coefficients of each feature we can try to get a general sense of their relative importance.  Here we see that by far the most important feature is 'GrLivArea' or the above grade (ground) living area square footage. This makes sense since the size of a house is typically a major indicator of its value.  We also observed earlier that 'GrLivArea' had one of the highest correlations with 'SalePrice'.  Other significant features include the dummified neighborhood variables (location, location, location!) as well as whether or not the house has wooden shingles.

Now let's try Lasso regression with 5-fold cross validation to see how it compares.


```python
# Import model and estimator packages
from sklearn.linear_model import LassoCV
```


```python
# Instantiate LassoCV model
lcv = LassoCV(alphas = alphas, cv=5)
```


```python
# Fit our model using our data and target
lcv_model = lcv.fit(X,y)
```


```python
# Take mean of all the rmse values for each alpha
lcv_rmse = np.mean(np.sqrt(lcv_model.mse_path_), axis=1)
lcv_rmse
```




    array([ 0.39922569,  0.39922569,  0.36334211,  0.32052322,  0.30933256,
            0.30650074,  0.30580213,  0.23817143,  0.16223063,  0.15639528,
            0.14310425,  0.14099626,  0.14391593])




```python
# Return optimal alpha
lcv_model.alpha_
```




    0.00050000000000000001




```python

```


```python
# Create dataframe of alpha and rmse values
lasso_df = pd.DataFrame(zip(lcv_model.alphas,lcv_rmse[::-1]), columns=['Alpha','RMSE'])
lasso_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alpha</th>
      <th>RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0001</td>
      <td>0.143916</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0005</td>
      <td>0.140996</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0010</td>
      <td>0.143104</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0050</td>
      <td>0.156395</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0100</td>
      <td>0.162231</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.1000</td>
      <td>0.238171</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.2500</td>
      <td>0.305802</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.5000</td>
      <td>0.306501</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0000</td>
      <td>0.309333</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.0000</td>
      <td>0.320523</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5.0000</td>
      <td>0.363342</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10.0000</td>
      <td>0.399226</td>
    </tr>
    <tr>
      <th>12</th>
      <td>20.0000</td>
      <td>0.399226</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = lasso_df.plot(x='Alpha',y='RMSE', xlim=[0,0.001],ylim=[.14,.144], legend=False, linewidth=2)
ax.set_ylabel('RMSE');
ax.set_xlabel(r'$\alpha$')
ax.set_title('Lasso Model');
```


![png](/img/Project-3_files/Project-3_65_0.png)


The above plot shows the RMSE is minimized in the LassoCV model with an &alpha; of 0.0005.  After that as &alpha; increases the model starts to have too much bias (the &beta; coefficients are too constrained).


```python
print 'lcv rmse: ', lcv_rmse.min()
```

    lcv rmse:  0.140996258854



```python
# Let's look at the beta coefficients
lcv_coef = pd.Series(lcv_model.coef_, index = X.columns)
```


```python
# Let's see how many coefficients were dropped to 0 through the regularization
print len(lcv_coef[lcv_coef != 0])
print len(lcv_coef[lcv_coef == 0])
```

    40
    20



```python
lcv_coef.plot(kind = "barh", color='r', figsize=(8,16))
plt.title("Coefficients in the Lasso Model");
```


![png](/img/Project-3_files/Project-3_70_0.png)


The biggest finding we make here is that the Lasso model drops 20 different &beta; coefficients to zero, which effectively reduces the number of feature inputs to 40.  We observe similar traits in the Lasso coefficients as we did with the Ridge coefficients.  'GrLivArea' seems to dominate in terms of feature importance.  To a lesser extent the dummified neighborhood features as well as 'LotArea' seem to play important roles in the model.

Based on an evaluation of our RMSE values it looks like the RidgeCV model with &alpha; = 2 gives the lowest error. Now that we have our three regression models (i.e., linear, RidgeCV, LassoCV) let's compare the results of our cross validation predictions with the $R^{2}$ metric.


```python
# Import model and estimator packages
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
```


```python
# Cross val predict and evaluate R-squared
cvp_lr_preds = cross_val_predict(lr, X, y, cv=5)

cvp_lr_r2 =  r2_score(y_true=y, y_pred=cvp_lr_preds)
cvp_lr_r2
```




    0.85024961489624973




```python
# Cross val predict and evaluate R-squared
cvp_rcv_preds = cross_val_predict(rcv_model, X, y, cv=5)

cvp_rcv_r2 =  r2_score(y_true=y, y_pred=cvp_rcv_preds)
cvp_rcv_r2
```




    0.87491839874896848




```python
# Cross val predict and evaluate R-squared
cvp_lcv_preds = cross_val_predict(lcv_model, X, y, cv=5)

cvp_lcv_r2 =  r2_score(y_true=y, y_pred=cvp_lcv_preds)
cvp_lcv_r2
```




    0.86953248037597608




```python
# Plot actual vs predicted for all 3 models
fig, axes = plt.subplots(1,3, figsize=(16,8))

axes[0].scatter(cvp_lr_preds, y, c='r')
axes[1].scatter(cvp_rcv_preds, y, c='b')
axes[2].scatter(cvp_lcv_preds, y, c='k')

axes[0].set_title('Linear Model, ' + r'$R{^2} = $' + str(round(cvp_lr_r2,4)))
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')
axes[0].set_xlim([10,14])

axes[1].set_title('Ridge Model, ' + r'$R{^2} = $' + str(round(cvp_rcv_r2,4)))
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')
axes[1].set_xlim([10,14])

axes[2].set_title('Lasso Model, ' + r'$R{^2} = $' + str(round(cvp_lcv_r2,4)))
axes[2].set_ylabel('Actual')
axes[2].set_xlabel('Predicted')
axes[2].set_xlim([10,14]);
```


![png](/img/Project-3_files/Project-3_77_0.png)



```python
# Plot residuals for all 3 models
fig, axes = plt.subplots(1,3, figsize=(16,8))

axes[0].scatter(cvp_lr_preds, y-cvp_lr_preds, c='r')
axes[1].scatter(cvp_rcv_preds, y-cvp_rcv_preds, c='b')
axes[2].scatter(cvp_lcv_preds, y-cvp_lcv_preds, c='k')

axes[0].set_title('Linear Model')
axes[0].set_ylabel('Residuals')
axes[0].set_xlabel('Predicted')
axes[0].set_xlim([10,14])
axes[0].set_ylim([-1.5,2])

axes[1].set_title('Ridge Model')
axes[1].set_ylabel('Residuals')
axes[1].set_xlabel('Predicted')
axes[1].set_xlim([10,14])
axes[1].set_ylim([-1.5,2])

axes[2].set_title('Lasso Model')
axes[2].set_ylabel('Residuals')
axes[2].set_xlabel('Predicted')
axes[2].set_xlim([10,14])
axes[2].set_ylim([-1.5,2]);
```


![png](/img/Project-3_files/Project-3_78_0.png)


Here we find that the RidgeCV model performs slightly better than the LassoCV model based on a comparison of $R^{2}$ values, or the percentage of variance explained.  The RidgeCV model has an $R^{2}$ of 0.8749 whereas the LassoCV model has an $R^{2}$ of 0.8695.  Visually, the two models appear to be very similar on the *Actual vs. Predicted* and *Residuals* plots.  The only difference seems to be a couple outliers. Both the RidgeCV and LassoCV models showed better $R^{2}$ scores than the linear regression model, which means that the linear model was overfit and by adding some regularization we were able to constrain the &beta; coefficients and reduce the variance (essentially "relaxing" the model slightly). To improve on the model further, in the future we could average several RidgeCV models or use ensembling techniques to combine the LassoCV and RidgeCV models in an effort to reduce the variance in the model even more.

In summary, we found that number of overall home sales peaked in Ames in 2009, whereas number of expensive home sales peaked in 2007.  A dramatic reduction in number of homes sold was observed in 2010.  The top districts in Ames for both number of expensive and overall home sales seem to be concentrated in the northern and western parts of the city. Using linear, Lasso and Ridge regressions we predicted home sales prices in Ames. A comparison of $R^{2}$ values demonstrated that the Ridge linear regression model was the best performer, and both the Lasso and Ridge models outperformed the linear regression model.  This proved that the original linear regression model was overfit and by constraining the &beta; coefficients we were able to reduce the variance of the model.
