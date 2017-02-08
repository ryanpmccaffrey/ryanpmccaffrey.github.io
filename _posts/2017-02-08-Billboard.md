---
layout: post
image: '/img/billboard.png'
title: "Billboard: What Makes a Hit Song?"
subtitle: "Category: Exploratory Data Analysis"
---


The Billboard Hot 100 is a weekly music chart used to rank songs based on various factors such as sales, online streaming and radio play.  It is largely regarded as the gold standard of music and record rankings.  The objective of this project is to explore a Billboard dataset from the year 2000 and try to determine what makes a song great and what keeps it on the charts.  We will start by coming up with some ideas, relevant questions, and hypotheses that we'd like explore in the data. After that we'll start working with the data, first by cleaning it up and then by doing some exploratory analysis.  We will end by evaluating our hypotheses to determine whether or not the trends or effects we hypothesized are actually real (and statistically significant).

This project will be organized around the following sections:
* Hypothesis brainstorming
* Cleaning the data 
* Exploratory analysis
* Hypothesis testing 

Before moving onto the hypothesis brainstorming let's list here the key assumptions we are making with the data:
1. We will assume that the data is accurate and that both the numeric and categorical variables are correctly assigned (e.g., we will not check individual records to ensure they are properly categorized with respect to 'genre', 'time' or any other variables).
2. We will use a significance level of 0.05 to test our null hypothesis.

## Hypothesis brainstorming

Before we dive deep into the data I think it's important to first come up with ideas about what factors we believe could influence the success of a song on the Billboard Hot 100 list.  We will be judging success both on number of weeks a song stays in the top 100 as well as the average ranking of that song over that time.

Here are some questions we would like to explore in the data:
- Does genre popularity ultimately influence or limit the longevity of a song on the charts?
- Are shorter songs more popular (easier to repeat)?
- Do songs that peak early tend to fade away more quickly?
- Do songs that peak early tend to have worse rankings?
- Are there seasonality effects that influence the average ranking or longevity of a song (e.g., Christmas or holiday songs having shorter lifespan on the charts)? 

## Cleaning the data

The raw data comes to us in a state that requires a little cleaning in order for us to make sense of it.  As part of the cleaning process, after most manipulations we will redisplay the first three lines of the dataframe with the .head(3) method to show the reader an updated view of the dataframe. Now let's get started by loading our Python packages and reading in our data.


```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

bb = pd.read_csv('assets/billboard.csv')
bb.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>artist.inverted</th>
      <th>track</th>
      <th>time</th>
      <th>genre</th>
      <th>date.entered</th>
      <th>date.peaked</th>
      <th>x1st.week</th>
      <th>x2nd.week</th>
      <th>x3rd.week</th>
      <th>...</th>
      <th>x67th.week</th>
      <th>x68th.week</th>
      <th>x69th.week</th>
      <th>x70th.week</th>
      <th>x71st.week</th>
      <th>x72nd.week</th>
      <th>x73rd.week</th>
      <th>x74th.week</th>
      <th>x75th.week</th>
      <th>x76th.week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>Destiny's Child</td>
      <td>Independent Women Part I</td>
      <td>3,38,00 AM</td>
      <td>Rock</td>
      <td>September 23, 2000</td>
      <td>November 18, 2000</td>
      <td>78</td>
      <td>63</td>
      <td>49</td>
      <td>...</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000</td>
      <td>Santana</td>
      <td>Maria, Maria</td>
      <td>4,18,00 AM</td>
      <td>Rock</td>
      <td>February 12, 2000</td>
      <td>April 8, 2000</td>
      <td>15</td>
      <td>8</td>
      <td>6</td>
      <td>...</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000</td>
      <td>Savage Garden</td>
      <td>I Knew I Loved You</td>
      <td>4,07,00 AM</td>
      <td>Rock</td>
      <td>October 23, 1999</td>
      <td>January 29, 2000</td>
      <td>71</td>
      <td>48</td>
      <td>43</td>
      <td>...</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 83 columns</p>
</div>



The first thing we do is check the dimensions of the dataframe.


```python
bb.shape
```




    (317, 83)



Here we find that the dataframe has 317 rows and 83 columns.  Let's take a closer look at the contents of the dataframe.

After taking a quick glance at the column names we opt to replace the periods with underscores in order to make it more human readable.  We also decide to rename the 'artist.inverted' column.


```python
bb.columns = [x.replace('.','_') for x in bb.columns]
bb.rename(columns={'artist_inverted': 'artist'}, inplace=True)
bb.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>artist</th>
      <th>track</th>
      <th>time</th>
      <th>genre</th>
      <th>date_entered</th>
      <th>date_peaked</th>
      <th>x1st_week</th>
      <th>x2nd_week</th>
      <th>x3rd_week</th>
      <th>...</th>
      <th>x67th_week</th>
      <th>x68th_week</th>
      <th>x69th_week</th>
      <th>x70th_week</th>
      <th>x71st_week</th>
      <th>x72nd_week</th>
      <th>x73rd_week</th>
      <th>x74th_week</th>
      <th>x75th_week</th>
      <th>x76th_week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>Destiny's Child</td>
      <td>Independent Women Part I</td>
      <td>3,38,00 AM</td>
      <td>Rock</td>
      <td>September 23, 2000</td>
      <td>November 18, 2000</td>
      <td>78</td>
      <td>63</td>
      <td>49</td>
      <td>...</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000</td>
      <td>Santana</td>
      <td>Maria, Maria</td>
      <td>4,18,00 AM</td>
      <td>Rock</td>
      <td>February 12, 2000</td>
      <td>April 8, 2000</td>
      <td>15</td>
      <td>8</td>
      <td>6</td>
      <td>...</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000</td>
      <td>Savage Garden</td>
      <td>I Knew I Loved You</td>
      <td>4,07,00 AM</td>
      <td>Rock</td>
      <td>October 23, 1999</td>
      <td>January 29, 2000</td>
      <td>71</td>
      <td>48</td>
      <td>43</td>
      <td>...</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 83 columns</p>
</div>



First, we check the 'year' column to see over how many years this dataset spans.


```python
bb['year'].unique()
```




    array([2000])



Since all years are 2000, we drop them from the dataframe as it provides no additional insight.


```python
bb.drop('year',1, inplace=True)
```


```python
bb.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>track</th>
      <th>time</th>
      <th>genre</th>
      <th>date_entered</th>
      <th>date_peaked</th>
      <th>x1st_week</th>
      <th>x2nd_week</th>
      <th>x3rd_week</th>
      <th>x4th_week</th>
      <th>...</th>
      <th>x67th_week</th>
      <th>x68th_week</th>
      <th>x69th_week</th>
      <th>x70th_week</th>
      <th>x71st_week</th>
      <th>x72nd_week</th>
      <th>x73rd_week</th>
      <th>x74th_week</th>
      <th>x75th_week</th>
      <th>x76th_week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Destiny's Child</td>
      <td>Independent Women Part I</td>
      <td>3,38,00 AM</td>
      <td>Rock</td>
      <td>September 23, 2000</td>
      <td>November 18, 2000</td>
      <td>78</td>
      <td>63</td>
      <td>49</td>
      <td>33</td>
      <td>...</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Santana</td>
      <td>Maria, Maria</td>
      <td>4,18,00 AM</td>
      <td>Rock</td>
      <td>February 12, 2000</td>
      <td>April 8, 2000</td>
      <td>15</td>
      <td>8</td>
      <td>6</td>
      <td>5</td>
      <td>...</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Savage Garden</td>
      <td>I Knew I Loved You</td>
      <td>4,07,00 AM</td>
      <td>Rock</td>
      <td>October 23, 1999</td>
      <td>January 29, 2000</td>
      <td>71</td>
      <td>48</td>
      <td>43</td>
      <td>31</td>
      <td>...</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
      <td>*</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 82 columns</p>
</div>



Next, we notice that a lot of the 'week' columns use \* to denote null values.  We write a function to remove \* and replace them with NaN.


```python
def replace_nulls(value):
    if value == '*':
        return np.nan
    else:
        return value

bb = bb.applymap(replace_nulls)
```


```python
bb.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>track</th>
      <th>time</th>
      <th>genre</th>
      <th>date_entered</th>
      <th>date_peaked</th>
      <th>x1st_week</th>
      <th>x2nd_week</th>
      <th>x3rd_week</th>
      <th>x4th_week</th>
      <th>...</th>
      <th>x67th_week</th>
      <th>x68th_week</th>
      <th>x69th_week</th>
      <th>x70th_week</th>
      <th>x71st_week</th>
      <th>x72nd_week</th>
      <th>x73rd_week</th>
      <th>x74th_week</th>
      <th>x75th_week</th>
      <th>x76th_week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Destiny's Child</td>
      <td>Independent Women Part I</td>
      <td>3,38,00 AM</td>
      <td>Rock</td>
      <td>September 23, 2000</td>
      <td>November 18, 2000</td>
      <td>78</td>
      <td>63</td>
      <td>49</td>
      <td>33</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Santana</td>
      <td>Maria, Maria</td>
      <td>4,18,00 AM</td>
      <td>Rock</td>
      <td>February 12, 2000</td>
      <td>April 8, 2000</td>
      <td>15</td>
      <td>8</td>
      <td>6</td>
      <td>5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Savage Garden</td>
      <td>I Knew I Loved You</td>
      <td>4,07,00 AM</td>
      <td>Rock</td>
      <td>October 23, 1999</td>
      <td>January 29, 2000</td>
      <td>71</td>
      <td>48</td>
      <td>43</td>
      <td>31</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 82 columns</p>
</div>



Next, let's take a closer look at the 'week' columns to see how many song counts we have for each week.


```python
bb.iloc[:,6:].apply(lambda x: x.count(), axis=0).to_frame()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>x1st_week</th>
      <td>317</td>
    </tr>
    <tr>
      <th>x2nd_week</th>
      <td>312</td>
    </tr>
    <tr>
      <th>x3rd_week</th>
      <td>307</td>
    </tr>
    <tr>
      <th>x4th_week</th>
      <td>300</td>
    </tr>
    <tr>
      <th>x5th_week</th>
      <td>292</td>
    </tr>
    <tr>
      <th>x6th_week</th>
      <td>280</td>
    </tr>
    <tr>
      <th>x7th_week</th>
      <td>269</td>
    </tr>
    <tr>
      <th>x8th_week</th>
      <td>260</td>
    </tr>
    <tr>
      <th>x9th_week</th>
      <td>253</td>
    </tr>
    <tr>
      <th>x10th_week</th>
      <td>244</td>
    </tr>
    <tr>
      <th>x11th_week</th>
      <td>236</td>
    </tr>
    <tr>
      <th>x12th_week</th>
      <td>222</td>
    </tr>
    <tr>
      <th>x13th_week</th>
      <td>210</td>
    </tr>
    <tr>
      <th>x14th_week</th>
      <td>204</td>
    </tr>
    <tr>
      <th>x15th_week</th>
      <td>197</td>
    </tr>
    <tr>
      <th>x16th_week</th>
      <td>182</td>
    </tr>
    <tr>
      <th>x17th_week</th>
      <td>177</td>
    </tr>
    <tr>
      <th>x18th_week</th>
      <td>166</td>
    </tr>
    <tr>
      <th>x19th_week</th>
      <td>156</td>
    </tr>
    <tr>
      <th>x20th_week</th>
      <td>146</td>
    </tr>
    <tr>
      <th>x21st_week</th>
      <td>65</td>
    </tr>
    <tr>
      <th>x22nd_week</th>
      <td>55</td>
    </tr>
    <tr>
      <th>x23rd_week</th>
      <td>48</td>
    </tr>
    <tr>
      <th>x24th_week</th>
      <td>46</td>
    </tr>
    <tr>
      <th>x25th_week</th>
      <td>38</td>
    </tr>
    <tr>
      <th>x26th_week</th>
      <td>36</td>
    </tr>
    <tr>
      <th>x27th_week</th>
      <td>29</td>
    </tr>
    <tr>
      <th>x28th_week</th>
      <td>24</td>
    </tr>
    <tr>
      <th>x29th_week</th>
      <td>20</td>
    </tr>
    <tr>
      <th>x30th_week</th>
      <td>20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>x47th_week</th>
      <td>5</td>
    </tr>
    <tr>
      <th>x48th_week</th>
      <td>4</td>
    </tr>
    <tr>
      <th>x49th_week</th>
      <td>4</td>
    </tr>
    <tr>
      <th>x50th_week</th>
      <td>4</td>
    </tr>
    <tr>
      <th>x51st_week</th>
      <td>4</td>
    </tr>
    <tr>
      <th>x52nd_week</th>
      <td>4</td>
    </tr>
    <tr>
      <th>x53rd_week</th>
      <td>4</td>
    </tr>
    <tr>
      <th>x54th_week</th>
      <td>2</td>
    </tr>
    <tr>
      <th>x55th_week</th>
      <td>2</td>
    </tr>
    <tr>
      <th>x56th_week</th>
      <td>2</td>
    </tr>
    <tr>
      <th>x57th_week</th>
      <td>2</td>
    </tr>
    <tr>
      <th>x58th_week</th>
      <td>2</td>
    </tr>
    <tr>
      <th>x59th_week</th>
      <td>2</td>
    </tr>
    <tr>
      <th>x60th_week</th>
      <td>2</td>
    </tr>
    <tr>
      <th>x61st_week</th>
      <td>2</td>
    </tr>
    <tr>
      <th>x62nd_week</th>
      <td>2</td>
    </tr>
    <tr>
      <th>x63rd_week</th>
      <td>2</td>
    </tr>
    <tr>
      <th>x64th_week</th>
      <td>2</td>
    </tr>
    <tr>
      <th>x65th_week</th>
      <td>1</td>
    </tr>
    <tr>
      <th>x66th_week</th>
      <td>0</td>
    </tr>
    <tr>
      <th>x67th_week</th>
      <td>0</td>
    </tr>
    <tr>
      <th>x68th_week</th>
      <td>0</td>
    </tr>
    <tr>
      <th>x69th_week</th>
      <td>0</td>
    </tr>
    <tr>
      <th>x70th_week</th>
      <td>0</td>
    </tr>
    <tr>
      <th>x71st_week</th>
      <td>0</td>
    </tr>
    <tr>
      <th>x72nd_week</th>
      <td>0</td>
    </tr>
    <tr>
      <th>x73rd_week</th>
      <td>0</td>
    </tr>
    <tr>
      <th>x74th_week</th>
      <td>0</td>
    </tr>
    <tr>
      <th>x75th_week</th>
      <td>0</td>
    </tr>
    <tr>
      <th>x76th_week</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>76 rows × 1 columns</p>
</div>



We find that zero artists have a hit song on the Billboard list after week 65.  Therefore, we remove the columns for weeks 66 through 76, which effectively caps the data at week 65.


```python
bb = bb[bb.columns[:71]]
bb.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>track</th>
      <th>time</th>
      <th>genre</th>
      <th>date_entered</th>
      <th>date_peaked</th>
      <th>x1st_week</th>
      <th>x2nd_week</th>
      <th>x3rd_week</th>
      <th>x4th_week</th>
      <th>...</th>
      <th>x56th_week</th>
      <th>x57th_week</th>
      <th>x58th_week</th>
      <th>x59th_week</th>
      <th>x60th_week</th>
      <th>x61st_week</th>
      <th>x62nd_week</th>
      <th>x63rd_week</th>
      <th>x64th_week</th>
      <th>x65th_week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Destiny's Child</td>
      <td>Independent Women Part I</td>
      <td>3,38,00 AM</td>
      <td>Rock</td>
      <td>September 23, 2000</td>
      <td>November 18, 2000</td>
      <td>78</td>
      <td>63</td>
      <td>49</td>
      <td>33</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Santana</td>
      <td>Maria, Maria</td>
      <td>4,18,00 AM</td>
      <td>Rock</td>
      <td>February 12, 2000</td>
      <td>April 8, 2000</td>
      <td>15</td>
      <td>8</td>
      <td>6</td>
      <td>5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Savage Garden</td>
      <td>I Knew I Loved You</td>
      <td>4,07,00 AM</td>
      <td>Rock</td>
      <td>October 23, 1999</td>
      <td>January 29, 2000</td>
      <td>71</td>
      <td>48</td>
      <td>43</td>
      <td>31</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 71 columns</p>
</div>



Now we inspect the 'time' column and observe that it does not make sense in its current format.  To handle this we write a function to convert the time into seconds, which we can more easily quantitatively analyze.  To accomplish the reformatting we call the function 'time_to_seconds' on the entire 'time' column by using the apply method in pandas.


```python
def time_to_seconds(x):
    x = x.split(',')
    return int(x[0])*60 + int(x[1])
bb['time'] = bb['time'].apply(time_to_seconds)
bb.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>track</th>
      <th>time</th>
      <th>genre</th>
      <th>date_entered</th>
      <th>date_peaked</th>
      <th>x1st_week</th>
      <th>x2nd_week</th>
      <th>x3rd_week</th>
      <th>x4th_week</th>
      <th>...</th>
      <th>x56th_week</th>
      <th>x57th_week</th>
      <th>x58th_week</th>
      <th>x59th_week</th>
      <th>x60th_week</th>
      <th>x61st_week</th>
      <th>x62nd_week</th>
      <th>x63rd_week</th>
      <th>x64th_week</th>
      <th>x65th_week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Destiny's Child</td>
      <td>Independent Women Part I</td>
      <td>218</td>
      <td>Rock</td>
      <td>September 23, 2000</td>
      <td>November 18, 2000</td>
      <td>78</td>
      <td>63</td>
      <td>49</td>
      <td>33</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Santana</td>
      <td>Maria, Maria</td>
      <td>258</td>
      <td>Rock</td>
      <td>February 12, 2000</td>
      <td>April 8, 2000</td>
      <td>15</td>
      <td>8</td>
      <td>6</td>
      <td>5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Savage Garden</td>
      <td>I Knew I Loved You</td>
      <td>247</td>
      <td>Rock</td>
      <td>October 23, 1999</td>
      <td>January 29, 2000</td>
      <td>71</td>
      <td>48</td>
      <td>43</td>
      <td>31</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 71 columns</p>
</div>



Next, we inspect the 'genre' column by calling its value_counts() method.


```python
bb['genre'].value_counts().to_frame()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rock</th>
      <td>103</td>
    </tr>
    <tr>
      <th>Country</th>
      <td>74</td>
    </tr>
    <tr>
      <th>Rap</th>
      <td>58</td>
    </tr>
    <tr>
      <th>Rock'n'roll</th>
      <td>34</td>
    </tr>
    <tr>
      <th>R&amp;B</th>
      <td>13</td>
    </tr>
    <tr>
      <th>R &amp; B</th>
      <td>10</td>
    </tr>
    <tr>
      <th>Pop</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Latin</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Electronica</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Gospel</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Jazz</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Reggae</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



There seem to be redundant genre types that have slightly different formating and/or wording.  Using the pandas apply method we call the function 'genre_fix' to combine similarly worded genre types that appear to be mislabeled (e.g., R & B and Rock'n'roll).  NOTE: Here we are interpreting the labels Rock and Rock'n'roll to be the same genre.


```python
def genre_fix(x):
    if x == 'R & B':
        return 'R&B'
    elif x == 'Rock\'n\'roll':
        return 'Rock'
    else:
        return x
bb['genre'] = bb['genre'].apply(genre_fix)
bb['genre'].value_counts().to_frame()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rock</th>
      <td>137</td>
    </tr>
    <tr>
      <th>Country</th>
      <td>74</td>
    </tr>
    <tr>
      <th>Rap</th>
      <td>58</td>
    </tr>
    <tr>
      <th>R&amp;B</th>
      <td>23</td>
    </tr>
    <tr>
      <th>Pop</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Latin</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Electronica</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Gospel</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Jazz</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Reggae</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Next we take a look at the columns of 'date_entered' and 'date_peaked'.  We find that they are in string format so we convert them to datetime format.


```python
bb['date_entered'] = pd.to_datetime(bb['date_entered'], format='%B %d, %Y')
bb['date_peaked'] = pd.to_datetime(bb['date_peaked'], format='%B %d, %Y')
```

Let's calculate the difference between the 'date_peaked' and 'date_entered' columns to figure out the time it takes each song to ascend to its top rank.


```python
bb.insert(6, 'delta_days_peaked', (bb['date_peaked']-bb['date_entered']).apply(lambda x: x / np.timedelta64(1,'D')).astype(int))
```

Let's also add a column to include the total number of weeks a song spends in the Top 100.  We'll call this column 'bb_weeks'.


```python
bb.insert(7, 'bb_weeks', bb.iloc[:,7:].apply(lambda x: x.count(), axis=1))
bb.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>track</th>
      <th>time</th>
      <th>genre</th>
      <th>date_entered</th>
      <th>date_peaked</th>
      <th>delta_days_peaked</th>
      <th>bb_weeks</th>
      <th>x1st_week</th>
      <th>x2nd_week</th>
      <th>...</th>
      <th>x56th_week</th>
      <th>x57th_week</th>
      <th>x58th_week</th>
      <th>x59th_week</th>
      <th>x60th_week</th>
      <th>x61st_week</th>
      <th>x62nd_week</th>
      <th>x63rd_week</th>
      <th>x64th_week</th>
      <th>x65th_week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Destiny's Child</td>
      <td>Independent Women Part I</td>
      <td>218</td>
      <td>Rock</td>
      <td>2000-09-23</td>
      <td>2000-11-18</td>
      <td>56</td>
      <td>28</td>
      <td>78</td>
      <td>63</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Santana</td>
      <td>Maria, Maria</td>
      <td>258</td>
      <td>Rock</td>
      <td>2000-02-12</td>
      <td>2000-04-08</td>
      <td>56</td>
      <td>26</td>
      <td>15</td>
      <td>8</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Savage Garden</td>
      <td>I Knew I Loved You</td>
      <td>247</td>
      <td>Rock</td>
      <td>1999-10-23</td>
      <td>2000-01-29</td>
      <td>98</td>
      <td>33</td>
      <td>71</td>
      <td>48</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 73 columns</p>
</div>



Now let's add one final column that calculates the mean ranking of each song over all the weeks (excluding NaN values).  We'll call this column 'avg_rank', but before we can do that we have to convert all the week columns into numeric values.


```python
bb.iloc[:,8:] = bb.iloc[:,8:].applymap(lambda x: pd.to_numeric(x, errors='ignore'))
```


```python
bb.insert(8, 'avg_rank', bb.iloc[:,8:].apply(np.nanmean,axis=1))
bb.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>track</th>
      <th>time</th>
      <th>genre</th>
      <th>date_entered</th>
      <th>date_peaked</th>
      <th>delta_days_peaked</th>
      <th>bb_weeks</th>
      <th>avg_rank</th>
      <th>x1st_week</th>
      <th>...</th>
      <th>x56th_week</th>
      <th>x57th_week</th>
      <th>x58th_week</th>
      <th>x59th_week</th>
      <th>x60th_week</th>
      <th>x61st_week</th>
      <th>x62nd_week</th>
      <th>x63rd_week</th>
      <th>x64th_week</th>
      <th>x65th_week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Destiny's Child</td>
      <td>Independent Women Part I</td>
      <td>218</td>
      <td>Rock</td>
      <td>2000-09-23</td>
      <td>2000-11-18</td>
      <td>56</td>
      <td>28</td>
      <td>14.821429</td>
      <td>78</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Santana</td>
      <td>Maria, Maria</td>
      <td>258</td>
      <td>Rock</td>
      <td>2000-02-12</td>
      <td>2000-04-08</td>
      <td>56</td>
      <td>26</td>
      <td>10.500000</td>
      <td>15</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Savage Garden</td>
      <td>I Knew I Loved You</td>
      <td>247</td>
      <td>Rock</td>
      <td>1999-10-23</td>
      <td>2000-01-29</td>
      <td>98</td>
      <td>33</td>
      <td>17.363636</td>
      <td>71</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 74 columns</p>
</div>



This concludes the data cleaning portion of this project.

## Exploratory analysis

Let's start the exploratory analysis by using Seaborn to create scatter plots and histograms of all relevant numeric data.


```python
sns.pairplot(bb.iloc[:,:9], hue='genre', palette="husl");
```


![png](/img/project-2_files/project-2_41_0.png)


From the above data we see a trend between the time it takes a song to peak ('delta_days_peaked') and the total number of weeks the song stays on the Billboard Hot 100 chart ('bb_weeks').  We also see a correlation between the mean rank of a song ('avg_rank') and the total number of weeks the song stays on the charts. We will keep both these trends in mind when we do our hypothesis testing.  The other thing that stands out on the above chart is the jump around week 20 of the 'bb_weeks' histogram.  Let's explore whether or not this anomaly is observed on a plot of number of songs versus week (after first entering the chart).


```python
bb.iloc[:,9:].apply(lambda x: x.count(), axis=0).reset_index().plot(kind='bar',figsize = (16,8), color='b', width = 1)
plt.xlabel('Week', fontsize = 18)
plt.ylabel('Number of Songs', fontsize = 18);
```


![png](/img/project-2_files/project-2_43_0.png)


Sure enough, we confirm the weird drop off in songs on the chart at around week 20.  This origin of this anomaly is unknown.


Now let's explore any possible effects of seasonality in the Billboard Hot 100 dataset.  If we plot the number of songs peaking on the charts as a function of month, we can see if there are any sudden upticks in popularity around the holidays.


```python
bb['date_peaked'].groupby(bb.date_peaked.dt.month).count().plot(kind="bar", color='b', width=.9)
plt.xlabel('Month')
plt.ylabel('Number of Songs')
plt.title(r'Songs with Peak Ranking on Billboard Chart, 2000');
```


![png](/img/project-2_files/project-2_46_0.png)


Here we observe seasonality effects with upticks in the months of December and January. This could be due to a change in music preferences around the holidays. In this case the surge seen in January would be a lagging indicator as any surge around Christmas time would be captured in the first week of January.  We will revisit this seasonality effect in our hypothesis testing to determine whether or not what we observe has both statistical significance and an impact on 'bb_weeks' or 'avg_rank'.  

Let's now turn to exploring the popularity of each music genre.


```python
bb_genre_percentages = bb['genre'].value_counts()/len(bb['genre'])*100
bb_genre_percentages.sort_values().plot(kind='barh', width=.9, color='b')
plt.xlabel('Percentage')
plt.ylabel('Genre')
plt.title(r'Percent of Songs on Billboard Chart by Genre, 2000');
```


![png](/img/project-2_files/project-2_49_0.png)


Here we find that Rock and Country combined account for more than two-thirds of all songs making an appearance on the 2000 Billboard Hot 100 chart.  We also observe that the six least popular genres account for less than 10% of all songs making a chart appearance.  In the next section we will explore whether or not a hit song that is categorized into one of the less popular music genres tends to have lower average rankings or shorter longevity on the charts.  

Let's now turn our attention to investigating the effects of "peaking early" in the charts.  


```python
bb['delta_days_peaked'].describe().to_frame()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>delta_days_peaked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>317.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>52.246057</td>
    </tr>
    <tr>
      <th>std</th>
      <td>40.867601</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>49.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>315.000000</td>
    </tr>
  </tbody>
</table>
</div>



The 50th percentile of 'delta_days_peaked' is 49 days. If we break that column down into two categories, namely 'peaked_early' (less than 49 days) and 'peaked_late' (more than 49 days) we can use a histogram to visualize the effects of peak timing on overall number of weeks on the chart and average ranking.


```python
early_peaks_weeks = bb[bb['delta_days_peaked']<=49]['bb_weeks']
late_peaks_weeks = bb[bb['delta_days_peaked']>49]['bb_weeks']
```


```python
bins = np.linspace(0, 60, 20)
plt.hist(early_peaks_weeks, bins, alpha=0.5, label='Early Peakers', color='r')
plt.hist(late_peaks_weeks, bins, alpha=0.5, label='Late Peakers', color ='b')
plt.xlabel('Weeks On the Billboard Chart')
plt.ylabel('Number of Songs')
plt.legend(loc='upper right');
```


![png](/img/project-2_files/project-2_55_0.png)



```python
early_peaks_ranks = bb[bb['delta_days_peaked']<=49]['avg_rank']
late_peaks_ranks = bb[bb['delta_days_peaked']>49]['avg_rank']
```


```python
bins = np.linspace(0, 100, 20)
plt.hist(early_peaks_ranks, bins, alpha=0.5, label='Early Peakers', color='r')
plt.hist(late_peaks_ranks, bins, alpha=0.5, label='Late Peakers', color ='b')
plt.xlabel('Average Ranking')
plt.ylabel('Number of Songs')
plt.legend(loc='upper left');
```


![png](/img/project-2_files/project-2_57_0.png)


The above two plots show clear delineations between songs that peak early and those that peak late, with the 7 week mark being the differentiator.  In the next section we will compare the two categories and discuss the statistical significance of their effects. 

The plot below shows a slight negative correlation between average ranking and number of weeks on the chart.  This makes sense as we would expect to see better ranked songs stay on the charts longer than worse ranked songs. 


```python
sns.jointplot(x='bb_weeks', y='avg_rank', data=bb, color='r').set_axis_labels('Number of Weeks', 'Average Rank');
```


![png](/img/project-2_files/project-2_60_0.png)


At the beginning of this project we also hypothesized that shorter songs may tend to have higher average rankings on the Billboard Hot 100 chart.  Since shorter songs are often more amenable to radio play as well as potential repeatability effects (i.e., hearing a song many times creates an increased affinity for that song despite initially disliking it).  However, the plot below demonstrates no relationship between average ranking and song length.  We will still test this hypothesis in the next section to confirm this claim.


```python
sns.jointplot(x='time', y='avg_rank', data=bb, color='r').set_axis_labels('Song Length (sec)', 'Average Rank');
```


![png](/img/project-2_files/project-2_62_0.png)


And lastly, we close the exploratory analysis phase by uncovering the answer to the question we've all been wondering: What was the most popular song of 2000?  The truth is there are two answers to this question.  There's the most popular song in terms of number of weeks on the charts and by average ranking on the charts.

The most popular song of 2000 by total number of weeks on the charts was Higher by Creed, which stayed on the charts for 57 weeks!


```python
bb[bb['bb_weeks']==np.max(bb['bb_weeks'])]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>track</th>
      <th>time</th>
      <th>genre</th>
      <th>date_entered</th>
      <th>date_peaked</th>
      <th>delta_days_peaked</th>
      <th>bb_weeks</th>
      <th>avg_rank</th>
      <th>x1st_week</th>
      <th>...</th>
      <th>x56th_week</th>
      <th>x57th_week</th>
      <th>x58th_week</th>
      <th>x59th_week</th>
      <th>x60th_week</th>
      <th>x61st_week</th>
      <th>x62nd_week</th>
      <th>x63rd_week</th>
      <th>x64th_week</th>
      <th>x65th_week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>46</th>
      <td>Creed</td>
      <td>Higher</td>
      <td>316</td>
      <td>Rock</td>
      <td>1999-09-11</td>
      <td>2000-07-22</td>
      <td>315</td>
      <td>57</td>
      <td>36.859649</td>
      <td>81</td>
      <td>...</td>
      <td>26.0</td>
      <td>29.0</td>
      <td>32.0</td>
      <td>39.0</td>
      <td>39.0</td>
      <td>43.0</td>
      <td>47.0</td>
      <td>50.0</td>
      <td>50.0</td>
      <td>49.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 74 columns</p>
</div>



The most popular song of 2000 by average ranking was Maria, Maria by Santana, which had an overall average ranking of 10.5!


```python
bb[bb['avg_rank']==np.min(bb['avg_rank'])]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>track</th>
      <th>time</th>
      <th>genre</th>
      <th>date_entered</th>
      <th>date_peaked</th>
      <th>delta_days_peaked</th>
      <th>bb_weeks</th>
      <th>avg_rank</th>
      <th>x1st_week</th>
      <th>...</th>
      <th>x56th_week</th>
      <th>x57th_week</th>
      <th>x58th_week</th>
      <th>x59th_week</th>
      <th>x60th_week</th>
      <th>x61st_week</th>
      <th>x62nd_week</th>
      <th>x63rd_week</th>
      <th>x64th_week</th>
      <th>x65th_week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Santana</td>
      <td>Maria, Maria</td>
      <td>258</td>
      <td>Rock</td>
      <td>2000-02-12</td>
      <td>2000-04-08</td>
      <td>56</td>
      <td>26</td>
      <td>10.5</td>
      <td>15</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 74 columns</p>
</div>



That concludes the exploratory data analysis portion of this project.  Now onto hypothesis testing!

## Hypothesis testing

The first finding we will test is whether or not the popularity of the genre impacts the overall longevity of a song on the charts.  In other words, does a popular song have a shorter lifespan on the charts if it comes from a less popular genre?  To test this we will compare the top three most popular music genres to the other genres to see if there is a difference in terms of the number of weeks a song lasts on the charts.  The null hypothesis will be that there is no difference in terms of longevity between a hit song coming from the Rock, Country or Rap genres as compared to the other music genres.  


```python
bb['genre'].value_counts()/len(bb['genre'])*100
```




    Rock           43.217666
    Country        23.343849
    Rap            18.296530
    R&B             7.255521
    Pop             2.839117
    Latin           2.839117
    Electronica     1.261830
    Gospel          0.315457
    Jazz            0.315457
    Reggae          0.315457
    Name: genre, dtype: float64



To test the null hypothesis we will use a two sample t-test.  


```python
stats.ttest_ind((bb[(bb['genre']=='Rock') | (bb['genre']=='Country') | (bb['genre']=='Rap')]['bb_weeks']),(bb[(bb['genre']!='Rock') & (bb['genre']!='Country') & (bb['genre']!='Rap')]['bb_weeks']))
```




    Ttest_indResult(statistic=2.0910257898375217, pvalue=0.037326725229114052)



We find that the p-value is 0.04, which is less than our significance level (0.05), therefore we reject the null hypothesis.  In other words, we conclude that hit songs coming from popular genres (such as Rock, Country or Rap) will tend to have longer lifespans on the chart than other hit songs coming from less popular genres.

Next, let's assess the phenomena of "peaking early".  As discussed previously, we define an "early peaker" as a song that reaches its best ranking within the first 7 weeks of entering the charts, and an "late peaker" as anything that peaks thereafter.  Because we already have values assigned to these variables from the previous section, we can go ahead and run the Student's t-test.


```python
stats.ttest_ind(early_peaks_weeks,late_peaks_weeks)
```




    Ttest_indResult(statistic=-14.20770095196071, pvalue=9.6012973865502593e-36)




```python
stats.ttest_ind(early_peaks_ranks,late_peaks_ranks)
```




    Ttest_indResult(statistic=11.465142028429195, pvalue=1.1468180988986631e-25)



From the above calculations we observe very low p-values and, therefore, again reject the null hypotheses.  In other words, from the t-tests we find there to be a high probability that songs that peak early will tend to have worse average rankings and fewer number of weeks on the charts as compared to songs that peak late. 

If you remember from earlier, we saw a jump in December and January in the number of songs peaking per month.  We thought that it might be an artifact of the holiday timing.  Let's explore this phenomenon further.  If we slice the data from the start of December to mid-January, we can measure whether or not the jump and its effects observed during that timeframe were statistically significant. 


```python
holiday_peak_weeks = bb[(bb['date_peaked']>'2000-12-01') | (bb['date_peaked']<='2000-01-15')]['bb_weeks']
other_peak_weeks = bb[(bb['date_peaked']<='2000-12-01') & (bb['date_peaked']>'2000-01-15')]['bb_weeks']
stats.ttest_ind(holiday_peak_weeks,other_peak_weeks)
```




    Ttest_indResult(statistic=-1.3012608949195783, pvalue=0.19412012711398596)




```python
holiday_peak_rank = bb[(bb['date_peaked']>'2000-12-01') | (bb['date_peaked']<='2000-01-15')]['avg_rank']
other_peak_rank = bb[(bb['date_peaked']<='2000-12-01') & (bb['date_peaked']>'2000-01-15')]['avg_rank']
stats.ttest_ind(holiday_peak_rank,other_peak_rank)
```




    Ttest_indResult(statistic=-0.091821503032108551, pvalue=0.92689822466562011)




```python
holiday_peak = bb[(bb['date_peaked']>'2000-12-01') | (bb['date_peaked']<='2000-01-15')]['delta_days_peaked']
other_peak = bb[(bb['date_peaked']<='2000-12-01') & (bb['date_peaked']>'2000-01-15')]['delta_days_peaked']
stats.ttest_ind(holiday_peak,other_peak)
```




    Ttest_indResult(statistic=-1.1275021135518761, pvalue=0.26038834354513546)



Based on the high p-values, we do not reject the null hypothesis for all hypothesis tests related to these winter seasonality observations.  We originally thought a brief surge in Christmas song popularity during the winter months could affect the average ranking and average number of weeks songs sustained on the chart.  The t-test results proved otherwise. Not only does this tell us that the jump in songs peaking in December and January had no statistically significant effect on average ranking or number of weeks songs stayed on the chart, but it tells us that the actual jump in songs peaking during those months was in itself not statistically significant compared to the other months.

In the beginning of this project we hypothesized that, perhaps, the length of a song may influence its average rank. The thinking was that shorter songs were more amenable to radio play and online streaming, which are believed to be key components to the Billboard Hot 100 ranking formula.  However, when we plot average ranking as a function of song length during the exploratory analysis section, it became obvious that a strong correlation was not present.  We will confirm this observation with a t-test calculation.  The null hypothesis is that the length of a song has no impact on its average ranking.  "Short songs" are defined to be songs that last 210sec or less, whereas "long songs" are considered to be 211sec or more. The somewhat arbitrary threshold for short/long songs was determined by doing some quick online research that 3:30 is the normal length of a typical song.


```python
short_songs = bb[(bb['time']<=210)]['avg_rank']
long_songs = bb[(bb['time']>210)]['avg_rank']
stats.ttest_ind(short_songs,long_songs)
```




    Ttest_indResult(statistic=0.20236561334325387, pvalue=0.8397615457833868)



This t-test yields a p-value of 0.83, which is much higher than our significance level of 0.05.  We thus do not reject the null hypothesis.  In other words, the t-test confirms that the length of a song has no impact on its average ranking.

In summary, we have uncovered several key characteristics and trends that could be useful in determining what exactly makes a successful song on the Billboard Hot 100 chart.  In this case we measure success both in terms of average ranking and number of weeks sustained on the chart.  We have found that just being a hit song is not good enough -- to truly have sustained success the song must be from a popular genre such as Rock, Country or Rap.  We have also concluded that the greatest songs on the chart tend not to peak early, but instead attain their best rank sometime after week 7.  Songs that peak late generally have better average rankings and stay on the charts longer.  Also, as we would expect, songs with higher average rankings tend to stay on the charts longer.  From our analysis we observed no statistically significant seasonality trends during the winter holiday months.  Lastly, we have also demonstrated that the length of a song has no discernible impact on song rank or longevity on the chart.

I hope you've had fun exploring some interesting trends in the Billboard Hot 100 chart dataset from the year 2000.  
