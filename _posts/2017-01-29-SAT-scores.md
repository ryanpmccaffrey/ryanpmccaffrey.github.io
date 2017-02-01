---
layout: post
image: '/img/sat.jpg'
title: SAT Scores in the United States
subtitle: "Category: Data Visualization"
---


The purpose of this project is to explore a dataset of SAT scores from around the country and explore ways to effectively visualize the data.  This data, taken from the College Board, gives the mean SAT math and verbal scores, and the participation rate for each state and the District of Columbia for the year 2001.  Exploratory analysis and plotting will be performed in Python.  Lastly, this project will culminate in a [D3](https://d3js.org/) visualization of SAT scores (combined) and participation rates mapped across the US.  


First, we start by loading all the required Python packages and reading in the data as a list.


```python
import numpy as np
import scipy.stats as stats
import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
my_csv_path = '../assets/sat_scores.csv'
sat_list = []
with open(my_csv_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        sat_list.append(row)
```

Next, we do a sanity check to make sure all states are accounted for.  The total number of rows should be 53 (50 states + 1 DC + 1 All + 1 column header).  


```python
len(sat_list)
```




    53



Although each state has posted mean Verbal and Math SAT scores, many of the states have low participation rates.  This finding is probably the result of the ACT test (a competitor to the SAT test) being popular among the southern and midwestern states.  In states where the ACT test dominates in popularity, the SAT scores may be biased or skewed toward the higher end due to the fact that students who take both exams tend to be more competitive and/or higher achieving students. With 20 states having participation rates of 20% or less, the data set would offer a fairer comparison if these states had higher participation rates. Lastly, the values of the SAT scores all seem reasonable (i.e., no entries with 0 or N/A).

Next, we remove the column header and create a key-value pair dictionary in case we ever want to perform further exploratory data analysis in the future.  For our dictionary we will use the state abbreviation as the key.  The values of the dictionary will be a list containing participation rate, verbal score, and math score.  We use a dictionary comprehension to create the dictionary. 


```python
labels = sat_list[0]
sat_list.remove(labels)
sat_dict = {i[0]:i[1:] for i in sat_list}

```

Next, we create a pandas dataframe to more easily handle and display the data.


```python
sat_df = pd.DataFrame(sat_list, columns=list(labels))
sat_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Rate</th>
      <th>Verbal</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CT</td>
      <td>82</td>
      <td>509</td>
      <td>510</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NJ</td>
      <td>81</td>
      <td>499</td>
      <td>513</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MA</td>
      <td>79</td>
      <td>511</td>
      <td>515</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NY</td>
      <td>77</td>
      <td>495</td>
      <td>505</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NH</td>
      <td>72</td>
      <td>520</td>
      <td>516</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RI</td>
      <td>71</td>
      <td>501</td>
      <td>499</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PA</td>
      <td>71</td>
      <td>500</td>
      <td>499</td>
    </tr>
    <tr>
      <th>7</th>
      <td>VT</td>
      <td>69</td>
      <td>511</td>
      <td>506</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ME</td>
      <td>69</td>
      <td>506</td>
      <td>500</td>
    </tr>
    <tr>
      <th>9</th>
      <td>VA</td>
      <td>68</td>
      <td>510</td>
      <td>501</td>
    </tr>
    <tr>
      <th>10</th>
      <td>DE</td>
      <td>67</td>
      <td>501</td>
      <td>499</td>
    </tr>
    <tr>
      <th>11</th>
      <td>MD</td>
      <td>65</td>
      <td>508</td>
      <td>510</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NC</td>
      <td>65</td>
      <td>493</td>
      <td>499</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GA</td>
      <td>63</td>
      <td>491</td>
      <td>489</td>
    </tr>
    <tr>
      <th>14</th>
      <td>IN</td>
      <td>60</td>
      <td>499</td>
      <td>501</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SC</td>
      <td>57</td>
      <td>486</td>
      <td>488</td>
    </tr>
    <tr>
      <th>16</th>
      <td>DC</td>
      <td>56</td>
      <td>482</td>
      <td>474</td>
    </tr>
    <tr>
      <th>17</th>
      <td>OR</td>
      <td>55</td>
      <td>526</td>
      <td>526</td>
    </tr>
    <tr>
      <th>18</th>
      <td>FL</td>
      <td>54</td>
      <td>498</td>
      <td>499</td>
    </tr>
    <tr>
      <th>19</th>
      <td>WA</td>
      <td>53</td>
      <td>527</td>
      <td>527</td>
    </tr>
    <tr>
      <th>20</th>
      <td>TX</td>
      <td>53</td>
      <td>493</td>
      <td>499</td>
    </tr>
    <tr>
      <th>21</th>
      <td>HI</td>
      <td>52</td>
      <td>485</td>
      <td>515</td>
    </tr>
    <tr>
      <th>22</th>
      <td>AK</td>
      <td>51</td>
      <td>514</td>
      <td>510</td>
    </tr>
    <tr>
      <th>23</th>
      <td>CA</td>
      <td>51</td>
      <td>498</td>
      <td>517</td>
    </tr>
    <tr>
      <th>24</th>
      <td>AZ</td>
      <td>34</td>
      <td>523</td>
      <td>525</td>
    </tr>
    <tr>
      <th>25</th>
      <td>NV</td>
      <td>33</td>
      <td>509</td>
      <td>515</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CO</td>
      <td>31</td>
      <td>539</td>
      <td>542</td>
    </tr>
    <tr>
      <th>27</th>
      <td>OH</td>
      <td>26</td>
      <td>534</td>
      <td>439</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MT</td>
      <td>23</td>
      <td>539</td>
      <td>539</td>
    </tr>
    <tr>
      <th>29</th>
      <td>WV</td>
      <td>18</td>
      <td>527</td>
      <td>512</td>
    </tr>
    <tr>
      <th>30</th>
      <td>ID</td>
      <td>17</td>
      <td>543</td>
      <td>542</td>
    </tr>
    <tr>
      <th>31</th>
      <td>TN</td>
      <td>13</td>
      <td>562</td>
      <td>553</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NM</td>
      <td>13</td>
      <td>551</td>
      <td>542</td>
    </tr>
    <tr>
      <th>33</th>
      <td>IL</td>
      <td>12</td>
      <td>576</td>
      <td>589</td>
    </tr>
    <tr>
      <th>34</th>
      <td>KY</td>
      <td>12</td>
      <td>550</td>
      <td>550</td>
    </tr>
    <tr>
      <th>35</th>
      <td>WY</td>
      <td>11</td>
      <td>547</td>
      <td>545</td>
    </tr>
    <tr>
      <th>36</th>
      <td>MI</td>
      <td>11</td>
      <td>561</td>
      <td>572</td>
    </tr>
    <tr>
      <th>37</th>
      <td>MN</td>
      <td>9</td>
      <td>580</td>
      <td>589</td>
    </tr>
    <tr>
      <th>38</th>
      <td>KS</td>
      <td>9</td>
      <td>577</td>
      <td>580</td>
    </tr>
    <tr>
      <th>39</th>
      <td>AL</td>
      <td>9</td>
      <td>559</td>
      <td>554</td>
    </tr>
    <tr>
      <th>40</th>
      <td>NE</td>
      <td>8</td>
      <td>562</td>
      <td>568</td>
    </tr>
    <tr>
      <th>41</th>
      <td>OK</td>
      <td>8</td>
      <td>567</td>
      <td>561</td>
    </tr>
    <tr>
      <th>42</th>
      <td>MO</td>
      <td>8</td>
      <td>577</td>
      <td>577</td>
    </tr>
    <tr>
      <th>43</th>
      <td>LA</td>
      <td>7</td>
      <td>564</td>
      <td>562</td>
    </tr>
    <tr>
      <th>44</th>
      <td>WI</td>
      <td>6</td>
      <td>584</td>
      <td>596</td>
    </tr>
    <tr>
      <th>45</th>
      <td>AR</td>
      <td>6</td>
      <td>562</td>
      <td>550</td>
    </tr>
    <tr>
      <th>46</th>
      <td>UT</td>
      <td>5</td>
      <td>575</td>
      <td>570</td>
    </tr>
    <tr>
      <th>47</th>
      <td>IA</td>
      <td>5</td>
      <td>593</td>
      <td>603</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SD</td>
      <td>4</td>
      <td>577</td>
      <td>582</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ND</td>
      <td>4</td>
      <td>592</td>
      <td>599</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MS</td>
      <td>4</td>
      <td>566</td>
      <td>551</td>
    </tr>
    <tr>
      <th>51</th>
      <td>All</td>
      <td>45</td>
      <td>506</td>
      <td>514</td>
    </tr>
  </tbody>
</table>
</div>



Let's check the datatypes of each column.


```python
print labels
print [type(sat_df[i][0]) for i in labels]
```

    ['State', 'Rate', 'Verbal', 'Math']
    [<type 'str'>, <type 'str'>, <type 'str'>, <type 'str'>]


After determining the 'Rate', 'Verbal' and 'Math' columns are of type string, we convert these columns to ints.


```python
sat_df['Rate'] = sat_df['Rate'].astype(int)
sat_df['Verbal'] = sat_df['Verbal'].astype(int)
sat_df['Math'] = sat_df['Math'].astype(int)
```

Now we create three distinct dictionaries each with 'State' as the key. The values of each dictionary are 'Rate', 'Verbal' and 'Math'.  We create these dictionaries to make it easy to perform further exploratory analysis in the future, if needed.


```python
dict_state_rate =  {k:v for k,v in zip(sat_df['State'],sat_df['Rate'])}
print dict_state_rate
```

    {'WA': 53, 'DE': 67, 'DC': 56, 'WI': 6, 'WV': 18, 'HI': 52, 'FL': 54, 'WY': 11, 'NH': 72, 'NJ': 81, 'NM': 13, 'TX': 53, 'LA': 7, 'NC': 65, 'ND': 4, 'NE': 8, 'TN': 13, 'NY': 77, 'PA': 71, 'RI': 71, 'NV': 33, 'VA': 68, 'CO': 31, 'AK': 51, 'AL': 9, 'AR': 6, 'VT': 69, 'IL': 12, 'GA': 63, 'IN': 60, 'IA': 5, 'OK': 8, 'AZ': 34, 'CA': 51, 'ID': 17, 'CT': 82, 'ME': 69, 'MD': 65, 'All': 45, 'MA': 79, 'OH': 26, 'UT': 5, 'MO': 8, 'MN': 9, 'MI': 11, 'KS': 9, 'MT': 23, 'MS': 4, 'SC': 57, 'KY': 12, 'OR': 55, 'SD': 4}



```python
dict_state_verbal = {k:v for k,v in zip(sat_df['State'],sat_df['Verbal'])}
print dict_state_verbal
```

    {'WA': 527, 'DE': 501, 'DC': 482, 'WI': 584, 'WV': 527, 'HI': 485, 'FL': 498, 'WY': 547, 'NH': 520, 'NJ': 499, 'NM': 551, 'TX': 493, 'LA': 564, 'NC': 493, 'ND': 592, 'NE': 562, 'TN': 562, 'NY': 495, 'PA': 500, 'RI': 501, 'NV': 509, 'VA': 510, 'CO': 539, 'AK': 514, 'AL': 559, 'AR': 562, 'VT': 511, 'IL': 576, 'GA': 491, 'IN': 499, 'IA': 593, 'OK': 567, 'AZ': 523, 'CA': 498, 'ID': 543, 'CT': 509, 'ME': 506, 'MD': 508, 'All': 506, 'MA': 511, 'OH': 534, 'UT': 575, 'MO': 577, 'MN': 580, 'MI': 561, 'KS': 577, 'MT': 539, 'MS': 566, 'SC': 486, 'KY': 550, 'OR': 526, 'SD': 577}



```python
dict_state_math = {k:v for k,v in zip(sat_df['State'],sat_df['Math'])}
print dict_state_math
```

    {'WA': 527, 'DE': 499, 'DC': 474, 'WI': 596, 'WV': 512, 'HI': 515, 'FL': 499, 'WY': 545, 'NH': 516, 'NJ': 513, 'NM': 542, 'TX': 499, 'LA': 562, 'NC': 499, 'ND': 599, 'NE': 568, 'TN': 553, 'NY': 505, 'PA': 499, 'RI': 499, 'NV': 515, 'VA': 501, 'CO': 542, 'AK': 510, 'AL': 554, 'AR': 550, 'VT': 506, 'IL': 589, 'GA': 489, 'IN': 501, 'IA': 603, 'OK': 561, 'AZ': 525, 'CA': 517, 'ID': 542, 'CT': 510, 'ME': 500, 'MD': 510, 'All': 514, 'MA': 515, 'OH': 439, 'UT': 570, 'MO': 577, 'MN': 589, 'MI': 572, 'KS': 580, 'MT': 539, 'MS': 551, 'SC': 488, 'KY': 550, 'OR': 526, 'SD': 582}


Now we compute some basic descriptive statistics using NumPy.


```python
print 'Rate min: ', np.min(sat_df['Rate'])
print 'Rate max: ', np.max(sat_df['Rate'])
print 'Rate median: ', np.median(sat_df['Rate'])
print 'Rate mean: ', round(np.mean(sat_df['Rate']),1)
print 'Rate std: ', round(np.std(sat_df['Rate']),1)
```

    Rate min:  4
    Rate max:  82
    Rate median:  33.5
    Rate mean:  37.2
    Rate std:  27.0



```python
print 'SAT Verbal min: ', np.min(sat_df['Verbal'])
print 'SAT Verbal max: ', np.max(sat_df['Verbal'])
print 'SAT Verbal median: ', np.median(sat_df['Verbal'])
print 'SAT Verbal mean: ', round(np.mean(sat_df['Verbal']),1)
print 'SAT Verbal std: ', round(np.std(sat_df['Verbal']),1)
```

    SAT Verbal min:  482
    SAT Verbal max:  593
    SAT Verbal median:  526.5
    SAT Verbal mean:  532.0
    SAT Verbal std:  32.9



```python
print 'SAT Math min: ', np.min(sat_df['Math'])
print 'SAT Math max: ', np.max(sat_df['Math'])
print 'SAT Math median: ', np.median(sat_df['Math'])
print 'SAT Math mean: ', round(np.mean(sat_df['Math']),1)
print 'SAT Math std: ', round(np.std(sat_df['Math']),1)
```

    SAT Math min:  439
    SAT Math max:  603
    SAT Math median:  521.0
    SAT Math mean:  531.5
    SAT Math std:  35.7


Using the Matplotlib package in Python we create histograms of each of the numeric columns.  


```python
plt.hist(sat_df['Rate'],10)
plt.xlabel('Participation Rate (%)')
plt.ylabel('Number of States')
plt.title(r'SAT Score Reporting, 2001')
plt.axis([0, 90, 0, 18])
plt.grid(True)
plt.show()
```


![png](/img/blog_project_1_files/blog_project_1_24_0.png)



```python
plt.hist(sat_df['Math'],12)
plt.xlabel('Average Math Score')
plt.ylabel('Number of States')
plt.title(r'United States SAT Scores, 2001')
plt.grid(True)
plt.show()
```


![png](/img/blog_project_1_files/blog_project_1_25_0.png)



```python
plt.hist(sat_df['Verbal'],12)
plt.xlabel('Average Verbal Score')
plt.ylabel('Number of States')
plt.title(r'United States SAT Scores, 2001')
plt.axis([470, 600, 0, 10])
plt.grid(True)
plt.show()
```


![png](/img/blog_project_1_files/blog_project_1_26_0.png)


Though working assumption was that the distributions of the SAT dataset would be normal, this is clearly not the case. The participation rate and SAT Verbal scores appear to be bimodal distributions and the SAT Math scores distribution appear to be positive skewed (with median < mean).

To better understand the relationships among the data we use the Seaborn package to create pair plots.


```python
sns.pairplot(sat_df)
plt.show()
```


![png](/img/blog_project_1_files/blog_project_1_29_0.png)



```python
sns.jointplot(x='Math', y='Verbal', data=sat_df, color='b')
plt.show()
```


![png](/img/blog_project_1_files/blog_project_1_30_0.png)



```python
sns.jointplot(x='Verbal', y='Rate', data=sat_df, color='b')
plt.show()
```


![png](/img/blog_project_1_files/blog_project_1_31_0.png)


There are a couple interesting relationships to note. First, we see a strong positive correlation between SAT Verbal and Math scores, with one notable outlier (OH). Ohio was the only state to have a mean SAT Math score of less than 450. This trend demonstrates that states with higher Math SAT scores tend to have higher Verbal SAT scores.  Second, we observe a negative correlation between SAT participation rate and SAT Verbal/Math scores.  This seems to in agreement with our hypothesis that low-performing students in ACT test dominated states will generally tend to forgo taking the SAT test. This leaves only the strongest performing students in these states to take both the ACT and SAT exams, thus skewing the SAT score distribution.  As a result, we find that higher participation rates correspond to lower SAT Verbal/Math scores, and vice versa.

Finally, using D3 we visualize the SAT participation rates and combined SAT scores with two Choropleth maps of the US.  These maps are made with Vida and are used to visually demonstrate the geographical trend of higher SAT scores being concentrated toward the center of the US, and higher SAT participation rates being concentrated toward the western and northeastern parts of the US.

<iframe src="https://vida.io/embed/rwhNanL8qXKHbfzb4?dashboard=1" width="1000" height="3700" seamless frameBorder="0" scrolling="no"></iframe>
