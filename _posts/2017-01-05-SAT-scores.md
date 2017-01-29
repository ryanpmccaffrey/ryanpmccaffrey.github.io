---
layout: post
image: '/img/sat.jpg'
title: SAT Scores in the United States
subtitle: "Category: Data Visualization"
---

This purpose of this project was to demonstrate blah blah blah.


# Project 1

## Step 1: Open the `sat_scores.csv` file. Investigate the data, and answer the questions below.



{% highlight python %}
import numpy as np
import scipy.stats as stats
import csv
import seaborn as sns
%matplotlib inline
{% highlight python %}

##### 1. What does the data describe?


```python
my_csv_path = '../assets/sat_scores.csv'
rows = []
with open(my_csv_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append(row)
        
print rows
```

    [['State', 'Rate', 'Verbal', 'Math'], ['CT', '82', '509', '510'], ['NJ', '81', '499', '513'], ['MA', '79', '511', '515'], ['NY', '77', '495', '505'], ['NH', '72', '520', '516'], ['RI', '71', '501', '499'], ['PA', '71', '500', '499'], ['VT', '69', '511', '506'], ['ME', '69', '506', '500'], ['VA', '68', '510', '501'], ['DE', '67', '501', '499'], ['MD', '65', '508', '510'], ['NC', '65', '493', '499'], ['GA', '63', '491', '489'], ['IN', '60', '499', '501'], ['SC', '57', '486', '488'], ['DC', '56', '482', '474'], ['OR', '55', '526', '526'], ['FL', '54', '498', '499'], ['WA', '53', '527', '527'], ['TX', '53', '493', '499'], ['HI', '52', '485', '515'], ['AK', '51', '514', '510'], ['CA', '51', '498', '517'], ['AZ', '34', '523', '525'], ['NV', '33', '509', '515'], ['CO', '31', '539', '542'], ['OH', '26', '534', '439'], ['MT', '23', '539', '539'], ['WV', '18', '527', '512'], ['ID', '17', '543', '542'], ['TN', '13', '562', '553'], ['NM', '13', '551', '542'], ['IL', '12', '576', '589'], ['KY', '12', '550', '550'], ['WY', '11', '547', '545'], ['MI', '11', '561', '572'], ['MN', '9', '580', '589'], ['KS', '9', '577', '580'], ['AL', '9', '559', '554'], ['NE', '8', '562', '568'], ['OK', '8', '567', '561'], ['MO', '8', '577', '577'], ['LA', '7', '564', '562'], ['WI', '6', '584', '596'], ['AR', '6', '562', '550'], ['UT', '5', '575', '570'], ['IA', '5', '593', '603'], ['SD', '4', '577', '582'], ['ND', '4', '592', '599'], ['MS', '4', '566', '551'], ['All', '45', '506', '514']]


This data describes the mean verbal and math SAT scores for each state (and DC) along with their associated participation rate.

##### 2. Does the data look complete? Are there any obvious issues with the observations?

The data do not look complete.  Although each state has mean verbal and math SAT scores posted, many of the states have low participation rates which means the data may be skewed or biased.  With 20 states having less than 20% of their schools reporting, the data set would have been more accurate and reliable if more states had higher participation rates.  The values of the SAT scores all seem reasonable (i.e., no entries with 0 or N/A).

##### 3. Create a data dictionary for the dataset.


```python
# Visualizing the data in lists
for i in rows[1:]:
    print i
```

    ['CT', '82', '509', '510']
    ['NJ', '81', '499', '513']
    ['MA', '79', '511', '515']
    ['NY', '77', '495', '505']
    ['NH', '72', '520', '516']
    ['RI', '71', '501', '499']
    ['PA', '71', '500', '499']
    ['VT', '69', '511', '506']
    ['ME', '69', '506', '500']
    ['VA', '68', '510', '501']
    ['DE', '67', '501', '499']
    ['MD', '65', '508', '510']
    ['NC', '65', '493', '499']
    ['GA', '63', '491', '489']
    ['IN', '60', '499', '501']
    ['SC', '57', '486', '488']
    ['DC', '56', '482', '474']
    ['OR', '55', '526', '526']
    ['FL', '54', '498', '499']
    ['WA', '53', '527', '527']
    ['TX', '53', '493', '499']
    ['HI', '52', '485', '515']
    ['AK', '51', '514', '510']
    ['CA', '51', '498', '517']
    ['AZ', '34', '523', '525']
    ['NV', '33', '509', '515']
    ['CO', '31', '539', '542']
    ['OH', '26', '534', '439']
    ['MT', '23', '539', '539']
    ['WV', '18', '527', '512']
    ['ID', '17', '543', '542']
    ['TN', '13', '562', '553']
    ['NM', '13', '551', '542']
    ['IL', '12', '576', '589']
    ['KY', '12', '550', '550']
    ['WY', '11', '547', '545']
    ['MI', '11', '561', '572']
    ['MN', '9', '580', '589']
    ['KS', '9', '577', '580']
    ['AL', '9', '559', '554']
    ['NE', '8', '562', '568']
    ['OK', '8', '567', '561']
    ['MO', '8', '577', '577']
    ['LA', '7', '564', '562']
    ['WI', '6', '584', '596']
    ['AR', '6', '562', '550']
    ['UT', '5', '575', '570']
    ['IA', '5', '593', '603']
    ['SD', '4', '577', '582']
    ['ND', '4', '592', '599']
    ['MS', '4', '566', '551']
    ['All', '45', '506', '514']



```python
# Removing the column labels and saving the data in a dictionary with the states as keys
sat_dict = {i[0]:i[1:] for i in rows[1:]}
sat_dict
```




    {'AK': ['51', '514', '510'],
     'AL': ['9', '559', '554'],
     'AR': ['6', '562', '550'],
     'AZ': ['34', '523', '525'],
     'All': ['45', '506', '514'],
     'CA': ['51', '498', '517'],
     'CO': ['31', '539', '542'],
     'CT': ['82', '509', '510'],
     'DC': ['56', '482', '474'],
     'DE': ['67', '501', '499'],
     'FL': ['54', '498', '499'],
     'GA': ['63', '491', '489'],
     'HI': ['52', '485', '515'],
     'IA': ['5', '593', '603'],
     'ID': ['17', '543', '542'],
     'IL': ['12', '576', '589'],
     'IN': ['60', '499', '501'],
     'KS': ['9', '577', '580'],
     'KY': ['12', '550', '550'],
     'LA': ['7', '564', '562'],
     'MA': ['79', '511', '515'],
     'MD': ['65', '508', '510'],
     'ME': ['69', '506', '500'],
     'MI': ['11', '561', '572'],
     'MN': ['9', '580', '589'],
     'MO': ['8', '577', '577'],
     'MS': ['4', '566', '551'],
     'MT': ['23', '539', '539'],
     'NC': ['65', '493', '499'],
     'ND': ['4', '592', '599'],
     'NE': ['8', '562', '568'],
     'NH': ['72', '520', '516'],
     'NJ': ['81', '499', '513'],
     'NM': ['13', '551', '542'],
     'NV': ['33', '509', '515'],
     'NY': ['77', '495', '505'],
     'OH': ['26', '534', '439'],
     'OK': ['8', '567', '561'],
     'OR': ['55', '526', '526'],
     'PA': ['71', '500', '499'],
     'RI': ['71', '501', '499'],
     'SC': ['57', '486', '488'],
     'SD': ['4', '577', '582'],
     'TN': ['13', '562', '553'],
     'TX': ['53', '493', '499'],
     'UT': ['5', '575', '570'],
     'VA': ['68', '510', '501'],
     'VT': ['69', '511', '506'],
     'WA': ['53', '527', '527'],
     'WI': ['6', '584', '596'],
     'WV': ['18', '527', '512'],
     'WY': ['11', '547', '545']}




```python
# sat_dict = sat_data.set_index('State').T.to_dict('list')
# sat_dict
```

## Step 2: Load the data.

##### 4. Load the data into a list of lists


```python
# Creating a list of lists including the column labels
sat_list = [i for i in rows]
sat_list
```




    [['State', 'Rate', 'Verbal', 'Math'],
     ['CT', '82', '509', '510'],
     ['NJ', '81', '499', '513'],
     ['MA', '79', '511', '515'],
     ['NY', '77', '495', '505'],
     ['NH', '72', '520', '516'],
     ['RI', '71', '501', '499'],
     ['PA', '71', '500', '499'],
     ['VT', '69', '511', '506'],
     ['ME', '69', '506', '500'],
     ['VA', '68', '510', '501'],
     ['DE', '67', '501', '499'],
     ['MD', '65', '508', '510'],
     ['NC', '65', '493', '499'],
     ['GA', '63', '491', '489'],
     ['IN', '60', '499', '501'],
     ['SC', '57', '486', '488'],
     ['DC', '56', '482', '474'],
     ['OR', '55', '526', '526'],
     ['FL', '54', '498', '499'],
     ['WA', '53', '527', '527'],
     ['TX', '53', '493', '499'],
     ['HI', '52', '485', '515'],
     ['AK', '51', '514', '510'],
     ['CA', '51', '498', '517'],
     ['AZ', '34', '523', '525'],
     ['NV', '33', '509', '515'],
     ['CO', '31', '539', '542'],
     ['OH', '26', '534', '439'],
     ['MT', '23', '539', '539'],
     ['WV', '18', '527', '512'],
     ['ID', '17', '543', '542'],
     ['TN', '13', '562', '553'],
     ['NM', '13', '551', '542'],
     ['IL', '12', '576', '589'],
     ['KY', '12', '550', '550'],
     ['WY', '11', '547', '545'],
     ['MI', '11', '561', '572'],
     ['MN', '9', '580', '589'],
     ['KS', '9', '577', '580'],
     ['AL', '9', '559', '554'],
     ['NE', '8', '562', '568'],
     ['OK', '8', '567', '561'],
     ['MO', '8', '577', '577'],
     ['LA', '7', '564', '562'],
     ['WI', '6', '584', '596'],
     ['AR', '6', '562', '550'],
     ['UT', '5', '575', '570'],
     ['IA', '5', '593', '603'],
     ['SD', '4', '577', '582'],
     ['ND', '4', '592', '599'],
     ['MS', '4', '566', '551'],
     ['All', '45', '506', '514']]




```python
# sat_list = [sat_data.columns.tolist()] + sat_data.values.tolist()
# sat_list
```

##### 5. Print the data


```python
# Printing the data
print sat_list
```

    [['State', 'Rate', 'Verbal', 'Math'], ['CT', '82', '509', '510'], ['NJ', '81', '499', '513'], ['MA', '79', '511', '515'], ['NY', '77', '495', '505'], ['NH', '72', '520', '516'], ['RI', '71', '501', '499'], ['PA', '71', '500', '499'], ['VT', '69', '511', '506'], ['ME', '69', '506', '500'], ['VA', '68', '510', '501'], ['DE', '67', '501', '499'], ['MD', '65', '508', '510'], ['NC', '65', '493', '499'], ['GA', '63', '491', '489'], ['IN', '60', '499', '501'], ['SC', '57', '486', '488'], ['DC', '56', '482', '474'], ['OR', '55', '526', '526'], ['FL', '54', '498', '499'], ['WA', '53', '527', '527'], ['TX', '53', '493', '499'], ['HI', '52', '485', '515'], ['AK', '51', '514', '510'], ['CA', '51', '498', '517'], ['AZ', '34', '523', '525'], ['NV', '33', '509', '515'], ['CO', '31', '539', '542'], ['OH', '26', '534', '439'], ['MT', '23', '539', '539'], ['WV', '18', '527', '512'], ['ID', '17', '543', '542'], ['TN', '13', '562', '553'], ['NM', '13', '551', '542'], ['IL', '12', '576', '589'], ['KY', '12', '550', '550'], ['WY', '11', '547', '545'], ['MI', '11', '561', '572'], ['MN', '9', '580', '589'], ['KS', '9', '577', '580'], ['AL', '9', '559', '554'], ['NE', '8', '562', '568'], ['OK', '8', '567', '561'], ['MO', '8', '577', '577'], ['LA', '7', '564', '562'], ['WI', '6', '584', '596'], ['AR', '6', '562', '550'], ['UT', '5', '575', '570'], ['IA', '5', '593', '603'], ['SD', '4', '577', '582'], ['ND', '4', '592', '599'], ['MS', '4', '566', '551'], ['All', '45', '506', '514']]


##### 6. Extract a list of the labels from the data, and remove them from the data.


```python
# Extracting the column labels
labels = sat_list[0]
print labels
# Removing the column labels and printing out data without labels
sat_list.remove(labels)
print sat_list
```

    ['State', 'Rate', 'Verbal', 'Math']
    [['CT', '82', '509', '510'], ['NJ', '81', '499', '513'], ['MA', '79', '511', '515'], ['NY', '77', '495', '505'], ['NH', '72', '520', '516'], ['RI', '71', '501', '499'], ['PA', '71', '500', '499'], ['VT', '69', '511', '506'], ['ME', '69', '506', '500'], ['VA', '68', '510', '501'], ['DE', '67', '501', '499'], ['MD', '65', '508', '510'], ['NC', '65', '493', '499'], ['GA', '63', '491', '489'], ['IN', '60', '499', '501'], ['SC', '57', '486', '488'], ['DC', '56', '482', '474'], ['OR', '55', '526', '526'], ['FL', '54', '498', '499'], ['WA', '53', '527', '527'], ['TX', '53', '493', '499'], ['HI', '52', '485', '515'], ['AK', '51', '514', '510'], ['CA', '51', '498', '517'], ['AZ', '34', '523', '525'], ['NV', '33', '509', '515'], ['CO', '31', '539', '542'], ['OH', '26', '534', '439'], ['MT', '23', '539', '539'], ['WV', '18', '527', '512'], ['ID', '17', '543', '542'], ['TN', '13', '562', '553'], ['NM', '13', '551', '542'], ['IL', '12', '576', '589'], ['KY', '12', '550', '550'], ['WY', '11', '547', '545'], ['MI', '11', '561', '572'], ['MN', '9', '580', '589'], ['KS', '9', '577', '580'], ['AL', '9', '559', '554'], ['NE', '8', '562', '568'], ['OK', '8', '567', '561'], ['MO', '8', '577', '577'], ['LA', '7', '564', '562'], ['WI', '6', '584', '596'], ['AR', '6', '562', '550'], ['UT', '5', '575', '570'], ['IA', '5', '593', '603'], ['SD', '4', '577', '582'], ['ND', '4', '592', '599'], ['MS', '4', '566', '551'], ['All', '45', '506', '514']]



```python
# labels = sat_data.columns.tolist()
# labels
# sat_list = sat_data.values.tolist()
# sat_list
```

##### 7. Create a list of State names extracted from the data. (Hint: use the list of labels to index on the State column)


```python
# Convert data to dataframe
import pandas as pd
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




```python
# Extract 'State" column from dataframe
print list(sat_df['State'])
```

    ['CT', 'NJ', 'MA', 'NY', 'NH', 'RI', 'PA', 'VT', 'ME', 'VA', 'DE', 'MD', 'NC', 'GA', 'IN', 'SC', 'DC', 'OR', 'FL', 'WA', 'TX', 'HI', 'AK', 'CA', 'AZ', 'NV', 'CO', 'OH', 'MT', 'WV', 'ID', 'TN', 'NM', 'IL', 'KY', 'WY', 'MI', 'MN', 'KS', 'AL', 'NE', 'OK', 'MO', 'LA', 'WI', 'AR', 'UT', 'IA', 'SD', 'ND', 'MS', 'All']



```python
# sat_data['State']
```

##### 8. Print the types of each column


```python
#sat_dict['State']
```


```python
# Access the first element of each column and print out the data type
print labels
print [type(sat_df[i][0]) for i in labels]
```

    ['State', 'Rate', 'Verbal', 'Math']
    [<type 'str'>, <type 'str'>, <type 'str'>, <type 'str'>]


##### 9. Do any types need to be reassigned? If so, go ahead and do it.


```python
# Convert 'Rate', 'Verbal' and 'Math' columns to int datatypes
sat_df['Rate'] = sat_df['Rate'].astype(int)
sat_df['Verbal'] = sat_df['Verbal'].astype(int)
sat_df['Math'] = sat_df['Math'].astype(int)
```

##### 10. Create a dictionary for each column mapping the State to its respective value for that column. 


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


##### 11. Create a dictionary with the values for each of the numeric columns


```python
# Ask for clarification on this one
```

## Step 3: Describe the data

##### 12. Print the min and max of each column


```python
print 'Rate min: ', sat_df['Rate'].min()
print 'Rate max: ', sat_df['Rate'].max()
```

    Rate min:  4
    Rate max:  82



```python
print 'SAT Verbal min: ', sat_df['Verbal'].min()
print 'SAT Verbal max: ', sat_df['Verbal'].max()
```

    SAT Verbal min:  482
    SAT Verbal max:  593



```python
print 'SAT Math min: ', sat_df['Math'].min()
print 'SAT Math max: ', sat_df['Math'].max()
```

    SAT Math min:  439
    SAT Math max:  603



```python

```


```python

```


```python

```

##### 13. Write a function using only list comprehensions, no loops, to compute Standard Deviation. Print the Standard Deviation of each numeric column.


```python
import numpy as np
print 'Rate std: ', np.std(sat_df['Rate'])
print 'SAT Verbal std: ', np.std(sat_df['Verbal'])
print 'SAT Math std: ', np.std(sat_df['Math'])
```

    Rate std:  27.0379964945
    SAT Verbal std:  32.9150949616
    SAT Math std:  35.6669961643


## Step 4: Visualize the data

##### 14. Using MatPlotLib and PyPlot, plot the distribution of the Rate using histograms.


```python
import matplotlib.pyplot as plt
#fig, ax = plt.subplots(figsize=(15,7))

plt.hist(sat_df['Rate'],10)
plt.xlabel('Participation Rate (%)')
plt.ylabel('Number of States')
plt.title(r'SAT Score Reporting, 2001')
plt.axis([0, 90, 0, 18])
plt.grid(True)

# states = {'NY': 25, 'NJ': 10, 'NE': 15, 'NH': 7}

# x = range(len(states))
# y = states.values()

# # set the title, make it bigger, and move it up in the y-direction
# ax.set_title('Number by State', fontsize=18, y=1.01)

# # set the x tick labels and center
# ax.set_xticks(x)
# ax.set_xticklabels(states.keys())

# # add some margin
# ax.set_ylim(0, max(states.values()) * 1.1)
# ax.set_xlim(-1,len(states))

# # chang the color
# plt.bar(x, y, align='center', color='r')
```


![png](project_1_files/project_1_47_0.png)



```python
sns.distplot(sat_df['Rate'],10)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x118976250>




![png](project_1_files/project_1_48_1.png)


##### 15. Plot the Math distribution


```python
plt.hist(sat_df['Math'],12)
plt.xlabel('Average Math Score')
plt.ylabel('Number of States')
plt.title(r'United States SAT Scores, 2001')
#plt.axis([400, 650, 0, 12])
plt.grid(True)
```


![png](project_1_files/project_1_50_0.png)



```python
sns.distplot(sat_df['Math'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x118b58d90>




![png](project_1_files/project_1_51_1.png)


##### 16. Plot the Verbal distribution


```python
plt.hist(sat_df['Verbal'],12)
plt.xlabel('Average Verbal Score')
plt.ylabel('Number of States')
plt.title(r'United States SAT Scores, 2001')
#plt.axis([450, 650, 0, 12])
plt.grid(True)
```


![png](project_1_files/project_1_53_0.png)



```python
sns.distplot(sat_df['Verbal'],12)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x119783f10>




![png](project_1_files/project_1_54_1.png)



```python

```

##### 17. What is the typical assumption for data distribution?

The typical assumption for most data distributions is that they are normal distributions.  This was not the case.  

##### 18. Does that distribution hold true for our data?

The participation rate and SAT Verbal scores appeared to be bimodal distributions and the SAT Math scores distribution appeared to be positive skewed (median < mean).  

##### 19. Plot some scatterplots. **BONUS**: Use a PyPlot `figure` to present multiple plots at once.


```python
plt.scatter(sat_df['Rate'],sat_df['Verbal'])
plt.xlabel('State Participation Rate')
plt.ylabel('Average Verbal SAT Score')
plt.title(r'United States SAT Scores, 2001')
```




    <matplotlib.text.Text at 0x1218eb410>




![png](project_1_files/project_1_61_1.png)



```python
plt.scatter(sat_df['Rate'],sat_df['Math'])
plt.xlabel('State Participation Rate')
plt.ylabel('Average Math SAT Score')
plt.title(r'United States SAT Scores, 2001')
```




    <matplotlib.text.Text at 0x120f2fa90>




![png](project_1_files/project_1_62_1.png)



```python
plt.scatter(sat_df['Verbal'],sat_df['Math'])
plt.xlabel('Average Verbal SAT Score')
plt.ylabel('Average Math SAT Score')
plt.title(r'United States SAT Scores, 2001')
```




    <matplotlib.text.Text at 0x11e2b4310>




![png](project_1_files/project_1_63_1.png)



```python
sns.pairplot(sat_df)
```




    <seaborn.axisgrid.PairGrid at 0x11ff5c510>




![png](project_1_files/project_1_64_1.png)


##### 20. Are there any interesting relationships to note?

There are a couple interesting relationships to note. First, we see a strong positive correlation between SAT Verbal and Math scores, with one notable outlier (OH).  Ohio was the only state to have an average SAT Math score of less than 450.  Second, we observe a negative correlation between SAT participation rate and SAT Verbal and Math scores.  [Fill in explanation for this trend here.]


```python
sns.jointplot(x='Math', y='Verbal', data=sat_df)
```




    <seaborn.axisgrid.JointGrid at 0x122087050>




![png](project_1_files/project_1_67_1.png)



```python
sns.jointplot(x='', y="yearly_salary", data=salary)
```

##### 21. Create box plots for each variable. 


```python
sns.jointplot(x='Verbal', y='Rate', data=sat_df)
```




    <seaborn.axisgrid.JointGrid at 0x1229e5050>




![png](project_1_files/project_1_70_1.png)



```python
sns.boxplot(sat_df['Rate'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x122a5e350>




![png](project_1_files/project_1_71_1.png)



```python
sns.boxplot(sat_df['Verbal'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x121bb9b50>




![png](project_1_files/project_1_72_1.png)



```python
sns.boxplot(sat_df['Math'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x121e8e850>




![png](project_1_files/project_1_73_1.png)


##### BONUS: Using Tableau, create a heat map for each variable using a map of the US. 


```python

```
