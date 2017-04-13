---
title: Investigating a Dataset
date: 2016-10-01 01:00:00 +03:00
header:
  overlay_image: https://www.newsroom24bd.org/images/cms-image-000006011.jpg
  teaser: https://www.newsroom24bd.org/images/cms-image-000006011.jpg
  overlay_filter: 0.5
excerpt: "Exploring information from passengers and crew on board the Titanic to identify the factors that affected survival rate."
tags:
- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter notebook
description: Posed a question about a dataset, then used NumPy and Pandas to answer
  that question based on the data and created a report to share the results.
comments: true
share: true
layout: single
---

{% include toc %}

# Introduction

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.


One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

**In the current Project, a dataset contains demographics and passenger information from 891 of the 2224 passengers and crew on board the Titanic is examined. The obvious question that will be answered by the end of the Project is which were the factors that made people more likely to survive.**

# About the Data

The Dataset is a highly structured dataset consisted of the following attributes:

VARIABLE DESCRIPTIONS:

|Variable|Description                                                         |
|--------|--------------------------------------------------------------------|
|survival|Survival (0 = No; 1 = Yes)                                          |
|pclass  |Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)                         |
|name    |Name                                                                |
|sex     |Sex                                                                 |
|age     |Age                                                                 |
|sibsp   |Number of Siblings/Spouses Aboard                                   |
|parch   |Number of Parents/Children Aboard                                   |
|ticket  |Ticket Number                                                       |
|fare    |Passenger Fare                                                      |
|cabin   |Cabin                                                               |
|embarked|Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)|

SPECIAL NOTES:
Pclass is a proxy for socio-economic status (SES)  
1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

Age is in Years; Fractional if Age less than One (1)  
If the Age is Estimated, it is in the form xx.5  

With respect to the family relation variables (i.e. sibsp and parch) some relations were ignored.  The following are the definitions used for sibsp and parch.  

* Sibling: Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic  
* Spouse:  Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)  
* Parent:  Mother or Father of Passenger Aboard Titanic  
* Child:   Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic  

Other family relatives excluded from this study include cousins, nephews/nieces, aunts/uncles, and in-laws.  Some children traveled only with a nanny, therefore parch=0 for them.  As well, some travelled with very close friends or neighbors in a village, however, the definitions do not support such relations.

## Loading the Dataset and the necessary libraries


```python
%matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import math
import numpy as np
from IPython.display import Image
from scipy.stats import norm
from scipy.stats import stats
```


```python
#Loading the data to a dataframe
#"titanic_original" will be the initial dataframe.
#All following dataframes will be alterations of "titanic_original"
filename = "titanic_data.csv"
titanic_original = pd.read_csv(filename)
```

# Preparing the Data

Before trying any data cleaning, let's visualize the Dataset and check the Data Types of its fields.


```python
#Previewing the data
titanic_original.head()
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



We will rename the "Embarked" column to "Port" because we want to reserve the "Embarked" as a variable name for the number of passengers from a specific that embarked (got on-board) on the ship.  
We will create a new DataFrame named "titanic_df" and we will use this for the rest of the analysis.


```python
titanic_df=titanic_original.rename(columns = {'Embarked':'Port'})

```

## Checking Data Types


```python
#Print the data types of each column
titanic_df.dtypes
```




    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Port            object
    dtype: object



The Data Types are the expected, so there is no need for any corrections on this level.

## Checking Completeness

Next, we will check the Dataset for any missing values.


```python
#Count the number of values on each column
titanic_df.count()
```




    PassengerId    891
    Survived       891
    Pclass         891
    Name           891
    Sex            891
    Age            714
    SibSp          891
    Parch          891
    Ticket         891
    Fare           891
    Cabin          204
    Port           889
    dtype: int64



There are some missing values in **Age**, **Cabin** and **Port of Embarkation** variables.  

We will not drop any records right now. Any drops will take place when we will examine the specific factors.

## Investigating Data Problems

Next, we will check for any surprising values in our Dataset. The expected values for each variable are listed in the following table:  
  
|Variable   |Expected Data                                                                                    |
|-----------|-------------------------------------------------------------------------------------------------|
|PassengerId|Continuous Integers, starting from "1" and ending to "891"                                       |
|Survived   |Integer of values "0" or "1"                                                                     |
|Pclass     |Integer of values "1", "2", or "3"                                                               |
|Name       |(Nothing to check here)                                                                          |
|Sex        |"male" or "female"                                                                               |
|Age        |Min and Max values should make sense                                                             |
|SibSp      |Min and Max values should make sense                                                             |
|Parch      |Min and Max values should make sense                                                             |
|Ticket     |(Nothing to check here)                                                                          |
|Fare       |Min and Max values should make sense                                                             |
|Cabin      |There should be one value per record and the values should be in the format DeckCabin# (e.g C128)|
|Embarked   |The values should be either "C", "Q", or "S"                                                     |

### PassengerId


```python
#Calculating the min/max values, the # of values and the existance of duplicates
min_val = titanic_df["PassengerId"].min()
max_val = titanic_df["PassengerId"].max()
num_val = titanic_df["PassengerId"].count()
dup_val = titanic_df.duplicated(subset=["PassengerId"]).sum()

d = [min_val, max_val, num_val, dup_val]
i = ["Min Value", "Max Value", "Number of values", "Duplicate values"]

df = pd.DataFrame({"PassengerId":d}, index=i)
df
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Min Value</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Max Value</th>
      <td>891</td>
    </tr>
    <tr>
      <th>Number of values</th>
      <td>891</td>
    </tr>
    <tr>
      <th>Duplicate values</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Since the minimum value is *1*, the maximum *891*, with 891 entries and no duplicates, the *PassengerId* is a continuous integer from 1 to 891.  
  
-----  
  
### Survived


```python
#Finding unique values in "Survived" column
titanic_df["Survived"].unique()
```




    array([0, 1])



The *Survived* column contains the expected values.  

It may sound a good idea to turn this variable to a boolean, but letting it as a integer will help us later to calculate the survival rates.  
More specifically, since "0" indicates non-survival and "1" survival, the average "Survived" of a sample (e.g. a group of passengers) equals the Survival Rate of the sample.  
  
---  
  
### Pclass


```python
#Finding unique values in "Pclass" column
titanic_df["Pclass"].unique()
```




    array([3, 1, 2])



The *Pclass* column contains the expected values.  
  
---  
  
### Sex


```python
#Finding unique values in "Sex" column
titanic_df["Sex"].unique()
```




    array(['male', 'female'], dtype=object)



The *Sex* column contains the expected values.  
  
---

### Age


```python
#Finding Min/Max values in "Age" column
min_val = titanic_df["Age"].min()
max_val = titanic_df["Age"].max()

d = [min_val, max_val]
i=["Min Value", "Max Value"]

df = pd.DataFrame({"Age":d}, index=i)
df
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Min Value</th>
      <td>0.42</td>
    </tr>
    <tr>
      <th>Max Value</th>
      <td>80.00</td>
    </tr>
  </tbody>
</table>
</div>



The *Age* column contains non surprising values.  
  
---  
  
### SibSp


```python
#Finding Min/Max values in "SibSp" column
min_val = titanic_df["SibSp"].min()
max_val = titanic_df["SibSp"].max()

d = [min_val, max_val]
i=["Min Value", "Max Value"]

df = pd.DataFrame({"SibSp":d}, index=i)
df
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SibSp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Min Value</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Max Value</th>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



The *SibSp* column contains non surprising values. 
  
---  
  
### Parch


```python
#Finding Min/Max values in "Parch" column
min_val = titanic_df["Parch"].min()
max_val = titanic_df["Parch"].max()

d = [min_val, max_val]
i=["Min Value", "Max Value"]

df = pd.DataFrame({"Parch":d}, index=i)
df
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Parch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Min Value</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Max Value</th>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



The *Parch* column contains non surprising values. 
  
---  
  
### Fare


```python
#Finding Min/Max values in "Fare" column
min_val = titanic_df["Fare"].min()
max_val = titanic_df["Fare"].max()

d = [min_val, max_val]
i=["Min Value", "Max Value"]

df = pd.DataFrame({"Fare":d}, index=i)
df
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Min Value</th>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>Max Value</th>
      <td>512.3292</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Finding the # of "0" fare records
(titanic_df["Fare"] == 0).astype(int).sum()
```




    15



We can see that there are 15 "0" fares which looks strange.  
Let's take a closer look on these records:


```python
#Return all records with "0" fare
titanic_df[titanic_df["Fare"] == 0]
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Port</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>179</th>
      <td>180</td>
      <td>0</td>
      <td>3</td>
      <td>Leonard, Mr. Lionel</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>263</th>
      <td>264</td>
      <td>0</td>
      <td>1</td>
      <td>Harrison, Mr. William</td>
      <td>male</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>112059</td>
      <td>0.0</td>
      <td>B94</td>
      <td>S</td>
    </tr>
    <tr>
      <th>271</th>
      <td>272</td>
      <td>1</td>
      <td>3</td>
      <td>Tornquist, Mr. William Henry</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>277</th>
      <td>278</td>
      <td>0</td>
      <td>2</td>
      <td>Parkes, Mr. Francis "Frank"</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>302</th>
      <td>303</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. William Cahoone Jr</td>
      <td>male</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>413</th>
      <td>414</td>
      <td>0</td>
      <td>2</td>
      <td>Cunningham, Mr. Alfred Fleming</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>466</th>
      <td>467</td>
      <td>0</td>
      <td>2</td>
      <td>Campbell, Mr. William</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>481</th>
      <td>482</td>
      <td>0</td>
      <td>2</td>
      <td>Frost, Mr. Anthony Wood "Archie"</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239854</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>597</th>
      <td>598</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. Alfred</td>
      <td>male</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>633</th>
      <td>634</td>
      <td>0</td>
      <td>1</td>
      <td>Parr, Mr. William Henry Marsh</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112052</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>674</th>
      <td>675</td>
      <td>0</td>
      <td>2</td>
      <td>Watson, Mr. Ennis Hastings</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239856</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>732</th>
      <td>733</td>
      <td>0</td>
      <td>2</td>
      <td>Knight, Mr. Robert J</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239855</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>806</th>
      <td>807</td>
      <td>0</td>
      <td>1</td>
      <td>Andrews, Mr. Thomas Jr</td>
      <td>male</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>112050</td>
      <td>0.0</td>
      <td>A36</td>
      <td>S</td>
    </tr>
    <tr>
      <th>815</th>
      <td>816</td>
      <td>0</td>
      <td>1</td>
      <td>Fry, Mr. Richard</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112058</td>
      <td>0.0</td>
      <td>B102</td>
      <td>S</td>
    </tr>
    <tr>
      <th>822</th>
      <td>823</td>
      <td>0</td>
      <td>1</td>
      <td>Reuchlin, Jonkheer. John George</td>
      <td>male</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>19972</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



There are some obvious similarities.  
All of them were males, embarked in Southampton and only one survived.  
The above facts make them look like crew members but further investigation to https://www.encyclopedia-titanica.org/ reveals that they were not.  
Instead, it seems that they had some relation with the White Star Line, owner of RMS Titanic.  

For example,  

**Mr Francis Parkes** was a member of *Harland & Wolff : Titanic Guarantee Group*, a Belfast team sent by shipbuilders Harland & Wolff to accompany the Titanic on her maiden voyage.  

Researching for **Leonard, Mr. Lionel** led me to the following reference:  
"*It is believed Shannon worked for the American Line and possibly held US citizenship, using the name **Lionel Leonard** for reasons unknown. By 1912 he was quartermaster of the SS Philadelphia but the coal strike caused scheduling problems and Philadelphia"s westbound voyage was canceled, with Andrew and several other shipmates (August Johnson, **William Cahoone Jr. Johnson**, Alfred John Carver, Thomas Storey and  **William Henry Törnquist**) forced to travel aboard Titanic as passengers.*" (https://www.encyclopedia-titanica.org/titanic-victim/lionel-leonard.html)  

The above facts lead to the conclusion that the "zero fare" passengers had some relation with the ship owner company and were traveling for free.  
(A detailed investigation of all the above names would be out of scope of the specific project.)  
  
----  
  
### Cabin


```python
#Counting the number of cabins in each entry
titanic_df["Cabin"].str.split(" ", expand=True).count().rename(lambda x: x+1)
```




    1    204
    2     24
    3      8
    4      2
    dtype: int64



We can see that:  
2   passengers have 4 registered cabins  
6   passengers have 3 registered cabins (8-2)  
14  passengers have 2 registered cabins (24-8-2)  
170 passengers have 1 registered cabin (204-24-8-2)

To further examine the *Cabin* data we will export them from the dataframe 


```python
#Extracting, removing empty and splitting entries
cabin = titanic_df["Cabin"]
cabin = cabin.dropna()
cabin = cabin.str.split(" ", expand=True)

#As an example, print the entries that have 3 cabins.
cabin.dropna(subset=[1,2])
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>C23</td>
      <td>C25</td>
      <td>C27</td>
      <td>None</td>
    </tr>
    <tr>
      <th>88</th>
      <td>C23</td>
      <td>C25</td>
      <td>C27</td>
      <td>None</td>
    </tr>
    <tr>
      <th>311</th>
      <td>B57</td>
      <td>B59</td>
      <td>B63</td>
      <td>B66</td>
    </tr>
    <tr>
      <th>341</th>
      <td>C23</td>
      <td>C25</td>
      <td>C27</td>
      <td>None</td>
    </tr>
    <tr>
      <th>438</th>
      <td>C23</td>
      <td>C25</td>
      <td>C27</td>
      <td>None</td>
    </tr>
    <tr>
      <th>679</th>
      <td>B51</td>
      <td>B53</td>
      <td>B55</td>
      <td>None</td>
    </tr>
    <tr>
      <th>742</th>
      <td>B57</td>
      <td>B59</td>
      <td>B63</td>
      <td>B66</td>
    </tr>
    <tr>
      <th>872</th>
      <td>B51</td>
      <td>B53</td>
      <td>B55</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



Passengers with PassengerIds 27, 88, 341 and 438 looks to occupy the same cabins.


```python
titanic_df.loc[[27, 88, 341, 438]]
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Port</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Charles Alexander</td>
      <td>male</td>
      <td>19.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>88</th>
      <td>89</td>
      <td>1</td>
      <td>1</td>
      <td>Fortune, Miss. Mabel Helen</td>
      <td>female</td>
      <td>23.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>341</th>
      <td>342</td>
      <td>1</td>
      <td>1</td>
      <td>Fortune, Miss. Alice Elizabeth</td>
      <td>female</td>
      <td>24.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>438</th>
      <td>439</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Mark</td>
      <td>male</td>
      <td>64.0</td>
      <td>1</td>
      <td>4</td>
      <td>19950</td>
      <td>263.0</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



It seems that some families had booked more than one adjacent cabins.   
We assume that there is nothing wrong with the data.
  
---  
  
### Embarked


```python
titanic_df["Port"].unique()
```




    array(['S', 'C', 'Q', nan], dtype=object)



We were expecting some null values, so everything looks good here.  
  
---  
  
According to the above findings, no problematic data found, thus there isn't any wrangling actions to perform.

# Data Exploration

In the current section we will investigate the correlation of several factors with the Survival Rate.  
An initial investigation can be made between the non-categorical data by using the *pandas.DataFrame.corr()* function.


```python
titanic_df.corr()
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PassengerId</th>
      <td>1.000000</td>
      <td>-0.005007</td>
      <td>-0.035144</td>
      <td>0.036847</td>
      <td>-0.057527</td>
      <td>-0.001652</td>
      <td>0.012658</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>-0.005007</td>
      <td>1.000000</td>
      <td>-0.338481</td>
      <td>-0.077221</td>
      <td>-0.035322</td>
      <td>0.081629</td>
      <td>0.257307</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>-0.035144</td>
      <td>-0.338481</td>
      <td>1.000000</td>
      <td>-0.369226</td>
      <td>0.083081</td>
      <td>0.018443</td>
      <td>-0.549500</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.036847</td>
      <td>-0.077221</td>
      <td>-0.369226</td>
      <td>1.000000</td>
      <td>-0.308247</td>
      <td>-0.189119</td>
      <td>0.096067</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>-0.057527</td>
      <td>-0.035322</td>
      <td>0.083081</td>
      <td>-0.308247</td>
      <td>1.000000</td>
      <td>0.414838</td>
      <td>0.159651</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>-0.001652</td>
      <td>0.081629</td>
      <td>0.018443</td>
      <td>-0.189119</td>
      <td>0.414838</td>
      <td>1.000000</td>
      <td>0.216225</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.012658</td>
      <td>0.257307</td>
      <td>-0.549500</td>
      <td>0.096067</td>
      <td>0.159651</td>
      <td>0.216225</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



**A strong correlation between the Survival Rate and Pclass and Fare** can be seen.  
It seems that the higher the socio-economic status of the passenger the better the possibility of surviving the accident.  
  
Further investigation will take place.
  
There are also some other "secondary" correlations, not directly relevant with the answers we are looking for:  
* Negative correlation between the Age and the Passenger's Class (The younger passengers could not affort an expensive class" ticket)  
* Negative correlation between the Passenger's Class and the Fare (The "higher" the class, the more expensive the ticket)  
* Negative correlation between the Passenger's Age and the number of Siblings (The older the passenger the fewer siblings onboard)
* Positive correlation between the number of Spouses and the number of Siblings (Large families onboard constituted both from siblings and spouses)  

Also, it would be useful to calculate the Survival Rate for the whole sample as a baseline .


```python
titanic_df["Survived"].mean()
```




    0.3838383838383838



In the analysis, we will need to group several dataframes and rename some of their columns.  
The following function takes an *original_df* as an input (titanic_df by default), drops the lines with NaNs across the *column* and creates a new dataframe named *df_name* grouped across the *column* axis.
Also it renames some of the variables.


```python
def grouped(column,original_df=titanic_df):
    
    a = original_df.dropna(subset=[column]).groupby(column).agg({"PassengerId" : "count", "Survived" : "sum"})
    a = a.rename(columns = {"PassengerId":"Embarked", "Survived":"Survived"})
    b = original_df.dropna(subset=[column]).groupby(column).agg({"Survived" : "mean"})
    b = b.rename(columns = {"Survived" : "Survival Rate"})
    df_name = pd.concat([a,b], axis=1)
        
    return df_name
```

## Survival Rate per Passenger's Class

By calculating the average value of the *Survived* variable for each Class, we are calculating the Survival Rate of each Class.


```python
#Create a grouped by "Pclass" DataFrame with the average "Survived"
#No need to dropna() because there are not NaN on "Pclass" or "Survived" variables
pclass_df = grouped("Pclass")
pclass_df
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Survived</th>
      <th>Survival Rate</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>216</td>
      <td>136</td>
      <td>0.629630</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
      <td>87</td>
      <td>0.472826</td>
    </tr>
    <tr>
      <th>3</th>
      <td>491</td>
      <td>119</td>
      <td>0.242363</td>
    </tr>
  </tbody>
</table>
</div>



The passengers of the 1st and 2nd Class had a greater than the average Survival Rate.


```python
#putting the plotting code in a function so we can called again in the conclusions
def conclusion1():
    
    plt.subplots(figsize = (14, 5))
    
    #Plotting the passengers distribution per Class
    plt.subplot(121)
    
    N = len(pclass_df.index)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars
    
    bar1=plt.bar(ind, pclass_df["Embarked"], width, color="#5975A4", label="Embarked")
    bar2=plt.bar(ind + width, pclass_df["Survived"], width, color='#5F9E6E', label="Survived")

    plt.xlabel("Passenger Class", fontsize=12)
    plt.ylabel("Number of Passengers", fontsize=12)
    plt.title("Passengers' Distributions per Class", fontsize=14)
    plt.xticks(ind + width, pclass_df.index.values)

    plt.legend(loc=2)
        
    #Plotting the Survival Rate per Class
    plt.subplot(122)

    bar3 = sns.barplot(x="Pclass", y="Survival Rate", data=pclass_df.reset_index(),color='#5975A4')

    #Adding the average Survival Rate
    plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Survival Rate", fontsize=12)
    plt.title("Survival Rate per Passenger's Class", fontsize=14)
    
    
conclusion1()
plt.show()
```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_61_0.png)


It is obvious that the "higher" (smaller number) the Passenger's Class, the higher the Survival Rate.  

## Survival Rate per Passenger's Gender

*"Women and children first" is a code of conduct whereby the lives of women and children are to be saved first in a life-threatening situation, typically abandoning ship, when survival resources such as lifeboats were limited*. (Source: https://en.wikipedia.org/wiki/Women_and_children_first)  
  
Let's see if the women on Titanic had a higher Survival Rate than the men.


```python
#Create a grouped by "Sex" DataFrame with the average "Survived"
#No need to dropna() because there are not NaN on "Sex" or "Survived" variables
sex_df = grouped("Sex")
sex_df
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Survived</th>
      <th>Survival Rate</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>314</td>
      <td>233</td>
      <td>0.742038</td>
    </tr>
    <tr>
      <th>male</th>
      <td>577</td>
      <td>109</td>
      <td>0.188908</td>
    </tr>
  </tbody>
</table>
</div>




```python
#putting the plotting code in a function so we can called again in the conclusions
def conclusion2():
    plt.subplots(figsize = (14, 5))
    
    #Plotting the passengers distribution per Class
    plt.subplot(121)
    
    N = len(sex_df.index)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars
    
    bar1=plt.bar(ind, sex_df["Embarked"], width, color="#5975A4", label="Embarked")
    bar2=plt.bar(ind + width, sex_df["Survived"], width, color='#5F9E6E', label="Survived")

    plt.xlabel("Gender", fontsize=12)
    plt.ylabel("Number of Passengers", fontsize=12)
    plt.title("Passengers' per Gender", fontsize=14)
    plt.xticks(ind + width, sex_df.index.values)

    plt.legend(loc=2)
    
    
    #Plotting the Survival Rate per Class
    plt.subplot(122)

    bar3 = sns.barplot(x="Sex", y="Survival Rate", data=sex_df.reset_index(),color='#5975A4')

    #Adding the average Survival Rate
    plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

    plt.xlabel("Gender", fontsize=12)
    plt.ylabel("Survival Rate", fontsize=12)
    plt.title("Survival Rate per Gender", fontsize=14)

conclusion2()
plt.show()
```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_66_0.png)


The Women had over 4 times the Survival Rate of the men.  
So far, this is the most crucial factor of the Survival Rate.

## Survival Rate per Passenger's Age

Investigating the second group of the "Women and children first" code of conduct, we will analyze the Age as survival factor.  
  
Since the Age variable is comprised of almost indiscrete values, it will have very small practical value to group the DataFrame by Age. A better approach would be to group the passengers in "Decades" so that each passenger will be "moved" to the nearest decade.  
  
The subsets that will be created will be (0,5),[5,15),[15,25),[25,35),[35,45),[45,55),[55,65),[65,75),[75,85).


```python
#Drop the NaN values
age_df=titanic_df.dropna(subset = ["Age"])

#A function that round the age to the neares decade
def decade(age):
    return int((round(age/10)*10))

#Applying the "decade" function to the "Age" column
Decade = age_df[["Age"]].applymap(decade)
Decade.columns = ["Decade"]

#Concatenate the new column to the "age_df" DataSet
dec_df = pd.concat([age_df, Decade], axis = 1)
```


```python
#Create a grouped by "Decade" DataFrame with the average "Survived"
#No need to dropna() because we have already droped the null values during the creation of the DataFrame

decade_df = grouped("Decade",dec_df)
decade_df
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Survived</th>
      <th>Survival Rate</th>
    </tr>
    <tr>
      <th>Decade</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40</td>
      <td>27</td>
      <td>0.675000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>38</td>
      <td>18</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>20</th>
      <td>200</td>
      <td>73</td>
      <td>0.365000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>201</td>
      <td>78</td>
      <td>0.388060</td>
    </tr>
    <tr>
      <th>40</th>
      <td>120</td>
      <td>51</td>
      <td>0.425000</td>
    </tr>
    <tr>
      <th>50</th>
      <td>73</td>
      <td>30</td>
      <td>0.410959</td>
    </tr>
    <tr>
      <th>60</th>
      <td>31</td>
      <td>12</td>
      <td>0.387097</td>
    </tr>
    <tr>
      <th>70</th>
      <td>10</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>80</th>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Plotting the resulting DataFrame
fig = plt.subplots(figsize = (14, 10))

plt.subplot(221)

N = len(decade_df.index)
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars
    
bar1=plt.bar(ind, decade_df["Embarked"], width, color="#5975A4", label="Embarked")
bar2=plt.bar(ind + width, decade_df["Survived"], width, color='#5F9E6E', label="Survived")

plt.xlabel("Age", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)
plt.title("Passengers' per Gender", fontsize=14)
plt.xticks(ind + width, decade_df.index.values)

plt.legend(loc=2)

#Survival Rate per Age
plt.subplot(222)

p = sns.barplot(x="Decade", y="Survival Rate", color='#5975A4', data=decade_df.reset_index())

#Adding the average Survival Rate
plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

p.set_xlabel("Age", fontsize=12)
p.set_ylabel("Survival Rate", fontsize=12)
p.set_title("Survival Rate per Passenger's Age", fontsize=14)

#Linear Regression
plt.subplot(223)

age_df=titanic_df.dropna(subset = ["Age"])

#An order of "4" has been selected so that the regression model will follow the histogram's trend
sns.regplot(x="Age", y="Survived", data=dec_df, order=4, y_jitter=0.01, scatter_kws={"s": 80});

#Adding the average Survival Rate
plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

plt.xlabel("Age", fontsize=12)
plt.ylabel("Survival Rate", fontsize=12)
plt.title("Survival Rate per Passenger's Age (regression)", fontsize=14)

plt.tight_layout()
plt.show()
```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_72_0.png)


In the above graphs, we can notice some extreme values to the right end of the scale (70: 0%, 80: 100%).
Let"s dig a little bit further by having a closer look at the passengers of these two subsets.


```python
#Return all the rows with "Decade" 70 or more.
dec_df.loc[dec_df["Decade"] >= 70]
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Port</th>
      <th>Decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>0</td>
      <td>2</td>
      <td>Wheadon, Mr. Edward H</td>
      <td>male</td>
      <td>66.0</td>
      <td>0</td>
      <td>0</td>
      <td>C.A. 24579</td>
      <td>10.5000</td>
      <td>NaN</td>
      <td>S</td>
      <td>70</td>
    </tr>
    <tr>
      <th>54</th>
      <td>55</td>
      <td>0</td>
      <td>1</td>
      <td>Ostby, Mr. Engelhart Cornelius</td>
      <td>male</td>
      <td>65.0</td>
      <td>0</td>
      <td>1</td>
      <td>113509</td>
      <td>61.9792</td>
      <td>B30</td>
      <td>C</td>
      <td>70</td>
    </tr>
    <tr>
      <th>96</th>
      <td>97</td>
      <td>0</td>
      <td>1</td>
      <td>Goldschmidt, Mr. George B</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17754</td>
      <td>34.6542</td>
      <td>A5</td>
      <td>C</td>
      <td>70</td>
    </tr>
    <tr>
      <th>116</th>
      <td>117</td>
      <td>0</td>
      <td>3</td>
      <td>Connors, Mr. Patrick</td>
      <td>male</td>
      <td>70.5</td>
      <td>0</td>
      <td>0</td>
      <td>370369</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
      <td>70</td>
    </tr>
    <tr>
      <th>280</th>
      <td>281</td>
      <td>0</td>
      <td>3</td>
      <td>Duane, Mr. Frank</td>
      <td>male</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>336439</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
      <td>70</td>
    </tr>
    <tr>
      <th>456</th>
      <td>457</td>
      <td>0</td>
      <td>1</td>
      <td>Millet, Mr. Francis Davis</td>
      <td>male</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>13509</td>
      <td>26.5500</td>
      <td>E38</td>
      <td>S</td>
      <td>70</td>
    </tr>
    <tr>
      <th>493</th>
      <td>494</td>
      <td>0</td>
      <td>1</td>
      <td>Artagaveytia, Mr. Ramon</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17609</td>
      <td>49.5042</td>
      <td>NaN</td>
      <td>C</td>
      <td>70</td>
    </tr>
    <tr>
      <th>630</th>
      <td>631</td>
      <td>1</td>
      <td>1</td>
      <td>Barkworth, Mr. Algernon Henry Wilson</td>
      <td>male</td>
      <td>80.0</td>
      <td>0</td>
      <td>0</td>
      <td>27042</td>
      <td>30.0000</td>
      <td>A23</td>
      <td>S</td>
      <td>80</td>
    </tr>
    <tr>
      <th>672</th>
      <td>673</td>
      <td>0</td>
      <td>2</td>
      <td>Mitchell, Mr. Henry Michael</td>
      <td>male</td>
      <td>70.0</td>
      <td>0</td>
      <td>0</td>
      <td>C.A. 24580</td>
      <td>10.5000</td>
      <td>NaN</td>
      <td>S</td>
      <td>70</td>
    </tr>
    <tr>
      <th>745</th>
      <td>746</td>
      <td>0</td>
      <td>1</td>
      <td>Crosby, Capt. Edward Gifford</td>
      <td>male</td>
      <td>70.0</td>
      <td>1</td>
      <td>1</td>
      <td>WE/P 5735</td>
      <td>71.0000</td>
      <td>B22</td>
      <td>S</td>
      <td>70</td>
    </tr>
    <tr>
      <th>851</th>
      <td>852</td>
      <td>0</td>
      <td>3</td>
      <td>Svensson, Mr. Johan</td>
      <td>male</td>
      <td>74.0</td>
      <td>0</td>
      <td>0</td>
      <td>347060</td>
      <td>7.7750</td>
      <td>NaN</td>
      <td>S</td>
      <td>70</td>
    </tr>
  </tbody>
</table>
</div>



There were 10 passengers in the Decade 70 that none survived and only one in the Decade 80, who survived.  
The last one can be considered as an outlier and removed from the sample.


```python
dec_df = dec_df.drop(630)
decade_df = grouped("Decade",dec_df)
decade_df
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Survived</th>
      <th>Survival Rate</th>
    </tr>
    <tr>
      <th>Decade</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40</td>
      <td>27</td>
      <td>0.675000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>38</td>
      <td>18</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>20</th>
      <td>200</td>
      <td>73</td>
      <td>0.365000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>201</td>
      <td>78</td>
      <td>0.388060</td>
    </tr>
    <tr>
      <th>40</th>
      <td>120</td>
      <td>51</td>
      <td>0.425000</td>
    </tr>
    <tr>
      <th>50</th>
      <td>73</td>
      <td>30</td>
      <td>0.410959</td>
    </tr>
    <tr>
      <th>60</th>
      <td>31</td>
      <td>12</td>
      <td>0.387097</td>
    </tr>
    <tr>
      <th>70</th>
      <td>10</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#putting the plotting code in a function so we can called again in the conclusions
def conclusion3():

    fig = plt.subplots(figsize = (14, 10))

    plt.subplot(221)

    N = len(decade_df.index)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    bar1=plt.bar(ind, decade_df["Embarked"], width, color="#5975A4", label="Embarked")
    bar2=plt.bar(ind + width, decade_df["Survived"], width, color='#5F9E6E', label="Survived")

    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Number of Passengers", fontsize=12)
    plt.title("Survival Rate per Passenger's Age", fontsize=14)
    plt.xticks(ind + width, decade_df.index.values)

    plt.legend(loc=2)

    #Survival Rate per Age
    plt.subplot(222)

    p = sns.barplot(x="Decade", y="Survival Rate", color='#5975A4', data=decade_df.reset_index())

    #Adding the average Survival Rate
    plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

    p.set_xlabel("Age", fontsize=12)
    p.set_ylabel("Survival Rate", fontsize=12)
    p.set_title("Survival Rate per Passenger's Age", fontsize=14)

    #Linear Regression
    plt.subplot(223)

    age_df=titanic_df.dropna(subset = ["Age"])

    #An order of "3" has been selected so that the regression model will follow the histogram's trend
    sns.regplot(x="Age", y="Survived", data=dec_df, order=3, y_jitter=0.01, scatter_kws={"s": 80});

    #Adding the average Survival Rate
    plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

    plt.xlabel("AgeAge", fontsize=12)
    plt.ylabel("Survival Rate", fontsize=12)
    plt.title("Survival Rate per Passenger's Age", fontsize=14)

    plt.tight_layout()

conclusion3()
plt.show()
```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_77_0.png)


From the above results, we can conclude that the Age was a crucial factor for the survival of the passengers with the children under 15 having the greatest probability to survive.  

The rest of the variables could not affect (at least obviously) the Survival Rate but let's continue the analysis in case there are connections our intuition cannot spot.

## Survival Rate per Number of Siblings/Spouses


```python
#Create a grouped by "SibSp" DataFrame with the average "Survived"
#No need to dropna() because there are not NaN on "SibSp" or "Survived" variables
SibSp_df = grouped("SibSp")

#Plotting the resulting DataFrame
fig = plt.subplots(figsize = (14, 10))

plt.subplot(221)

N = len(SibSp_df.index)
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

bar1=plt.bar(ind, SibSp_df["Embarked"], width, color="#5975A4", label="Embarked")
bar2=plt.bar(ind + width, SibSp_df["Survived"], width, color='#5F9E6E', label="Survived")

plt.xlabel("Number of Siblings/Spouses", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)
plt.title("Passengers' per Number of Siblings/Spouses", fontsize=14)
plt.xticks(ind + width, SibSp_df.index.values)

plt.legend(loc=2)

plt.subplot(222)

p = sns.barplot(x="SibSp", y="Survival Rate", ci=None, color='#5975A4', data=SibSp_df.reset_index())

#Adding the average Survival Rate
plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

p.set_xlabel("Number of Siblings/Spouses", fontsize=12)
p.set_ylabel("Survival Rate", fontsize=12)
p.set_title("Survival Rate per Number of Siblings/Spouses", fontsize=14)

plt.subplot(223)

sns.regplot(x="SibSp", y="Survived", data=titanic_df, order=2, scatter_kws={"s": 80});

#Adding the average Survival Rate
plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

plt.xlabel("Number of Siblings/Spouses", fontsize=12)
plt.ylabel("Survival Rate", fontsize=12)
plt.title("Survival Rate per Number of Siblings/Spouses", fontsize=14)

plt.tight_layout()
plt.show()
```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_80_0.png)


There is a patern in the plot but as we saw earlier, there is a negative correlation between *Number of Siblings/Spouses* and *Age*.  
Since, the *Number of Siblings/Spouses* doesn't make much sense to affect the *Survival Rate* we can assume that *Age* is a Common Cause for both *Survival Rate* and *Number of Siblings/Spouses*.


```python
fig = plt.subplots(figsize = (14, 10))

plt.figure(1)

plt.subplot(221)

p = sns.barplot(x="Decade", y="Survived", ci=None, color='#5975A4', data=dec_df.reset_index())

#Adding the average Survival Rate
plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

p.set_xlabel("Age", fontsize=12)
p.set_ylabel("Survival Rate", fontsize=12)
p.set_title("Survival Rate per Passenger's Age", fontsize=14)

plt.subplot(222)

p = sns.barplot(x="SibSp", y="Survival Rate", ci=None, color='#5975A4', data=SibSp_df.reset_index())

#Adding the average Survival Rate
plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

p.set_xlabel("Number of Siblings/Spouses", fontsize=12)
p.set_ylabel("Survival Rate", fontsize=12)
p.set_title("Survival Rate per Number of Siblings/Spouses", fontsize=14)

plt.subplot(223)
sns.regplot(x="Age", y="Survived", data=dec_df, order=3, y_jitter=0.01, scatter_kws={"s": 80});

#Adding the average Survival Rate
plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

plt.xlabel("Age", fontsize=12)
plt.ylabel("Survival Rate", fontsize=12)
plt.title("Survival Rate per Passenger's Age", fontsize=14)

plt.subplot(224)

sns.regplot(x="SibSp", y="Survived", data=titanic_df, order=2, y_jitter=0.01, scatter_kws={"s": 80});

#Adding the average Survival Rate
plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

plt.xlabel("Number of Siblings/Spouses", fontsize=12)
plt.ylabel("Survival Rate", fontsize=12)
plt.title("Survival Rate per Number of Siblings/Spouses", fontsize=14)

plt.tight_layout()
plt.show()
```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_82_0.png)


This negative corelation between the *Passenger's Age* and the *Number of Siblings/Spouses* can be further highlighted in the following plot.


```python
a = pd.DataFrame(data = pd.DataFrame(titanic_df.dropna(subset=['Age'])).dropna(subset=['SibSp']))
sns.lmplot(x="SibSp", y="Age", data=a)

plt.xlabel("Number of Siblings/Spouses", fontsize=12)
plt.ylabel("Age", fontsize=12)
plt.title("Correlation between Number of Siblings/Spouses & Age", fontsize=14)

#plt.tight_layout()
plt.show()
```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_84_0.png)


## Survival Rate per Number of Parents/Children Aboard


```python
#Create a grouped by "Parch" DataFrame with the average "Survived"
#No need to dropna() because there are not NaN on "Parch" or "Survived" variables
parch_df = grouped("Parch")
```


```python
plt.subplots(figsize = (14, 5))
    
#Plotting the passengers distribution per Class
plt.subplot(121)

N = len(parch_df.index)
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

bar1=plt.bar(ind, parch_df["Embarked"], width, color="#5975A4", label="Embarked")
bar2=plt.bar(ind + width, parch_df["Survived"], width, color='#5F9E6E', label="Survived")

plt.xlabel("Number of Parents/Children", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)
plt.title("Passengers' Distributions per Number of Parents/Children", fontsize=14)
plt.xticks(ind + width, parch_df.index.values)

plt.legend(loc=1)

#Plotting the resulting DataFrame
plt.subplot(122)

p = sns.barplot(x="Parch", y="Survival Rate", ci=None, color='#5975A4', data=parch_df.reset_index())

#Adding the average Survival Rate
plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

p.set_xlabel("Number of Parents/Children", fontsize=12)
p.set_ylabel("Survival Rate", fontsize=12)
p.set_title("Survival Rate per Number of Parents/Children", fontsize=14)

plt.show()

```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_87_0.png)


The above diagram cannot give a clear picture of a correlation between the *Number of Parents/Children* and the Survival rate. We can say though that the passengers that were traveling with 1 to 3 Parents/Children had a greater Survival Ratio.

## Survival Rate per Fare

We know that the "Higher" the Passenger's Class the higher the fare, so we are expecting a possitive correlation between the Fare and the Survival Rate since we have already concluded that the Passenger's Class was a Critical Factor.


```python
fig = plt.subplots(figsize = (14, 5))

plt.figure(1)

plt.subplot(121)

sns.regplot(x="Fare", y="Survived", data=titanic_df, order=1, scatter_kws={"s": 80});

#Adding the average Survival Rate
plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

plt.xlabel("Fare", fontsize=12)
plt.ylabel("Survival Rate", fontsize=12)
plt.title("Survival Rate per Fare", fontsize=14)

plt.subplot(122)

sns.distplot(titanic_df["Fare"], kde=False)

plt.xlabel("Fare", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)
plt.title("Number of Passengers per Fare", fontsize=14)

plt.show()
```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_91_0.png)


And if we remove the 300+ fare outliers:


```python
fig = plt.subplots(figsize = (14, 5))

d = titanic_df[titanic_df['Fare'] < 300]

plt.figure(1)

plt.subplot(121)

sns.regplot(x="Fare", y="Survived", data=d, order=1, truncate=True, scatter_kws={"s": 80});

#Adding the average Survival Rate
plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

plt.xlabel("Fare", fontsize=12)
plt.ylabel("Survival Rate", fontsize=12)
plt.title("Survival Rate per Fare", fontsize=14)

plt.subplot(122)

sns.distplot(d["Fare"], kde=False)

plt.xlabel("Fare", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)
plt.title("Number of Passengers per Fare", fontsize=14)

plt.show()
```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_93_0.png)


As we expected, the *Fare* was a critical Survival Factor.

## Survival Rate per Port of Embarkation

Finally, let's visualize the Survival Rate per Port of Embarkation to find out if the passengers from the three ports had the same Survival Rates.


```python
embarked_df = grouped("Port")
embarked_df = embarked_df.set_index([['Cherbourg' , 'Queenstown', 'Southampton']])
```


```python
plt.subplots(figsize = (14, 5))
    
#Plotting the passengers distribution per Class
plt.subplot(121)

N = len(embarked_df.index)
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

bar1=plt.bar(ind, embarked_df["Embarked"], width, color="#5975A4", label="Embarked")
bar2=plt.bar(ind + width, embarked_df["Survived"], width, color='#5F9E6E', label="Survived")

plt.xlabel("Port of Embarkation", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)
plt.title("Passengers' Distributions per Port of Embarkation", fontsize=14)
plt.xticks(ind + width, embarked_df.index.values)

plt.legend(loc=2)

plt.subplot(122)

p = sns.barplot(x="Embarked", y="Survival Rate", ci=None, color='#5975A4', data=embarked_df.reset_index())

#Adding the average Survival Rate
plt.axhline(y=0.3838383838383838, ls='dashed', color='#0B559F', alpha=0.6)

p.set_xlabel("Port of Embarkation", fontsize=12)
p.set_ylabel("Survival Rate", fontsize=12)
p.set_title("Survival Rate per Port of Embarkation", fontsize=14)
p.set_xticklabels(embarked_df.index.values)

plt.show()
```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_98_0.png)


There are significant variations between the three ports.  
Let's explore the allocation of Gender and Passenger's Class on each port. 


```python
fig = plt.subplots(figsize = (14, 5))

d = titanic_df[titanic_df['Fare'] < 300]

plt.figure(1)

plt.subplot(121)

a = titanic_df.groupby(['Port', 'Sex'])
b = a['PassengerId'].count().reset_index()

p = sns.barplot(x='Port', y='PassengerId', hue='Sex', data=b)

p.set_xlabel("Port of Embarkation", fontsize=12)
p.set_ylabel("Number of Passengers", fontsize=12)
p.set_title("Number of Passengers per Port", fontsize=14)
p.set_xticklabels(['Cherbourg' , 'Queenstown', 'Southampton'])
p.legend(title="Gender", loc=2)

plt.subplot(122)

c = titanic_df.groupby(['Port', 'Pclass'])
d = c['PassengerId'].count().reset_index()

p = sns.barplot(x='Port', y='PassengerId', hue='Pclass', data=d,)

p.set_xlabel("Port of Embarkation", fontsize=12)
p.set_ylabel("Number of Passengers", fontsize=12)
p.set_title("Number of Passengers per Port of Embarcation", fontsize=14)
p.set_xticklabels(['Cherbourg' , 'Queenstown', 'Southampton'])
p.legend(title="Passenger's Class", loc=2)

plt.show()
```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_100_0.png)


The port with the higher Survival Rate is the one with the most higher ratio of "prestigious" passengers and a good female/male ratio and the one with the lowest Rate the "worst" ratio in both categories. This explain the significant differences between the three ports.

# Conclusions

Following the above analysis we can conclude that the most critical factors for the survival of the passengers were:  
* Gender
* Age
* Socio-economic Status

More specifically, women had over 4 times the Survival Rate of men (74.2% against 18.9%)...


```python
sex_df
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Survived</th>
      <th>Survival Rate</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>314</td>
      <td>233</td>
      <td>0.742038</td>
    </tr>
    <tr>
      <th>male</th>
      <td>577</td>
      <td>109</td>
      <td>0.188908</td>
    </tr>
  </tbody>
</table>
</div>




```python
conclusion2()
plt.show()
```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_106_0.png)


...the "Upper Class" nearly 3 times more chances than the "Lower Class" (63.0% against 24.2%)...


```python
pclass_df
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Survived</th>
      <th>Survival Rate</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>216</td>
      <td>136</td>
      <td>0.629630</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
      <td>87</td>
      <td>0.472826</td>
    </tr>
    <tr>
      <th>3</th>
      <td>491</td>
      <td>119</td>
      <td>0.242363</td>
    </tr>
  </tbody>
</table>
</div>




```python
conclusion1()
plt.show()
```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_109_0.png)


...and coming to the age factor, the most privileged were the infants (ages under 5) with a Survival Rate of 67.5%, almost double the average.


```python
decade_df
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Survived</th>
      <th>Survival Rate</th>
    </tr>
    <tr>
      <th>Decade</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40</td>
      <td>27</td>
      <td>0.675000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>38</td>
      <td>18</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>20</th>
      <td>200</td>
      <td>73</td>
      <td>0.365000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>201</td>
      <td>78</td>
      <td>0.388060</td>
    </tr>
    <tr>
      <th>40</th>
      <td>120</td>
      <td>51</td>
      <td>0.425000</td>
    </tr>
    <tr>
      <th>50</th>
      <td>73</td>
      <td>30</td>
      <td>0.410959</td>
    </tr>
    <tr>
      <th>60</th>
      <td>31</td>
      <td>12</td>
      <td>0.387097</td>
    </tr>
    <tr>
      <th>70</th>
      <td>10</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
conclusion3()
plt.show()
```


![png](/assets/2016-10-01-Investigating-a-Dataset/output_112_0.png)


The above conclusions are tentative and further statistical analysis is required in order to prove their validity.

# References
Udacity - https://www.udacity.com/  
Encyclopedia Titanica - https://www.encyclopedia-titanica.org  
Wikipedia - https://en.wikipedia.org

**Dataset**: https://d17h27t6h515a5.cloudfront.net/topher/2016/September/57e9a84c_titanic-data/titanic-data.csv
