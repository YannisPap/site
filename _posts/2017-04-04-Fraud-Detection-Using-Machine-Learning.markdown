---
layout: single
header:
  overlay_image: /assets/Fraud-Detection-Using-Machine-Learning/banner.jpg
  teaser: http://images4.static-bluray.com/reviews/759_5.jpg
  overlay_filter: 0.5
  caption: "Photo credit: [**Impawards**](http://www.impawards.com/)"
title:  "Fraud Detection Using Machine Learning (Analysis)"
excerpt: "Identified which employees are more likely to have committed fraud by applying machine learning to financial and email data."
date:   2017-04-04 15:26:52 +0300
tags:
- Python
- Scikit-learn
- Machine learning
- Natural language processing
- Feature selection
- Verifying machine learning performance
---

{% include toc %}

# Introduction

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.
These data have been combined with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.  

In this project, I am building **a person of interest identifier based on financial and email data**, made public as a result of the Enron scandal.  

The data have been combined in the form of a dictionary, where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person.

# The Dataset
## Features

The features included in the dataset can be divided in three categories, Salary Features, Stock Features and Email Features. Bellow you may find the full feature list with  brief definition of each one.

### Salary Features

| Payments            | Definitions of Category Groupings|
|:--------------------|:---------------------------------|
| **Salary**              | Reflects items such as base salary, executive cash allowances, and benefits payments.|
| **Bonus**               | Reflects annual cash incentives paid based upon company performance. Also may include other retention payments.|
| **Long Term Incentive** | Reflects long-term incentive cash payments from various long-term incentive programs designed to tie executive compensation to long-term success as measuredagainst key performance drivers and business objectives over a multi-year period, generally 3 to 5 years.|
| **Deferred Income**     | Reflects voluntary executive deferrals of salary, annual cash incentives, and long-term cash incentives as well as cash fees deferred by non-employee directorsunder a deferred compensation arrangement. May also reflect deferrals under a stock option or phantom stock unit in lieu of cash arrangement.|
|**Deferral Payments**   | Reflects distributions from a deferred compensation arrangement due to termination of employment or due to in-service withdrawals as per plan provisions.|
| **Loan Advances**       | Reflects total amount of loan advances, excluding repayments, provided by the Debtor in return for a promise of repayment. In certain instances, the terms of thepromissory notes allow for the option to repay with stock of the company.|
| **Other**               | Reflects items such as payments for severence, consulting services, relocation costs, tax advances and allowances for employees on international assignment (i.e.housing allowances, cost of living allowances, payments under Enron’s Tax Equalization Program, etc.). May also include payments provided with respect toemployment agreements, as well as imputed income amounts for such things as use of corporate aircraft. |
| **Expenses**            | Reflects reimbursements of business expenses. May include fees paid for consulting services.|
| **Director Fees**       | Reflects cash payments and/or value of stock grants made in lieu of cash payments to non-employee directors.|
| **Total Payments**      | Sum of the above values|  

***
  
  
### Stock Features

| Stock Value              | Definitions of Category Groupings|
|:-------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Exercised Stock Options**  | Reflects amounts from exercised stock options which equal the market value in excess of the exercise price on the date the options were exercised either throughcashless (same-day sale), stock swap or cash exercises. The reflected gain may differ from that realized by the insider due to fluctuations in the market price andthe timing of any subsequent sale of the securities. |
| **Restricted Stock**         | Reflects the gross fair market value of shares and accrued dividends (and/or phantom units and dividend equivalents) on the date of release due to lapse of vestingperiods, regardless of whether deferred.|
| **Restricted StockDeferred** | Reflects value of restricted stock voluntarily deferred prior to release under a deferred compensation arrangement.|
| **Total Stock Value**        | Sum of the above values |

***

### email Features

| Variable                      | Definition                                                                    |
|:------------------------------|:------------------------------------------------------------------------------|
| ***to messages***             | Total number of emails received (person's inbox)                              |
| ***email address***           | Email address of the person                                                   |
| ***from poi to this person*** | Number of emails received by POIs                                             |
| ***from messages***           | Total number of emails sent by this person                                    |
| ***from this person to poi*** | Number of emails sent by this person to a POI.                                |
| ***shared receipt with poi*** | Number of emails addressed by someone else to a POI where this person was CC. |  

***

## Data Exploration


```python
#Importing libraries and magics
%matplotlib inline

import sys
sys.path.append("./code/")

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import re

from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC

from tester import test_classifier
import warnings

# Load the dictionary containing the dataset
with open("./dataset/final_project_dataset.pkl", "rb") as data_file:
    data_init = pickle.load(data_file)

#Converting the dataset from a python dictionary to a pandas dataframe
data_df = pd.DataFrame.from_dict(data_init, orient='index')
data_df.shape
```
    (146, 21)
```

```python
data_df.head()
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>email_address</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>...</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALLEN PHILLIP K</th>
      <td>201955</td>
      <td>2902</td>
      <td>2869717</td>
      <td>4484442</td>
      <td>NaN</td>
      <td>4175000</td>
      <td>phillip.allen@enron.com</td>
      <td>-126027</td>
      <td>-3081055</td>
      <td>1729541</td>
      <td>...</td>
      <td>47</td>
      <td>1729541</td>
      <td>2195</td>
      <td>152</td>
      <td>65</td>
      <td>False</td>
      <td>304805</td>
      <td>1407</td>
      <td>126027</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BADUM JAMES P</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>178980</td>
      <td>182466</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>257817</td>
      <td>...</td>
      <td>NaN</td>
      <td>257817</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BANNANTINE JAMES M</th>
      <td>477</td>
      <td>566</td>
      <td>NaN</td>
      <td>916197</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>james.bannantine@enron.com</td>
      <td>-560222</td>
      <td>-5104</td>
      <td>5243487</td>
      <td>...</td>
      <td>39</td>
      <td>4046157</td>
      <td>29</td>
      <td>864523</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>465</td>
      <td>1757552</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAXTER JOHN C</th>
      <td>267102</td>
      <td>NaN</td>
      <td>1295738</td>
      <td>5634343</td>
      <td>NaN</td>
      <td>1200000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1386055</td>
      <td>10623258</td>
      <td>...</td>
      <td>NaN</td>
      <td>6680544</td>
      <td>NaN</td>
      <td>2660303</td>
      <td>NaN</td>
      <td>False</td>
      <td>1586055</td>
      <td>NaN</td>
      <td>3942714</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAY FRANKLIN R</th>
      <td>239671</td>
      <td>NaN</td>
      <td>260455</td>
      <td>827696</td>
      <td>NaN</td>
      <td>400000</td>
      <td>frank.bay@enron.com</td>
      <td>-82782</td>
      <td>-201641</td>
      <td>63014</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>145796</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



The dataset contains information of 21 features from 146 employees.  

"NaN"s are actually strings so I will replace them with Numpy's "NaN"s so I can count the information I have across the variables.


```python
data_df.replace(to_replace='NaN', value=np.nan, inplace=True)
```


```python
data_df.count().sort_values()
```




    loan_advances                  4
    director_fees                 17
    restricted_stock_deferred     18
    deferral_payments             39
    deferred_income               49
    long_term_incentive           66
    bonus                         82
    to_messages                   86
    shared_receipt_with_poi       86
    from_this_person_to_poi       86
    from_poi_to_this_person       86
    from_messages                 86
    other                         93
    salary                        95
    expenses                      95
    exercised_stock_options      102
    restricted_stock             110
    email_address                111
    total_payments               125
    total_stock_value            126
    poi                          146
    dtype: int64



We can see that the dataset is quite sparse with some variables like *Total Payments* and *Total Stock Value* having values for most of the employees but some others like *Loan Advances* and *Director Fees* that we have information for too few employees.  
I am wondering if there are any records for all missing values. For this I will remove the *email_address* field since we cannot use it somehow in the analysis and I will create a temporary copy without the label (POI).

By paying attention to the [Payments Schedule]({{ site.url }}/assets/enron61702insiderpay.pdf) which is the source of the dataset, we can see that the empty values, except the *email_address*, are actually "0". After cleaning the dataset I will impute the empty values with "0". For now, I will remove the *email_address* field since we cannot use it somehow in the analysis.


```python
#dropping 'poi' and 'email_address' variables
data_df = data_df.drop(["email_address"], axis=1)
data_temp = data_df.drop(["poi"], axis=1)
data_temp[data_temp.isnull().all(axis=1)]
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>expenses</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LOCKHART EUGENE E</th>
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
</div>



"LOCKHART EUGENE E" has all the values missing and will be removed


```python
data_df = data_df.drop(["LOCKHART EUGENE E"], axis=0)
```

Next, since some values are related I would like to rearrange the columns in he following order:  

| **POI** | *All payment features* | **Total Payments** | *All stock features* | **Total Stocks** | *Incoming emails features* | **All incoming mails** | *sent emails features* | **All sent emails** |


```python
cols = [
    'poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income',
    'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees',
    'total_payments', 'exercised_stock_options', 'restricted_stock',
    'restricted_stock_deferred', 'total_stock_value',
    'from_poi_to_this_person', 'shared_receipt_with_poi', 'to_messages',
    'from_this_person_to_poi', 'from_messages'
]
data_df = data_df[cols]
data_df.head()
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>poi</th>
      <th>salary</th>
      <th>bonus</th>
      <th>long_term_incentive</th>
      <th>deferred_income</th>
      <th>deferral_payments</th>
      <th>loan_advances</th>
      <th>other</th>
      <th>expenses</th>
      <th>director_fees</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
      <th>from_poi_to_this_person</th>
      <th>shared_receipt_with_poi</th>
      <th>to_messages</th>
      <th>from_this_person_to_poi</th>
      <th>from_messages</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALLEN PHILLIP K</th>
      <td>False</td>
      <td>201955.0</td>
      <td>4175000.0</td>
      <td>304805.0</td>
      <td>-3081055.0</td>
      <td>2869717.0</td>
      <td>NaN</td>
      <td>152.0</td>
      <td>13868.0</td>
      <td>NaN</td>
      <td>4484442.0</td>
      <td>1729541.0</td>
      <td>126027.0</td>
      <td>-126027.0</td>
      <td>1729541.0</td>
      <td>47.0</td>
      <td>1407.0</td>
      <td>2902.0</td>
      <td>65.0</td>
      <td>2195.0</td>
    </tr>
    <tr>
      <th>BADUM JAMES P</th>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>178980.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3486.0</td>
      <td>NaN</td>
      <td>182466.0</td>
      <td>257817.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>257817.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BANNANTINE JAMES M</th>
      <td>False</td>
      <td>477.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-5104.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>864523.0</td>
      <td>56301.0</td>
      <td>NaN</td>
      <td>916197.0</td>
      <td>4046157.0</td>
      <td>1757552.0</td>
      <td>-560222.0</td>
      <td>5243487.0</td>
      <td>39.0</td>
      <td>465.0</td>
      <td>566.0</td>
      <td>0.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>BAXTER JOHN C</th>
      <td>False</td>
      <td>267102.0</td>
      <td>1200000.0</td>
      <td>1586055.0</td>
      <td>-1386055.0</td>
      <td>1295738.0</td>
      <td>NaN</td>
      <td>2660303.0</td>
      <td>11200.0</td>
      <td>NaN</td>
      <td>5634343.0</td>
      <td>6680544.0</td>
      <td>3942714.0</td>
      <td>NaN</td>
      <td>10623258.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAY FRANKLIN R</th>
      <td>False</td>
      <td>239671.0</td>
      <td>400000.0</td>
      <td>NaN</td>
      <td>-201641.0</td>
      <td>260455.0</td>
      <td>NaN</td>
      <td>69.0</td>
      <td>129142.0</td>
      <td>NaN</td>
      <td>827696.0</td>
      <td>NaN</td>
      <td>145796.0</td>
      <td>-82782.0</td>
      <td>63014.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_df.replace(to_replace="NaN", value=0, inplace=True)

```

Now that the features are in the right order, we can examine the statistics of the dataset.


```python
data_df.describe()
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>bonus</th>
      <th>long_term_incentive</th>
      <th>deferred_income</th>
      <th>deferral_payments</th>
      <th>loan_advances</th>
      <th>other</th>
      <th>expenses</th>
      <th>director_fees</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
      <th>from_poi_to_this_person</th>
      <th>shared_receipt_with_poi</th>
      <th>to_messages</th>
      <th>from_this_person_to_poi</th>
      <th>from_messages</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.450000e+02</td>
      <td>1.450000e+02</td>
      <td>1.450000e+02</td>
      <td>1.450000e+02</td>
      <td>1.450000e+02</td>
      <td>1.450000e+02</td>
      <td>1.450000e+02</td>
      <td>1.450000e+02</td>
      <td>1.450000e+02</td>
      <td>1.450000e+02</td>
      <td>1.450000e+02</td>
      <td>1.450000e+02</td>
      <td>1.450000e+02</td>
      <td>1.450000e+02</td>
      <td>145.000000</td>
      <td>145.000000</td>
      <td>145.000000</td>
      <td>145.000000</td>
      <td>145.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.683342e+05</td>
      <td>1.342671e+06</td>
      <td>6.692680e+05</td>
      <td>-3.854019e+05</td>
      <td>4.418227e+05</td>
      <td>1.157586e+06</td>
      <td>5.894693e+05</td>
      <td>7.123619e+04</td>
      <td>1.955643e+04</td>
      <td>4.380626e+06</td>
      <td>4.211583e+06</td>
      <td>1.761321e+06</td>
      <td>2.065786e+04</td>
      <td>5.886335e+06</td>
      <td>38.489655</td>
      <td>697.765517</td>
      <td>1230.013793</td>
      <td>24.455172</td>
      <td>361.075862</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.211001e+06</td>
      <td>8.121319e+06</td>
      <td>4.059716e+06</td>
      <td>2.386279e+06</td>
      <td>2.750583e+06</td>
      <td>9.682311e+06</td>
      <td>3.694784e+06</td>
      <td>4.341759e+05</td>
      <td>1.194559e+05</td>
      <td>2.702539e+07</td>
      <td>2.615843e+07</td>
      <td>1.093676e+07</td>
      <td>1.444650e+06</td>
      <td>3.636916e+07</td>
      <td>74.088359</td>
      <td>1075.128126</td>
      <td>2232.153003</td>
      <td>79.527073</td>
      <td>1445.944684</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-2.799289e+07</td>
      <td>-1.025000e+05</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-2.604490e+06</td>
      <td>-7.576788e+06</td>
      <td>-4.409300e+04</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-3.834600e+04</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.025000e+05</td>
      <td>0.000000e+00</td>
      <td>3.246000e+04</td>
      <td>0.000000e+00</td>
      <td>2.520550e+05</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.106920e+05</td>
      <td>3.000000e+05</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>9.720000e+02</td>
      <td>2.153000e+04</td>
      <td>0.000000e+00</td>
      <td>9.665220e+05</td>
      <td>6.087500e+05</td>
      <td>3.605280e+05</td>
      <td>0.000000e+00</td>
      <td>9.760370e+05</td>
      <td>4.000000</td>
      <td>114.000000</td>
      <td>312.000000</td>
      <td>0.000000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.714420e+05</td>
      <td>8.000000e+05</td>
      <td>3.753040e+05</td>
      <td>0.000000e+00</td>
      <td>1.025900e+04</td>
      <td>0.000000e+00</td>
      <td>1.506560e+05</td>
      <td>5.394700e+04</td>
      <td>0.000000e+00</td>
      <td>1.979596e+06</td>
      <td>1.729541e+06</td>
      <td>8.530640e+05</td>
      <td>0.000000e+00</td>
      <td>2.332399e+06</td>
      <td>41.000000</td>
      <td>900.000000</td>
      <td>1607.000000</td>
      <td>14.000000</td>
      <td>52.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.670423e+07</td>
      <td>9.734362e+07</td>
      <td>4.852193e+07</td>
      <td>0.000000e+00</td>
      <td>3.208340e+07</td>
      <td>8.392500e+07</td>
      <td>4.266759e+07</td>
      <td>5.235198e+06</td>
      <td>1.398517e+06</td>
      <td>3.098866e+08</td>
      <td>3.117640e+08</td>
      <td>1.303223e+08</td>
      <td>1.545629e+07</td>
      <td>4.345095e+08</td>
      <td>528.000000</td>
      <td>5521.000000</td>
      <td>15149.000000</td>
      <td>609.000000</td>
      <td>14368.000000</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the 3 categories have different orders of magnitude with the biggest different between the *email features* and the other two categories. Most probably we will need to scale the dataset.

## Outlier Investigation

My first attempt to spot any possible outliers will be visual.  
I will use Seaborn's pairplot which present in the same time the distribution of the variables and a scatter plot representation of them. Since the number of variables are too many to plot them all, I will use the 4 with the higher variance.


```python
sns.pairplot(data=data_df, vars=["total_payments", "exercised_stock_options", "restricted_stock", "total_stock_value"], hue="poi")
```




    <seaborn.axisgrid.PairGrid at 0x7f8d995f6048>




![png]({{ site.url }}/assets/2017-04-04-Fraud-Detection-Using-Machine-Learning/output_30_1.png)


There are two datapoints far away from the cluster of the rest. I will use the *Total Payments* to find them.


```python
data_df.total_payments.nlargest(2)
```




    TOTAL            309886585.0
    LAY KENNETH L    103559793.0
    Name: total_payments, dtype: float64



The first one 'TOTAL', is the totals on the [Payments Schedule]({{ site.url }}/assets/enron61702insiderpay.pdf) and not a person so it should be removed.  
The second one is not an outlier, it is just the huge payment and stock value of the CEO and chairman of Enron, Kenneth Lay. Datapoints like this are not outliers; in fact anomalies like this may lead us to the rest of the POIs. 
These extreme values lead the rest of the employees to the bottom left corner of the scatterplot. Let's use a logarithmic scale for both axes to unclutter the plot.


```python
data_df.drop("TOTAL", inplace=True)
```


```python
sns.pairplot(data=data_df, vars=["total_payments", "exercised_stock_options", "restricted_stock", "total_stock_value"], hue="poi")
```




    <seaborn.axisgrid.PairGrid at 0x7f8d7fe96c50>




![png]({{ site.url }}/assets/2017-04-04-Fraud-Detection-Using-Machine-Learning/output_35_1.png)


With the "TOTAL" removed the scatter plots are much more uncluttered and we can see some trends on them.  
We can notice also a negative value on *Restricted Stock* variable, an indication that more outliers may exist.  
Since the data have been taken from a financial statement an error may have introduced during the data entry / scrapping.  
We can make a first sanity by checking if the individual values sum with the totals of each category (*Total Payments*, *Total Stock Value*).


```python
print(data_df.sum()[1:11])
print("---")
print("Sum all 'payment' variables:", sum(data_df.sum()[1:10]))
```

    salary                  26704229.0
    bonus                   97343619.0
    long_term_incentive     48521928.0
    deferred_income        -27890391.0
    deferral_payments       31980896.0
    loan_advances           83925000.0
    other                   42805453.0
    expenses                 5094049.0
    director_fees            1437166.0
    total_payments         325304226.0
    dtype: float64
    ---
    Sum all 'payment' variables: 309921949.0



```python
print(data_df.sum()[11:15])
print("---")
print("Sum all 'stock' variables:", sum(data_df.sum()[11:14]))
```

    exercised_stock_options      298915485.0
    restricted_stock             125069226.0
    restricted_stock_deferred     10572178.0
    total_stock_value            419009128.0
    dtype: float64
    ---
    Sum all 'stock' variables: 434556889.0


We can see that the totals do not match. I need to check employee by employee to find the errors.


```python
alist = []
for line in data_df.itertuples():
    if sum(line[2:11]) != line[11] or sum(line[12:15]) != line[15]:
        alist.append(line[0])
data_df.loc[alist]
```




<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>poi</th>
      <th>salary</th>
      <th>bonus</th>
      <th>long_term_incentive</th>
      <th>deferred_income</th>
      <th>deferral_payments</th>
      <th>loan_advances</th>
      <th>other</th>
      <th>expenses</th>
      <th>director_fees</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
      <th>from_poi_to_this_person</th>
      <th>shared_receipt_with_poi</th>
      <th>to_messages</th>
      <th>from_this_person_to_poi</th>
      <th>from_messages</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BELFER ROBERT</th>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-102500.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3285.0</td>
      <td>102500.0</td>
      <td>3285.0</td>
      <td>0.0</td>
      <td>44093.0</td>
      <td>-44093.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>BHATNAGAR SANJAY</th>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>137864.0</td>
      <td>0.0</td>
      <td>137864.0</td>
      <td>15456290.0</td>
      <td>2604490.0</td>
      <td>-2604490.0</td>
      <td>15456290.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>463.0</td>
      <td>523.0</td>
      <td>1.0</td>
      <td>29.0</td>
    </tr>
  </tbody>
</table>
</div>



Comparing the above results with the [Payments Schedule]({{ site.url }}/assets/enron61702insiderpay.pdf) we can see that there are some errors in the data.  
The right values are:

<div class="table">
<table border="1" class="dataframe">
  <tr>
    <th></th>
    <th>poi</th>
    <th>salary</th>
    <th>deferral_payments</th>
    <th>loan_advances</th>
    <th>bonus</th>
    <th>deferred_income</th>
    <th>expenses</th>
    <th>other</th>
    <th>long_term_incentive</th>
    <th>director_fees</th>
    <th>total_payments</th>
    <th>restricted_stock_deferred</th>
    <th>exercised_stock_options</th>
    <th>restricted_stock</th>
    <th>total_stock_value</th>
  </tr>
  <tr>
    <th>BELFER ROBERT</th>
    <td>False</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>-102500</td>
    <td>3285</td>
    <td>0</td>
    <td>0</td>
    <td>102500</td>
    <td>3285</td>
    <td>-44093</td>
    <td>0</td>
    <td>44093</td>
    <td>0</td>
  </tr>
  <tr>
    <th>BHATNAGAR SANJAY</th>
    <td>False</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>137864</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>137864</td>
    <td>-2604490</td>
    <td>15456290</td>
    <td>2604490</td>
    <td>15456290</td>
  </tr>
</table>
</div>
<br>

```python
data_df.loc["BELFER ROBERT", :] = [
    False, 0, 0, 0, 0, -102500, 3285, 0, 0, 102500, 3285, -44093, 0, 44093, 0,
    0, 0, 0, 0, 0
]
data_df.loc["BHATNAGAR SANJAY", :] = [
    False, 0, 0, 0, 0, 0, 137864, 0, 0, 0, 137864, -2604490, 15456290, 2604490,
    15456290, 0, 463, 523, 1, 29
]
```

Now that we do not have any more outliers we can plot the two aggregated variables, *Total Payments* and *Total Stock Value*.


```python
fig1, ax = plt.subplots()
for poi, data in data_df.groupby(by="poi"):
    ax.plot(data['total_payments'],data['total_stock_value'],'o', label=poi)
ax.legend()
plt.xscale('symlog')
plt.yscale('symlog')
plt.xlabel("Total Payments")
plt.ylabel("Total Stock Value")

plt.show()
```


![png]({{ site.url }}/assets/2017-04-04-Fraud-Detection-Using-Machine-Learning/stock_to_payments.png)


We can see that there are some persons with zero salary or bonus (or both) and none of them is a POI. Since we have a sparse number of POIs it might be beneficial to remove them to have a more dense dataset. I will create a copy of the dataset with the specific persons removed for future evaluation.


```python
data_nbs = data_df[data_df.salary > 0]
data_nbs = data_nbs[data_nbs.bonus > 0]
data_nbs.shape
```




    (81, 20)



"TOTAL" was a false entry in the dataset, I want to investigate if there are more. We can notice that the indexes / names in the dataset are in the form *Sirname Name Initial*. I will search all the indexes using regular expressions and print the indexes that do not follow this pattern.


```python
for index in data_df.index:
    if re.match('^[A-Z]+\s[A-Z]+(\s[A-Z])?$', index):
        continue
    else:
        print(index)
```

    BLAKE JR. NORMAN P
    BOWEN JR RAYMOND M
    DERRICK JR. JAMES V
    DONAHUE JR JEFFREY M
    GARLAND C KEVIN
    GLISAN JR BEN F
    OVERDYKE JR JERE C
    PEREIRA PAULO V. FERRAZ
    SULLIVAN-SHAKLOVITZ COLLEEN
    THE TRAVEL AGENCY IN THE PARK
    WALLS JR ROBERT H
    WHITE JR THOMAS E
    WINOKUR JR. HERBERT S
    YEAGER F SCOTT


There is a "suspicious" index. The **THE TRAVEL AGENCY IN THE PARK**, isn't obviously a name of an employee.


```python
data_df.loc[["THE TRAVEL AGENCY IN THE PARK"]]
```

<div class="table">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>poi</th>
      <th>salary</th>
      <th>bonus</th>
      <th>long_term_incentive</th>
      <th>deferred_income</th>
      <th>deferral_payments</th>
      <th>loan_advances</th>
      <th>other</th>
      <th>expenses</th>
      <th>director_fees</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
      <th>from_poi_to_this_person</th>
      <th>shared_receipt_with_poi</th>
      <th>to_messages</th>
      <th>from_this_person_to_poi</th>
      <th>from_messages</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>THE TRAVEL AGENCY IN THE PARK</th>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>362096.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>362096.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



By cross-checking the above record, we can find that it is a travel agency.

>Payments were made by Enron employees on account of business-related travel to The Travel Agency in the Park (later Alliance Worldwide), which was coowned
by the sister of Enron's former Chairman. Payments made by the Debtor to reimburse employees for these expenses have not been included.

Since it is not an employee we should remove it.


```python
data_df = data_df.drop(["THE TRAVEL AGENCY IN THE PARK"], axis=0)
```

As a final step in outlier investigation, I will search for extreme values. I suspect that because of the nature of the problem, the extreme values is an essential information and they should be kept but let's spot them first.  
I'm using Tukey Fences with 3 IQRs for every single feature.


```python
def outliers_iqr(dataframe, features):
    result = set()
    for feature in features:
        ys = dataframe[[feature]]
        quartile_1, quartile_3 = np.percentile(ys, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = int(round(quartile_1 - (iqr * 3)))
        upper_bound = int(round(quartile_3 + (iqr * 3)))
        partial_result = list(np.where((ys > upper_bound) | (ys < lower_bound))[0])
        print(feature, len(partial_result))
        result.update(partial_result)
        
    print("-----------------------------------------------------")
    print("")
    print("Total number of records with extreme values: " + str(len(result)))
    
    return list(result)
```


```python
cols.remove("poi")
xtr_values =outliers_iqr(data_df, cols)
```

    salary 1
    bonus 5
    long_term_incentive 11
    deferred_income 24
    deferral_payments 34
    loan_advances 5
    other 13
    expenses 2
    director_fees 15
    total_payments 5
    exercised_stock_options 11
    restricted_stock 10
    restricted_stock_deferred 17
    total_stock_value 11
    from_poi_to_this_person 11
    shared_receipt_with_poi 4
    to_messages 6
    from_this_person_to_poi 13
    from_messages 20
    -----------------------------------------------------
    
    Total number of records with extreme values: 93



```python
a = data_df.loc[:, "poi"].value_counts()
poi_density = a[1]/(a[0]+a[1])
print("POI density on the original dataset: " + str(poi_density))
a = data_df.ix[xtr_values, "poi"].value_counts()
poi_density_xtr = a[1]/(a[0]+a[1])
poi_density_xtr = a[1]/(a[0]+a[1])
print("POI density on the subset with the extreme values: " + str(poi_density_xtr))

print("Difference: " +str((poi_density_xtr - poi_density) / poi_density))
```

    POI density on the original dataset: 0.125874125874
    POI density on the subset with the extreme values: 0.161290322581
    Difference: 0.281362007168


We see that in the subset of employees with extreme value to at least one variable, there are 28% more POIs than in the general dataset. This justify our intuition that the extreme values are related with being a POI, thus we will not remove them.

Now that the dataset is clear of outliers we can find the final dimensions and split the labels from the features and have a first scoring as a baseline for the rest of the analysis. I will use the LinearSVC classifier which seems the more appropriate to begin.  
The dataset is quite sparse to use the usual *training*/*testing* splitting so instead I will use the whole dataset for training and cross validation for testing. The procedure has been code in the ```test_classifier()``` function, part of the *tester.py* file.


```python
data_df.shape
```




    (143, 20)




```python
data_df.loc[:, "poi"].value_counts()
```




    False    125
    True      18
    Name: poi, dtype: int64




```python
def do_split(data):
    X = data.copy()
    #Removing the poi labels and put them in a separate array, transforming it
    #from True / False to 0 / 1
    y = X.pop("poi").astype(int)
    
    return X, y, 
```


```python
X, y = do_split(data_nbs)
```


```python
#test_classifier() demands the dataset in a dictionary and the features labels
#in a list with 'poi' first.
features = data_df.columns.tolist()
data = data_df.to_dict(orient='index')

test_classifier(LinearSVC(random_state=42), data, features)
```

    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
         verbose=0)
    	Accuracy: 0.72867	Precision: 0.20293	Recall: 0.35350	F1: 0.25784	F2: 0.30782
    	Total predictions: 15000	True positives:  707	False positives: 2777	False negatives: 1293	True negatives: 10223
    


We are interested in the ability of the classifier not to label as Person Of Interest (POI) a person that is not, and also to find all the POIs so the metrics that we are most interested are **Precision** and **Recall**. Since we want to maximize both in the same time we will try to maximize the **F1** score which can be interpreted as a weighted average of the precision and recall.  
We can see that the initial scores are very low with the LinearSVC classifier being poor in classifying the right persons. This result might be either due to inability of the specific algorithm to make a good prediction on the specific dataset or because of the need of preprocessing of the dataset (or both).  
In the following steps, I will explore the impact of adding and transforming features to the performance of the model. Finally, I will try several algorithm families to end up with a combination of features / algorithm to build the best performing model for the specific problem.

# Optimize Feature Selection/Engineering

## Create new features

In some cases the value of a variable is less important than its proportion to an aggregated value. As an example from the current dataset a bonus of 100,000 is less informative than a bonus 3 times the salary, or "500 sent email to POIs" is far less informative than "half of the sent emails have been sent to POIs".  
For this reason I will add the proportions of each variable to its category's sum.


```python
data = data_df.copy()
data.loc[:, "salary_p"] = data.loc[:, "salary"]/data.loc[:, "total_payments"]
data.loc[:, "deferral_payments_p"] = data.loc[:, "deferral_payments"]/data.loc[:, "total_payments"]
data.loc[:, "loan_advances_p"] = data.loc[:, "loan_advances"]/data.loc[:, "total_payments"]
data.loc[:, "bonus_p"] = data.loc[:, "bonus"]/data.loc[:, "total_payments"]
data.loc[:, "deferred_income_p"] = data.loc[:, "deferred_income"]/data.loc[:, "total_payments"]
data.loc[:, "expenses_p"] = data.loc[:, "expenses"]/data.loc[:, "total_payments"]
data.loc[:, "other_p"] = data.loc[:, "other"]/data.loc[:, "total_payments"]
data.loc[:, "long_term_incentive_p"] = data.loc[:, "long_term_incentive"]/data.loc[:, "total_payments"]
data.loc[:, "director_fees_p"] = data.loc[:, "director_fees"]/data.loc[:, "total_payments"]

data.loc[:, "restricted_stock_deferred_p"] = data.loc[:, "restricted_stock_deferred"]/data.loc[:, "total_stock_value"]
data.loc[:, "exercised_stock_options_p"] = data.loc[:, "exercised_stock_options"]/data.loc[:, "total_stock_value"]
data.loc[:, "restricted_stock_p"] = data.loc[:, "restricted_stock"]/data.loc[:, "total_stock_value"]

data.loc[:, "from_poi_to_this_person_p"] = data.loc[:, "from_poi_to_this_person"]/data.loc[:, "to_messages"]
data.loc[:, "shared_receipt_with_poi_p"] = data.loc[:, "shared_receipt_with_poi"]/data.loc[:, "to_messages"]

data.loc[:, "from_this_person_to_poi_p"] = data.loc[:, "from_this_person_to_poi"]/data.loc[:, "from_messages"]
    
data.replace(to_replace=np.NaN, value=0, inplace=True)
data.replace(to_replace=np.inf, value=0, inplace=True)
data.replace(to_replace=-np.inf, value=0, inplace=True)
```

Now we can plot the importance of the features of the "enriched" dataset by using the same classifier.


```python
def plot_importance(dataset):
    X, y = do_split(dataset)

    selector = SelectPercentile(percentile=100)
    a = selector.fit(X, y)

    plt.figure(figsize=(12,9))
    sns.barplot(y=X.columns, x=a.scores_)
```


```python
plot_importance(data)
```


![png]({{ site.url }}/assets/2017-04-04-Fraud-Detection-Using-Machine-Learning/features_importance.png)


Comparing the newly created features with the original we can see that the proportions of "Long Term Incentive", "Restricted Stock Deferred" and "From This Person to POI" score higher than the original features. We will keep these and remove the original values. to avoid bias the model towards a specific feature by using both the original value and its proportion.


```python
#Adding the proportions
data_df.loc[:, "long_term_incentive_p"] = data_df.loc[:, "long_term_incentive"]/data_df.loc[:, "total_payments"]
data_df.loc[:, "restricted_stock_deferred_p"] = data_df.loc[:, "restricted_stock_deferred"]/data_df.loc[:, "total_stock_value"]
data_df.loc[:, "from_this_person_to_poi_p"] = data_df.loc[:, "from_this_person_to_poi"]/data_df.loc[:, "from_messages"]
#Removing the original values.
data_df.drop("long_term_incentive", axis=1)
data_df.drop("restricted_stock_deferred", axis=1)
data_df.drop("from_this_person_to_poi", axis=1)
#Correcting NaN and infinity values created by zero divisions
data_df.replace(to_replace=np.NaN, value=0, inplace=True)
data_df.replace(to_replace=np.inf, value=0, inplace=True)
data_df.replace(to_replace=-np.inf, value=0, inplace=True)
```


```python
plot_importance(data_df)
```


![png]({{ site.url }}/assets/2017-04-04-Fraud-Detection-Using-Machine-Learning/features_importance2.png)


## Feature Selection

For feature selection I will use LinearSVC classifier and I will test both univariate feature selection (KBest) and Primary Component Analysis.


```python
X, y = do_split(data_df)
```


```python
warnings.filterwarnings('ignore')

pipe = Pipeline([('reduce_dim', PCA(random_state=42)),
                 ('classify', LinearSVC(random_state=42))])

N_FEATURES_OPTIONS = list(range(2,21))

param_grid = [
    {
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
    },
    {
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
    },
]
reducer_labels = ['PCA', 'KBest']
cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

X, y = do_split(data_df)
grid.fit(X, y)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
mean_scores = mean_scores.reshape(-1, len(N_FEATURES_OPTIONS))
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)
plt.figure(figsize=(12,9))

for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label)

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Classification accuracy')
plt.ylim((0, 0.4))
plt.legend(loc='upper left')

plt.show()
grid.best_estimator_
```


![png]({{ site.url }}/assets/2017-04-04-Fraud-Detection-Using-Machine-Learning/feature_reduction.png)





    Pipeline(steps=[('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=5, random_state=42,
      svd_solver='auto', tol=0.0, whiten=False)), ('classify', LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
         verbose=0))])



We can see that we get the best result if we reduce the features of the model to :


```python
selector = SelectKBest(k=6)
selector.fit(X, y)
X.columns[selector.get_support()].tolist()
```




    ['salary',
     'bonus',
     'exercised_stock_options',
     'total_stock_value',
     'long_term_incentive_p',
     'from_this_person_to_poi_p']



The score of LinearSVC on the reduced dataset is:


```python
features = data_df.columns.tolist()
data_dict = data_df.to_dict(orient='index')
test_classifier(grid.best_estimator_, data_dict, features)
```

    Pipeline(steps=[('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=5, random_state=42,
      svd_solver='auto', tol=0.0, whiten=False)), ('classify', LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
         verbose=0))])
    	Accuracy: 0.74813	Precision: 0.18068	Recall: 0.25150	F1: 0.21028	F2: 0.23322
    	Total predictions: 15000	True positives:  503	False positives: 2281	False negatives: 1497	True negatives: 10719
    


There is a 17% improvement on the f1 score with 20% better precision and 12% better recall.

## Features Scaling

There are features in the dataset with big differences in scaling. For example *salary* appears with values between 0 and 2.5 millions while *from_this_person_to_poi_p* takes values in the range [0,1). Some of the algorithms we will evaluate may behave badly because of these differences so we need to scale them.  

Having sparse data it is better to use ```MaxAbsScaler()``` for this transformation.


```python
#warnings.filterwarnings('ignore')

pipe = Pipeline([('scale', MaxAbsScaler()),
                 ('reduce_dim', PCA(random_state=42)),
                 ('classify', LinearSVC(random_state=42))])

N_FEATURES_OPTIONS = list(range(2, 21))

param_grid = [
    {
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
    },
    {
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
    },
]
reducer_labels = ['PCA', 'KBest']
cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

grid.fit(X, y)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
mean_scores = mean_scores.reshape(-1, len(N_FEATURES_OPTIONS))
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)
plt.figure(figsize=(12,9))

for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label)

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Classification accuracy')
plt.ylim((0, 0.4))
plt.legend(loc='upper left')

plt.show()
grid.best_estimator_
```


![png]({{ site.url }}/assets/2017-04-04-Fraud-Detection-Using-Machine-Learning/output_86_0.png)





    Pipeline(steps=[('scale', MaxAbsScaler(copy=True)), ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=2, random_state=42,
      svd_solver='auto', tol=0.0, whiten=False)), ('classify', LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
         verbose=0))])




```python
features = data_df.columns.tolist()
data = data_df.to_dict(orient='index')
test_classifier(grid.best_estimator_, data, features)
```

    Pipeline(steps=[('scale', MaxAbsScaler(copy=True)), ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=2, random_state=42,
      svd_solver='auto', tol=0.0, whiten=False)), ('classify', LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
         verbose=0))])
    	Accuracy: 0.86867	Precision: 0.53589	Recall: 0.11200	F1: 0.18528	F2: 0.13305
    	Total predictions: 15000	True positives:  224	False positives:  194	False negatives: 1776	True negatives: 12806
    


This is a quite interesting result. After scaling, model's Precision doubled but Recall fell lower than half, giving the model lower f1 score than before. I'll edit the pipeline to add the option of no-scaling and also the StandardScaler which is a more frequently used option but not very good for sparse data.


```python
pipe = Pipeline([('scale', MaxAbsScaler()),
                 ('reduce_dim', PCA(random_state=42)),
                 ('classify', LinearSVC(random_state=42))])

N_FEATURES_OPTIONS = list(range(2, 21))

param_grid = [
    {
        'scale':[None, MaxAbsScaler(), StandardScaler()],
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
    },
    {
        'scale':[None, MaxAbsScaler(), StandardScaler()],
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
    },
]
reducer_labels = ['PCA', 'KBest']
cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

grid.fit(X, y)
grid.best_estimator_
```




    Pipeline(steps=[('scale', MaxAbsScaler(copy=True)), ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=2, random_state=42,
      svd_solver='auto', tol=0.0, whiten=False)), ('classify', LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
         verbose=0))])



It's now clear that LinearSVC performs better without scaling but I will keep the scaling options in the pipeline because other algorithms may have different behaviour.

# Algorithm Selection and Tuning

Having finished with the preprocessing of the dataset I will begin evaluating the performance of suitable algorithms. First I will evaluate different algorithms' families by using their most usual member with some generalized attributes and once I conclude with the family I will evaluate different members.

## Algorithm Family Selection

### Support Vector Machine

From Support Vector Machines, I will evaluate Support Vector Classifier with the default kernel (RBF) and Penalty Parameters 0.1, 1, 10


```python
#warnings.filterwarnings('ignore')

pipe = Pipeline([('scale', MaxAbsScaler()),
                 ('reduce_dim', PCA(random_state=42)),
                 ('classify', SVC(random_state=42))])

N_FEATURES_OPTIONS = list(range(2, 10))
C_VALUES = [0.1, 1, 10]


param_grid = [
    {
        'scale':[None, MaxAbsScaler()],
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_VALUES
    },
    {
        'scale':[None, MaxAbsScaler()],
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_VALUES
    },
]

cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

grid.fit(X, y)

print(grid.best_estimator_)
```

    Pipeline(steps=[('scale', MaxAbsScaler(copy=True)), ('reduce_dim', SelectKBest(k=4, score_func=<function f_classif at 0x7f8da22401e0>)), ('classify', SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=42, shrinking=True,
      tol=0.001, verbose=False))])



```python
test_classifier(grid.best_estimator_, data, features)
```

    Pipeline(steps=[('scale', MaxAbsScaler(copy=True)), ('reduce_dim', SelectKBest(k=4, score_func=<function f_classif at 0x7f8da22401e0>)), ('classify', SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=42, shrinking=True,
      tol=0.001, verbose=False))])
    	Accuracy: 0.86467	Precision: 0.43590	Recall: 0.05100	F1: 0.09132	F2: 0.06194
    	Total predictions: 15000	True positives:  102	False positives:  132	False negatives: 1898	True negatives: 12868
    


Support Vector Machines gave as the highest Precision so far but in the same time the worst Recall achieving an F1 score of 0.05405


```python
h = .001  #step size in the mesh

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

scale = MaxAbsScaler()
X_trans = scale.fit_transform(X)
select = SelectKBest(k=2)
X_trans = select.fit_transform(X_trans, y)
y_trans = y.values

clf = SVC(C=1,
          cache_size=200,
          class_weight=None,
          coef0=0.0,
          decision_function_shape=None,
          degree=3,
          gamma='auto',
          kernel='rbf',
          max_iter=-1,
          probability=False,
          random_state=42,
          shrinking=True,
          tol=0.001,
          verbose=False)
clf.fit(X_trans, y_trans)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_trans[:, 0].min() - 0.1, X_trans[:, 0].max() + 0.1
y_min, y_max = X_trans[:, 1].min() - 0.1, X_trans[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y, cmap=cmap_bold, alpha=0.3)

# Legend Data
classes = ['POI','Non-POI']
class_colours = ['#0000FF', '#FF0000']
recs = []
for i in range(0,len(class_colours)):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))

# Plot Legend ('loc' == position):
plt.legend(recs,classes,loc=4)
plt.title("Support Vectors Classifier with RBF kernel")
plt.xlabel(X.columns[select.get_support()][0])
plt.ylabel(X.columns[select.get_support()][1])

fig.savefig('Figures/svc.png')
plt.show()
```


![png]({{ site.url }}/assets/2017-04-04-Fraud-Detection-Using-Machine-Learning/svc.png)


### Nearest Neighbors

From the Nearest Neighbors family, I will evaluate *KNeighborsClassifier*. Since this algorithm is very fast, I'm able to evaluate several parameters.


```python
#warnings.filterwarnings('ignore')

pipe = Pipeline([('scale', MaxAbsScaler()),
                 ('reduce_dim', PCA(random_state=42)),
                 ('classify', KNeighborsClassifier())])

N_FEATURES_OPTIONS = list(range(2, 21))
N_NEIGHBORS = [1, 3, 5]

param_grid = [
    {
        'scale': [None, MaxAbsScaler()],
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__weights': ["uniform", "distance"],
        'classify__n_neighbors': N_NEIGHBORS,
        'classify__p':[1, 2]
    },
    {
        'scale': [None, MaxAbsScaler()],
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__weights': ["uniform", "distance"],
        'classify__n_neighbors': N_NEIGHBORS,
        'classify__p':[1, 2]
    },
]

cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

grid.fit(X, y)


print(grid.best_estimator_)
```

    Pipeline(steps=[('scale', MaxAbsScaler(copy=True)), ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=2, random_state=42,
      svd_solver='auto', tol=0.0, whiten=False)), ('classify', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=3, p=2,
               weights='distance'))])



```python
features = data_df.columns.tolist()
data = data_df.to_dict(orient='index')
test_classifier(grid.best_estimator_, data, features)
```

    Pipeline(steps=[('scale', MaxAbsScaler(copy=True)), ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=2, random_state=42,
      svd_solver='auto', tol=0.0, whiten=False)), ('classify', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=3, p=2,
               weights='distance'))])
    	Accuracy: 0.85593	Precision: 0.43385	Recall: 0.26400	F1: 0.32826	F2: 0.28643
    	Total predictions: 15000	True positives:  528	False positives:  689	False negatives: 1472	True negatives: 12311
    


*KNeighborsClassifier* so far has the most balanced result across Precision and Recall. It has the best F1 score so far.

### Ensemble Methods

In the Ensemble Methods, I will evaluate both Random Forest from the Averaging subcategory and AdaBoost from Boosting Methods.  
Both classifiers are based on recursive partitioning, so they do not require features to be normalized or scaled, since it is invariant to monotonic transformations of the features, thus we can remove the scaling from the pipeline.

#### Averaging Methods


```python
#warnings.filterwarnings('ignore')

pipe = Pipeline([('reduce_dim', PCA(random_state=42)),
                 ('classify',  RandomForestClassifier(random_state=42))])

N_FEATURES_OPTIONS = list(range(2, 21))
N_TREES = [1, 2, 3]

param_grid = [
    {
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__n_estimators': N_TREES,
        'classify__criterion': ["gini", "entropy"]
    },
    {
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__n_estimators': N_TREES,
        'classify__criterion': ["gini", "entropy"]
    },
]

cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

grid.fit(X, y)


print(grid.best_estimator_)
```

    Pipeline(steps=[('reduce_dim', SelectKBest(k=20, score_func=<function f_classif at 0x7f8da22401e0>)), ('classify', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=3, n_jobs=1, oob_score=False, random_state=42,
                verbose=0, warm_start=False))])



```python
features = data_df.columns.tolist()
data = data_df.to_dict(orient='index')
test_classifier(grid.best_estimator_, data, features)
```

    Pipeline(steps=[('reduce_dim', SelectKBest(k=20, score_func=<function f_classif at 0x7f8da22401e0>)), ('classify', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=3, n_jobs=1, oob_score=False, random_state=42,
                verbose=0, warm_start=False))])
    	Accuracy: 0.84667	Precision: 0.38497	Recall: 0.25100	F1: 0.30387	F2: 0.26978
    	Total predictions: 15000	True positives:  502	False positives:  802	False negatives: 1498	True negatives: 12198
    


Random Forest has also a balanced behavior but the scores are significantly lower comparing to Nearest Neighbor.

#### Boosting methods


```python
#warnings.filterwarnings('ignore')

pipe = Pipeline([('reduce_dim', PCA(random_state=42)),
                 ('classify',  AdaBoostClassifier(random_state=42))])

N_FEATURES_OPTIONS = list(range(2, 21))
N_ESTIMATORS = [1, 10, 100]

param_grid = [
    {
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__n_estimators': N_ESTIMATORS,
        'classify__algorithm': ['SAMME', 'SAMME.R']
    },
    {
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__n_estimators': N_ESTIMATORS,
        'classify__algorithm': ['SAMME', 'SAMME.R']
    },
]

cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

grid.fit(X, y)


print(grid.best_estimator_)
```

    Pipeline(steps=[('reduce_dim', SelectKBest(k=4, score_func=<function f_classif at 0x7f8da22401e0>)), ('classify', AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=1.0,
              n_estimators=10, random_state=42))])



```python
features = data_df.columns.tolist()
data = data_df.to_dict(orient='index')
test_classifier(grid.best_estimator_, data, features)
```

    Pipeline(steps=[('reduce_dim', SelectKBest(k=4, score_func=<function f_classif at 0x7f8da22401e0>)), ('classify', AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=1.0,
              n_estimators=10, random_state=42))])
    	Accuracy: 0.85087	Precision: 0.38185	Recall: 0.19150	F1: 0.25508	F2: 0.21271
    	Total predictions: 15000	True positives:  383	False positives:  620	False negatives: 1617	True negatives: 12380
    


AdaBoost performed a little better than Random Forest but still the difference with Nearest Neighbor is quite strong. It seems like a safe conclusion that the best classifiers family for the specific problem is Nearest Neighbor.

## Algorithm Selection

### K-Nearest Neighbor

We have already evaluated K-Nearest Neighbor and we have the following results:  
```Accuracy: 0.85193	Precision: 0.43233	Recall: 0.35300	F1: 0.38866	F2: 0.36645
	Total predictions: 15000	True positives:  706	False positives:  927	False negatives: 1294	True negatives: 12073```

### Nearest Centroid Classifier


```python
#warnings.filterwarnings('ignore')

pipe = Pipeline([('scale', MaxAbsScaler()),
                 ('reduce_dim', PCA(random_state=42)),
                 ('classify', NearestCentroid())])

N_FEATURES_OPTIONS = list(range(2, 5))

param_grid = [
    {
        'scale': [None, MaxAbsScaler()],
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__metric': ["euclidean", "manhattan"],
        'classify__shrink_threshold': [None, 0.1, 1, 10]
    },
    {
        'scale': [None, MaxAbsScaler()],
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__metric': ["euclidean", "manhattan"],
        'classify__shrink_threshold': [None, 0.1, 1, 10]
    },
]

cv = StratifiedShuffleSplit(random_state=42)
grid = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)

grid.fit(X, y)

my_classifier = grid.best_estimator_
my_classifier
```




    Pipeline(steps=[('scale', MaxAbsScaler(copy=True)), ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=4, random_state=42,
      svd_solver='auto', tol=0.0, whiten=False)), ('classify', NearestCentroid(metric='manhattan', shrink_threshold=None))])




```python
features = data_df.columns.tolist()
data = data_df.to_dict(orient='index')
test_classifier(my_classifier, data, features)
```

    Pipeline(steps=[('scale', MaxAbsScaler(copy=True)), ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=4, random_state=42,
      svd_solver='auto', tol=0.0, whiten=False)), ('classify', NearestCentroid(metric='manhattan', shrink_threshold=None))])
    	Accuracy: 0.73193	Precision: 0.29648	Recall: 0.73600	F1: 0.42268	F2: 0.56768
    	Total predictions: 15000	True positives: 1472	False positives: 3493	False negatives:  528	True negatives: 9507
    



```python
h = .02  #step size in the mesh

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

scale = MaxAbsScaler()
X_trans = scale.fit_transform(X)
pca = PCA(copy=True,
          iterated_power='auto',
          n_components=2,
          random_state=42,
          svd_solver='auto',
          tol=0.0,
          whiten=False)
X_trans = pca.fit_transform(X_trans)
y_trans = y.values

clf = NearestCentroid(metric='manhattan', shrink_threshold=None)
clf.fit(X_trans, y_trans)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_trans[:, 0].min() - 0.1, X_trans[:, 0].max() + 0.1
y_min, y_max = X_trans[:, 1].min() - 0.1, X_trans[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y_trans, cmap=cmap_bold, alpha=0.3)

# Legend Data
classes = ['POI','Non-POI']
class_colours = ['#0000FF', '#FF0000']
recs = []
for i in range(0,len(class_colours)):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))

# Plot Legend ('loc' == position):
plt.legend(recs,classes,loc=4)

plt.title("NearestCentroid classifier with manhattan metric")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")

plt.show()
fig.savefig('Figures/nearest_centroid.png')
```


![png]({{ site.url }}/assets/2017-04-04-Fraud-Detection-Using-Machine-Learning/nearest_centroid.png)


Nearest Centroid Classifier achieved an even better f1 score (0.44) with an average Precision and an excellent Recall.

## Algorithm Tuning

Nearest Centroid Classifier is a very fast algorithm and additionally it accepts too few arguments, as a result we were able to try all the different combination during the evaluation.  
We can conclude that the best classification of the employees to POIs can be achieved with the following steps:  

|Step             |Algorithm(attributes)                                                                                           |
|:---------------:|:--------------------------------------------------------------------------------------------------------------:|
|Scaling          |MaxAbsScaler(copy=True)                                                                                         |
|Feature Selection|PCA(copy=True, iterated_power='auto', n_components=2, random_state=42, svd_solver='auto', tol=0.0, whiten=False)|
|Classification   |NearestCentroid(metric='manhattan', shrink_threshold=None)                                                      |

# Project Closure

## Dumping files


```python
pickle.dump(data_df.to_dict(orient='index'), open("my_dataset.pkl", "wb"))
pickle.dump(my_classifier, open("my_classifier.pkl", "wb"))
pickle.dump(features, open("my_feature_list.pkl", "wb"))
```

# Future Improvements

There are at least two improvements I could have done to the Project.  

The first one is a tweak in the classification algorithm.  
We noticed in the scatterplot between Total Stock Value vs Total Payments, that none of the persons who had zero Salary or zero Stocks was a POI.


```python
fig1
```




![png]({{ site.url }}/assets/2017-04-04-Fraud-Detection-Using-Machine-Learning/stock_to_payments.png)



Actually, this is so sensible that we could say that it is always true and thus create a rule out of this. If you consider that the persons fall under this category are {{146-81}} out of 146, we could improve the performance of our classification up to:


```python
((65*1)+(81*0.44))/(65+81)
```




    0.6893150684931507



(*of course this score isn't exact because the algorithm selection and evaluation process should be followed again with the dataset without the zero Payments / Stocks employees*)

So, I'm leaving for a future improvement the creation of a custom classifier that would classify as non-POI every employee with zero Payments or Stocks and by using a machine learning algorithm for the rest.

***

The second one is the usage of the content of the emails on the classification. Currently, I just used aggregated values from email headers in the form of the "emails features. Using NLP to dig into the actual content of the emails and vectorize them, may improve the classification further.
