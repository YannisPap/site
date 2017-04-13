---
title: Exploring Uber demand in NYC (Summary)
date: 2017-02-10 20:10:01 +02:00
excerpt: "Exploring ridership of New York's boroughs and creating a set of prediction models."
header:
  overlay_image: https://media.timeout.com/images/103444978/image.jpg
  overlay_filter: 0.5
  caption: "Photo credit: [**Timeout**](https://www.timeout.com)"
  teaser: https://media.timeout.com/images/103444978/image.jpg
tags:
- RStudio
- R packages
- Exploratory Data Analysis Techniques
- Plotting in R
- Exploratory data analysis techniques
description:
- Used R and applied Exploratory Data Analysis (EDA) techniques to explore relationships
  in one variable to multiple variables, distributions, outliers, anomalies and finally
  create a model that predicts ridership.
comments: true
share: true
layout: single
---

{% include toc %}

# Introduction

The dataset I used for this project included data of Uber cars' ridership in the city of New York for the first six months of 2015. As I was exploring it, I noticed that, against my initial intuition, the weather variables had not any or very weak impact on the ridership.  
Going further in my analysis it was getting more clear that the demand follows specific patterns both during the day and during the week.  
Also, I noticed a general trend of rising demand during the six months, led the total demand from 2,000 pickups per hour to 3,500.  

Using the above conclusions I was able to model the demand with forecasting horizons from a week to next hour. These models can be used in different occasions. For example someone could use the weekly forecasting model to have a general view of the next week's demand. On the other hand, a real time system could compare the prediction per borough, with the positions of Uber cars and highlight the areas accordingly to drivers' applications helping them to roam more efficiently through the city.  

Since the model is based on past observations it is prone to wrong estimations on very irregular conditions. Additionally since current observations affect future prediction, demand out of the ordinary levels may lead to wrong estimation at some point to later predictions.  

Below you may find the final plots of the project. There is also available a [Full analysis](https://yannispap.github.io/Exploring-Uber-Demand/)

***

# Pickups Distribution

![download.png](/assets/2017-02-10-Exploring-Uber-Demand/download.png)

The distribution of the four major boroughs, on a square rooted scale, are mainly normal to bimodal because of the quick rise of the demand during the morning hours.  
Staten Island's pickups follow a geometric distribution because of the very small demand in the area.  
Finally, on EWR the demand is practically zero with a very few pickups that we may consider as outliers.

***

# Pickups Heatmap

![plot2.png](/assets/2017-02-10-Exploring-Uber-Demand/plot2.png)

On the above heat maps we can see the demand pattern on each borough.  
The four major boroughs follow the same pattern both during the day and through the week.  
On working days the demand falls after midnight and then at around 6 o'clock start rising quickly, then it hits a plateau during the afternoon and then rises again during the evening/night. On the X axis (during the week), the demand starts low on Monday and then rises until Saturday, when it tops and then on Sunday starts falling again. The pattern is more obvious on Manhattan and Brooklyn.  
On the two minor boroughs, Staten Island's demand looks random during the day but again we can see that the demand slightly rises as we move through the week. EWR, as we noted before has practically no demand.

***

# Prediction Results

![plot3.png](/assets/2017-02-10-Exploring-Uber-Demand/plot3.png)

I concluded the Exploratory Data Analysis process with the creation of some models to predict the demand. In general, the models have a very good fit with just one occasion of underestimating the actual demand on the highest day of the six months period.
