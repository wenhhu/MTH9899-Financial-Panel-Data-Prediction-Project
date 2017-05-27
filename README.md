# MTH9899-Machine-Learning-Final-Project

This repo is the final project of MTH9899 Machine learning course. 

## Data
Data The dataset you will be given consists of approximately 140k rows of data with 31
features. Due to the proprietary nature of the dataset, many of the times, symbols etc
have been replaced:

### Rows
* Each row represents an observation. It is your discretion whether to keep them all, consider a subset, or maybe even enlarge it.

### Columns
* timestamp: the numbers are ordered, i.e "0" represents earlier time than "1"
* id: different numbers represent different fnancial assets
* 'x*': There are 27 cols with names that start with 'x' - those will be the features that you might consider using for your models
* 'y': dependent variable that your model(s) will be predicting
* 'weight': weights associated with each observation
We will be using weighted R2 as final benchmark. Since there might be slightly different
ways to calculate, we will use sklearn r2 score implementation as the only benchmark.
