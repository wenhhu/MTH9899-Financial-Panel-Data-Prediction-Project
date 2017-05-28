# MTH9899-Machine-Learning-Final-Project

This repo is the final project of MTH9899 Machine learning course. The goal of this project is to apply various ML algorithms to a real-world Financial dataset and predict the movement of market. 

## Data description
Data The dataset you will be given consists of approximately 140k rows of data with 31 features. Due to the proprietary nature of the dataset, many of the times, symbols etc have been replaced:

### Rows
* Each row represents an observation. It is your discretion whether to keep them all, consider a subset, or maybe even enlarge it.

### Columns
* timestamp: the numbers are ordered, i.e "0" represents earlier time than "1"
* id: different numbers represent different fnancial assets
* 'x*': There are 27 cols with names that start with 'x' - those will be the features that you might consider using for your models
* 'y': dependent variable that your model(s) will be predicting
* 'weight': weights associated with each observation

We choose to use Weighted R2 calculated with sklearn r2 score implementation as the only metric of our prediction.

### Holdout data
A hold-out test dataset that has about 25% of the data is preserved.

## Model briefing
The finalized model are an ensembled model consists of ExtraTree and XGBoost. The stacking of the results from these estimators is done through a randomforest classifier. Cross validation is done thoroughly in XGBoost model both in a forward chaining style and naive style with 4/1 splitting. ExtraTree parameter tuning is done more intuitively due to time limit. More details can be found in the document. Our final hold-out score (weighted r-square) is 44 bps.

