import pickle
import numpy as np

import pandas as pd
import xgboost as xgb

# Please change the dir to the training and hold out dir
with open('../ml_finalproj_train_vF.pkl', 'rb') as f:
    rawdata = pickle.load(f)
    
with open('../ml_finalproj_holdout.pkl', 'rb') as f:
    rawdata_holdout = pickle.load(f)

# Utility functions

def winsorize(x):
	# winsorze data up to 4 times of standard deviation
    y = x.copy()
    thresh = 4*np.std(x)
    y[y > thresh] = thresh
    y[y < -thresh] = -thresh
    return y

def AddMovingAvg(data):
	# Add moving average features
    new_columns = ['x22_avg','x17_avg','x2_avg','x25_30_avg','x42_51_avg']
    new_data = pd.concat([data, pd.DataFrame(columns=new_columns)], axis=1)
    new_data['x25_30'] = new_data.x25+new_data.x30
    new_data['x42_51'] = new_data.x42+new_data.x51

    ids = new_data['id'].unique()
    for i in ids:
        s_i = new_data.loc[new_data.id==i, ['x22','x17','x2','x25_30','x42_51']]
        new_data.loc[data.id==i, 'x22_avg'] = s_i.x22.rolling(window=5, min_periods=1).mean()
        new_data.loc[data.id==i, 'x17_avg'] = s_i.x17.rolling(window=5, min_periods=1).mean()
        new_data.loc[data.id==i, 'x2_avg'] = s_i.x2.rolling(window=5, min_periods=1).mean()
        new_data.loc[data.id==i, 'x25_30_avg'] = s_i.x25_30.rolling(window=5, min_periods=1).mean()
        new_data.loc[data.id==i, 'x42_51_avg'] = s_i.x42_51.rolling(window=5, min_periods=1).mean()

    for f in new_columns:
        new_data[f] = pd.to_numeric(new_data[f])
        
    return new_data

def AddDiff(datain, cols):
	# Add differencing data corresponding to certain column names in the data
    new_data = pd.concat([datain, pd.DataFrame(np.zeros([datain.shape[0], len(cols)]), columns=[col+'_diff' for col in cols])], axis=1)
    ids = new_data['id'].unique()
    for i in ids:
        s_i = datain.loc[datain.id==i, cols]
        new_data.loc[datain.id==i, [col+'_diff' for col in cols]] = s_i.diff().values
    return new_data

def ave_predict(models, X, n):
	# Aggregate the results of xgboost estimator with different random seed
    res = np.zeros(n)
    for model in models:
        res = res + model.predict(X)
    return res/len(models)

def preprocess(datain, add_diff):
	# Preprocess the input data for XGBoost estimator
    datain[continum].apply(winsorize)

    if add_diff:
        feat = feat + [i+'_diff' for i in continum]
        
    datain = AddMovingAvg(datain)
    if add_diff:
        data = AddDiff(datain, continum)
    else:
        data = datain
    return data

############################################################################################################

# Data precessing for XGBoost

categorical = []
for i in rawdata.columns:
    if i!='timestamp' and rawdata[i][:1000].unique().shape[0]<20:
        categorical.append(i)
categorical
    
notin = ['id','y','weight','timestamp']
# Find continuous variable
continum = [i for i in rawdata.columns if i not in categorical and i not in notin]
filt = ['x22', 'x25', 'x30', 'x42', 'x17','x0','x13']
syn = ['x22_avg','x17_avg','x2_avg','x25_30_avg','x42_51_avg']
# Filter out the features which won't be included in our model
feat = continum+categorical+syn
feat = [i for i in feat if i not in filt]

data_train = preprocess(rawdata, False)

############################################################################################################

# XGBoost Training session

print("Start training XGBoost estimator...")
startTime = time.time()

X_train = data_train[feat].values
y_train = data_train.y.values
weight_train = data_train.weight.values

xgbmat_train = xgb.DMatrix(X_train, label=y_train, weight=np.log(weight_train), feature_names=feat)

# Grid search is used to find the optimal parameters
params_xgb = {'objective'		:'reg:linear',
              'eta'             : 0.05,
              'max_depth'       : 10,
              'gamma'           : 0.00001,
              'subsample'       : 0.5,
              'colsample_bytree': 0.8,
              'min_child_weight': 200,
              'base_score' 		: 0.0000,
              'rounds'     		:  20,
              'early_stopping_rounds' : True
             }

# Multiple XGBoost estimators with different random seeds is ensembled to make our prediction more stable and robust
models = []
for i in range(10):
    params_xgb['seed'] = 2333+i*100
    bst = xgb.train(params_xgb, xgbmat_train)
    models.append(bst)

endTime = time.time()
print('XGBoost Training time: ', endTime-startTime)

############################################################################################################

# Extra Tree Training session

print("Start training ExtraTree estimator...")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import time

raw_data = rawdata

ids = raw_data['id'].unique()
timestamp = raw_data['timestamp'].unique()
ids.sort()

categorical = []
for i in raw_data.columns:
    if i!='timestamp' and raw_data[i][:2000].unique().shape[0]<30:
        categorical.append(i)

notin = ['id','timestamp','y','weight']

continum = [i for i in raw_data.columns if i not in categorical and i not in notin]

new_data=AddMovingAvg(raw_data)

new_features= ['x22_avg','x17_avg','x2_avg', 'x25_30_avg', 'x42_51_avg']
columns_to_use = continum+new_features+['x6', 'x46']
columns_to_use.remove('x22')
columns_to_use.remove('x17')

data_train_extra = new_data.copy()
data_train_extra['x29_1'] = (data_train_extra.x29==1).astype(int)
data_train_extra['x29_2'] = (data_train_extra.x29==2).astype(int)
data_train_extra['x29_3'] = (data_train_extra.x29==3).astype(int)
columns_to_use = columns_to_use + ['x29_1', 'x29_2', 'x29_3']

step = 50

startTime = time.time()

forest = ExtraTreesRegressor(n_estimators=1000, max_depth=10, min_samples_leaf=250, n_jobs=4, bootstrap=False)
forest.fit(data_train_extra[columns_to_use], data_train_extra.y, data_train_extra.weight)

endTime = time.time()
print('time: ', endTime-startTime, 's')

############################################################################################################

# Model ensemble session

print("Start ensembling models...")
from sklearn.ensemble import RandomForestClassifier

xgb_train = ave_predict(models, xgbmat_train, X_train.shape[0])
extra_train = forest.predict(data_train_extra[columns_to_use])

# Calculate the absolute error of two estimators
err_xgb = abs(y_train - xgb_train)
err_extra = abs(y_train - extra_train)

# Absolute error is taken to be the model selection indicator
model_select_train = [1 if err_xgb[i]<err_extra[i] else 0 for i in range(len(err_xgb)) ]
# Use rf classifier to select optimal estimator for a certain row
model_selector = RandomForestClassifier(max_depth=6, n_estimators=40, random_state=1)
model_selector.fit(X_train, model_select_train)

############################################################################################################

# Prediction session
print("Start predicting...")
data_test = preprocess(rawdata_holdout, False)
X_test = data_test[feat].values
weight_test = data_test.weight.values
xgbmat_test = xgb.DMatrix(X_test, weight=np.log(weight_test), feature_names=feat)
xgb_test = ave_predict(models, xgbmat_test, X_test.shape[0])

data_test_extra=AddMovingAvg(rawdata_holdout)
new_features= ['x22_avg','x17_avg','x2_avg', 'x25_30_avg', 'x42_51_avg']
columns_to_use = continum+new_features+['x6', 'x46']
columns_to_use.remove('x22')
columns_to_use.remove('x17')

data_test_extra['x29_1'] = (data_test_extra.x29==1).astype(int)
data_test_extra['x29_2'] = (data_test_extra.x29==2).astype(int)
data_test_extra['x29_3'] = (data_test_extra.x29==3).astype(int)
columns_to_use = columns_to_use + ['x29_1', 'x29_2', 'x29_3']
extra_test = forest.predict(data_test_extra[columns_to_use])

model_select_test = model_selector.predict(data_test[feat])

res = [xgb_test[i] if model_select_test[i] else extra_test[i] for i in range(X_test.shape[0])]
print("Done!")
