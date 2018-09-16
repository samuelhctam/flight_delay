import numpy as np
import pandas as pd
import re
import os

import os
import datetime
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use(u'ggplot')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import roc_auc_score

import xgboost
from lightgbm import LGBMClassifier
import lightgbm as lgb

#Read data
df=pd.read_csv('Data/flight_delays_data.csv')
airport = pd.read_csv('Data/airports.csv')
airline = pd.read_csv('Data/airlines.csv',header=None, names=['index','tmp1','tmp2','tmp3','tmp4','tmp5','tmp6','tmp7'])

#Combing the data
df['Airline_update']=np.where(df.Airline.notnull()==True, df.Airline, df.flight_no.str[0:2])
airport2 = airport[airport.type != 'closed'][airport.iata_code.notnull()][airport.iata_code != '0']
airport2.sort_values(by=['iata_code','latitude_deg'],ascending=False)
airport3=airport2.drop_duplicates('iata_code')
airline2=airline.drop_duplicates('tmp3')

#backup
df_copy= df.copy()
# df=df_copy

df = df.merge(airport3,left_on = 'Arrival', right_on = 'iata_code',how='left').merge(airline2,left_on='Airline_update',right_on='tmp3', how='left')
print df.shape
print df_copy.shape

#since it is more a binary targeting, so create new target as 1 and 0
df['Target'] = (df['is_claim'] == 800).astype(int)
#feature_engineering
#change flight_date to date_formate and extract month and date.
df['flight_date']=pd.to_datetime(df['flight_date'], format="%Y/%m/%d")
#delay_time to nummerical, cancel set all to 3
# df['feat_delay_time']=np.where(df['delay_time']=='Cancelled', 30, df['delay_time']).astype(float)
# delay_time is not something we know in advance and cannot use for prediction

df['feat_Week']=df['Week']
df['feat_std_hour']=df['std_hour']
df['feat_month']=df['flight_date'].dt.month
df['feat_day']=df['flight_date'].dt.day
df['feat_flight_no_len']=df.flight_no.str[2:].str.len()
df['cat_feat_day_of_week'] = df['flight_date'].dt.weekday_name

df['tmp6'].value_counts().head(10)

#5% of the data
#899114*0.05
df['cat_feat_flight_country']=np.where(df['tmp6'].isin(['Hong Kong SAR of China','DRAGON','China','United States','Japan']),df['tmp6'],'Others')
df['cat_feat_flight_country'].value_counts()
df['cat_feat_type']=df['type']
df['feat_type_num']=np.where(df['type'] == 'large_airport', 3, np.where(df['type']=='medium_airport',2,1))
df['feat_internationl_airport']=df['name'].str.contains('International').astype(int)
df['feat_latitude_deg']=df['latitude_deg']
df['feat_longitude_deg']=df['longitude_deg']
df['cat_feat_iso_country']=np.where(df['iso_country'].isin(['CN','TW','JP','TH','AU','SG']),df['iso_country'],'Others')

feature= [y for y in df.columns if y[0:5]=='feat_']
for feat in feature:
    df['num_'+feat]=df[feat]
def dummified(df, col):
    df_tmp = pd.get_dummies(df[[col]])
    df_tmp.columns = ["num_feat_"+y for y in df_tmp.columns]
    return pd.concat([df, df_tmp], axis = 1)#df.join(df_tmp)
cat_feature= [y for y in df.columns if y[0:9]=='cat_feat_']
for cat_feat in cat_feature:
    df=dummified(df,cat_feat)
num_feature= [y for y in df.columns if y[0:9]=='num_feat_']

test=df['Target']
# train = df[.drop('Target',axis= 1)]
train = df[feature]

X_train, X_test, y_train, y_test = train_test_split(df[num_feature], df['Target'] ,test_size = 0.333, random_state = 0)
def train_model(classifier, feature_vector_train, label, feature_vector_valid,valid_y):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict_proba(feature_vector_valid)[:,1]

    predictions_amount =predictions * 800

    mean_squared_error = sum((predictions_amount-y_test*800)**2)/len(predictions_amount)

    mean_absolute_error = sum(abs(predictions_amount-y_test*800))/len(predictions_amount)

    return predictions,mean_squared_error,mean_absolute_error
# Linear Classifier
prob,Q2,Q1 = train_model(LogisticRegression(), X_train, y_train, X_test, y_test)
print "LR mean_squared_error: ", Q2, "LR mean_absolute_error: ", Q1
# DTree
prob,Q2,Q1 = train_model(tree.DecisionTreeClassifier(),X_train, y_train, X_test, y_test)
print "RF mean_squared_error: ", Q2, "RF mean_absolute_error: ", Q1
# RF
prob,Q2,Q1 = train_model(RandomForestClassifier(),X_train, y_train, X_test, y_test)
print "RF mean_squared_error: ", Q2, "RF mean_absolute_error: ", Q1
#XGboost
prob,Q2,Q1 = train_model(xgboost.XGBClassifier(),X_train, y_train, X_test, y_test)
print "Xgb mean_squared_error: ", Q2, "Xgb mean_absolute_error: ", Q1
#LightGBM
for c in df[cat_feature]:
    col_type = df[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        df[c] = df[c].astype('category')
X_train, X_test, y_train, y_test = train_test_split(df[feature+cat_feature], df['Target'] ,test_size = 0.333, random_state = 0)
feature = feature+cat_feature
lgb_train = lgb.Dataset(X_train, label = y_train, categorical_feature = cat_feature,free_raw_data=False )
lgb_eval = lgb.Dataset(X_test, label = y_test, categorical_feature = cat_feature, reference=lgb_train)

# params = {
#     'objective': 'binary',
#     'metric': 'binary_logloss',
#     'verbose': -1
# }

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

gbm = lgb.train(params, lgb_train ,num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print   'mean_squared_error = ', sum((y_pred-y_test*800)**2)/len(y_pred)
print   'mean_absolute_error = ', sum(abs(y_pred-y_test*800))/len(y_pred)
gbm = lgb.train(params, lgb_train ,num_boost_round=2000,
                valid_sets=lgb_eval,
            early_stopping_rounds=10)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print   'mean_squared_error = ', sum((y_pred-y_test*800)**2)/len(y_pred)
print   'mean_absolute_error = ', sum(abs(y_pred-y_test*800))/len(y_pred)

# dump model with pickle
import pickle
with open('model.pkl', 'wb') as fout:
    pickle.dump(gbm, fout)
# load model with pickle to predict
with open('model.pkl', 'rb') as fin:
    pkl_bst = pickle.load(fin)
