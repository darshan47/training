# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:05:40 2019

@author: Darshan.Shah
"""
import pandas as pd
from elasticsearch import Elasticsearch
import numpy as np

## Connection to elasticsearch
es = Elasticsearch(['ec2-3-19-110-7.us-east-2.compute.amazonaws.com:9200'])
doc ='{    "size" : 10,"_source": ["system.load.1", "@timestamp"],"query":{"bool":{"must":{"exists":{"field":"system.load.1"}}} }}'
res= es.search(index="index-metrics-sinclair-dev-2019.07.13", body=doc)

print(res)

## To dataframe
from pandasticsearch import Select
pandas_df = Select.from_dict(res).to_pandas()
print(pandas_df)

## Formatting data
df2 = pandas_df[['@timestamp','system.load.1']]
df3=df2.rename(columns={"@timestamp": "time"})
df3['time']=pd.to_datetime(df3.time)   
df3['second']=df3.time.dt.second
df3['minute']=df3.time.dt.minute
df3['hour']=df3.time.dt.hour
df3=df3.drop(['time'], axis=1)
labels = np.array(df3['system.load.1'])
df3= df3.drop('system.load.1', axis = 1)
df3_list = list(df3.columns)
df3 = np.array(df3)

## Creating Test & Train data
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(df3, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

## Running algorithm
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 10, random_state = 42)
rf.fit(train_features, train_labels);
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)

## Caclulating Accuracy
print("Test_label {}".format(test_labels))
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')