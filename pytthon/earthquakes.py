import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter   
import datetime as dt
import sklearn as sk
from sklearn import linear_model
import datetime as dt
from datetime import datetime as dtdt

# Read the data
df = pd.read_csv('../csv/database_earthquakes.csv')

# We will use only the following columns:
# 'Date', 'Time', 'Latitude', 'Longitude', 'Type', 'Depth', 'Magnitude', 'Root Mean Square', 'ID', 'Source', 'Location Source', 'Magnitude Source', 'Status'
df = df[['Date', 'Time', 'Latitude', 'Longitude', 'Type', 'Depth', 'Magnitude', 'Root Mean Square', 'ID', 'Source', 'Location Source', 'Magnitude Source', 'Status']]
df['Month'] = df['Date'].astype('string').str[0:2].astype('float')
df['Year'] = df['Date'].astype('string').str[6:10].astype('float')
df['Day'] = df['Date'].astype('string').str[3:5].astype('float')
df['Date'] = pd.to_datetime(df['Date'])

def predict_earthquakes(df):
    df['Date'] = (pd.to_datetime(df['Date']))
    
    df = df[['Date', 'Longitude', 'Latitude', 'Depth', 'Magnitude']].copy()
    for i in range(1, 4):
        # getting the indexes of all the earthquakes that happened between date_minus_i and date_minus_i+1
        df[f'date_minus_{i}indexes'] = df['Date'] - pd.Timedelta(days=i)
        df[f'date_minus_{i}indexes'] = df[f'date_minus_{i}indexes'].apply(lambda x: df[(df['Date'] >= x) 
                                                                                        & (df['Date'] <= x + pd.Timedelta(days=1))
                                                                                       ].index)

        # get the mean of the longitude, latitude, depth and magnitude of all the earthquakes that happened between date_minus_i and date_minus_i+1
        df[f'Mean_Lgtd_date_minus{i}'] = df[f'date_minus_{i}indexes'].apply(lambda x: df.loc[x, 'Longitude'].mean())
        df[f'Mean_Lttd_date_minus{i}'] = df[f'date_minus_{i}indexes'].apply(lambda x: df.loc[x, 'Latitude'].mean())
        df[f'Mean_Depth_date_minus{i}'] = df[f'date_minus_{i}indexes'].apply(lambda x: df.loc[x, 'Depth'].mean())
        df[f'Mean_Magnitude_date_minus{i}'] = df[f'date_minus_{i}indexes'].apply(lambda x: df.loc[x, 'Magnitude'].mean())

    df.to_csv('earthquakes_cleaned.csv', index=False)
    # # # # exit()
    # # # # df = pd.read_csv('earthquakes_cleaned.csv')
    # print(df)

    # preparing the data for the model
    df[
        ['Mean_Lgtd_date_minus1', 'Mean_Lgtd_date_minus2', 'Mean_Lgtd_date_minus3', 
            'Mean_Lttd_date_minus1', 'Mean_Lttd_date_minus2', 'Mean_Lttd_date_minus3', 
            'Mean_Depth_date_minus1', 'Mean_Depth_date_minus2', 'Mean_Depth_date_minus3', 
            'Mean_Magnitude_date_minus1', 'Mean_Magnitude_date_minus2', 'Mean_Magnitude_date_minus3']]=df[
        ['Mean_Lgtd_date_minus1', 'Mean_Lgtd_date_minus2', 'Mean_Lgtd_date_minus3', 
            'Mean_Lttd_date_minus1', 'Mean_Lttd_date_minus2', 'Mean_Lttd_date_minus3', 
            'Mean_Depth_date_minus1', 'Mean_Depth_date_minus2', 'Mean_Depth_date_minus3', 
            'Mean_Magnitude_date_minus1', 'Mean_Magnitude_date_minus2', 'Mean_Magnitude_date_minus3']].fillna(0)
    
    df_train = df[df['Date'] < '2016-01-01']
    df_test = df[df['Date'] >= '2016-01-01']
    df_train['Date'] = df_train['Date'].astype('string')
    df_train['Date'] = df_train['Date'].apply(lambda x: dtdt.strptime(x, '%Y-%m-%d'))
    df_train['Date'] = df_train['Date'].apply(lambda x: x.timestamp())
    df_test['Date'] = df_test['Date'].astype('string')
    df_test['Date'] = df_test['Date'].apply(lambda x: dtdt.strptime(x, '%Y-%m-%d'))
    df_test['Date'] = df_test['Date'].apply(lambda x: x.timestamp())

    X_train = df_train[['Date', 'Mean_Lgtd_date_minus1', 'Mean_Lgtd_date_minus2', 'Mean_Lgtd_date_minus3', 
                        'Mean_Lttd_date_minus1', 'Mean_Lttd_date_minus2', 'Mean_Lttd_date_minus3', 
                        'Mean_Depth_date_minus1', 'Mean_Depth_date_minus2', 'Mean_Depth_date_minus3', 
                        'Mean_Magnitude_date_minus1', 'Mean_Magnitude_date_minus2', 'Mean_Magnitude_date_minus3']]
    y_train = df_train[['Date', 'Longitude', 'Latitude', 'Depth', 'Magnitude']] 
    
    X_test = df_test[['Date', 'Mean_Lgtd_date_minus1', 'Mean_Lgtd_date_minus2', 'Mean_Lgtd_date_minus3', 
                        'Mean_Lttd_date_minus1', 'Mean_Lttd_date_minus2', 'Mean_Lttd_date_minus3', 
                        'Mean_Depth_date_minus1', 'Mean_Depth_date_minus2', 'Mean_Depth_date_minus3', 
                        'Mean_Magnitude_date_minus1', 'Mean_Magnitude_date_minus2', 'Mean_Magnitude_date_minus3']]
    y_test = df_test[['Date', 'Longitude', 'Latitude', 'Depth', 'Magnitude']]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    print(y_test)
    
    model = linear_model.LinearRegression()
    
    model.fit(X_train, y_train)
    
    print("training the model...")
    y_pred = model.predict(X_test)
    print(y_pred)
    
    df_pred = pd.DataFrame(y_pred, columns=['Date', 'Longitude', 'Latitude', 'Depth', 'Magnitude'])
    
    df_pred['Date'] = df_pred['Date'].apply(lambda x: dtdt.fromtimestamp(x).strftime('%Y-%m-%d'))
    df_pred.to_csv('earthquakes_predicted.csv', index=False)
    
    mse = sk.metrics.mean_squared_error(y_test, y_pred)

    
    print('Mean Squared Error:', mse)
    
predict_earthquakes(df)
    
exit()
