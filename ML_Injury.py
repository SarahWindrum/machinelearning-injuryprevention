# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 08:55:04 2022

@author: SarahWindrum
"""
# Using data to predict injury / non-injury

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import & Prepare Dataset
injury_dataset = pd.read_csv('Injury_Detail.csv')
injury_dataset['Days_Missed'] = pd.to_numeric(injury_dataset['Days_Missed'])

# Clean Columns 
print(injury_dataset.head)
print(injury_dataset.columns)
print(injury_dataset.Injury_Site.unique())
print(injury_dataset.Injury_Side.unique())
print(injury_dataset.Injury_Type.unique())
print(injury_dataset.When.unique())
print(injury_dataset.Mechanism.unique()) #Freehand

# 129 rows x 15 columns
# Position (categorical F or B)
# Player Number (discrete numerical)
# Week No (discrete numerical)
# Injury date (44---)
# Injury description (freehand -> word association)
# Mechanism behind injury (freehand -> word association)
# Injury Site (categorical)
# Injury Type (categorical)
# Injury Side (categorical)
# Contact (Boolean Y/N) 
# When (training, game, non-rugby)
# Return date (44---)
# Days missed (continuous numerical)
# Preventable (Y/N)
# Month (categorical)
# Season is 2021/2022

# Null values in Side, Contact, & Preventable (not known?)
injury_dataset_2 = injury_dataset.fillna('Not_Known', inplace=False)

# Plot categorical value counts from 129 Injury Records
injury_dataset_2['Injury_Site'].value_counts()[injury_dataset_2.Injury_Site.unique()].plot(kind='bar')
injury_dataset_2['Injury_Type'].value_counts()[injury_dataset_2.Injury_Type.unique()].plot(kind='bar')
injury_dataset_2['Injury_Side'].value_counts()[injury_dataset_2.Injury_Side.unique()].plot(kind='bar') # Null
injury_dataset_2['Contact '].value_counts()[injury_dataset_2['Contact '].unique()].plot(kind='bar') # Null
injury_dataset_2['When'].value_counts()[injury_dataset_2.When.unique()].plot(kind='bar')
injury_dataset_2['Month'].value_counts()[injury_dataset_2.Month.unique()].plot(kind='bar')
injury_dataset_2['Position'].value_counts()[injury_dataset_2.Position.unique()].plot(kind='bar')
injury_dataset_2['Week_No'].value_counts()[injury_dataset_2.Week_No.unique()].plot(kind='bar')
injury_dataset_2['Player'].value_counts().plot(kind='bar')
injury_dataset_2['Preventable'].value_counts()[injury_dataset_2.Preventable.unique()].plot(kind='bar') # Null 

# Numerical Mean & Std 
print(injury_dataset.Days_Missed.describe())
print(injury_dataset.Days_Missed.mean())

# Mean 25 days missed per injury -> high variance -> Std 37 
# Median / 50% -> 12 days missed or less
plt.boxplot(injury_dataset.Days_Missed) # 300 days skewing average

# What is average days missed per player injured? 
days_per_player_subset = injury_dataset_2[['Player', 'Days_Missed']]
average_days = days_per_player_subset.groupby('Player').mean()
print(average_days)
plt.figure()
average_days.plot(kind="bar")
plt.xlabel('Player')
plt.ylabel('Days Missed')
plt.title('Days Missed per Player 2021/22 Season')
plt.legend(['Days Missed'], loc='upper left')
plt.show()

# What about days missed per player per injury? e.g. Player 63 had 9 separate injuries
injury_counts = injury_dataset_2['Player'].value_counts()
print(injury_counts)
print(injury_counts.mean())
print(average_days.mean())
# Mean of 3 injuries and 33 days missed per player = 11 days missed per player per injury 

# Add in mean & median line to identify high risk month and activity 
plt.scatter('Month', 'Days_Missed', data=injury_dataset_2, cmap="Blues", alpha=0.5)
plt.axhline(y=np.nanmean(injury_dataset_2.Days_Missed), color='red', linestyle='--', label='Mean')
plt.axhline(y=np.median(injury_dataset_2.Days_Missed), color='yellow', linestyle='--', label='Median')
plt.xlabel('Month')
plt.ylabel('Days Missed')
plt.title('Days Missed per Month 2021/22 Season')
plt.legend(loc='upper left', ncol = 2)
plt.show()

plt.scatter('When', 'Days_Missed', data=injury_dataset_2, marker='x', cmap="Blues", alpha=0.5)
plt.axhline(y=np.nanmean(injury_dataset_2.Days_Missed), color='red', linestyle='--', label='Mean')
plt.axhline(y=np.median(injury_dataset_2.Days_Missed), color='yellow', linestyle='--', label='Median')
plt.xlabel('Activity')
plt.ylabel('Days Missed')
plt.title('Days Missed per Injury Activity 2021/22 Season')
plt.legend(loc='center right')
plt.show()

# Are more days missed with preventable or non-preventable injuries? 
plt.scatter('Preventable', 'Days_Missed', data=injury_dataset_2, cmap="reds", alpha=0.5)
plt.axhline(y=np.nanmean(injury_dataset_2.Days_Missed), color='red', linestyle='--', label='Mean')
plt.axhline(y=np.median(injury_dataset_2.Days_Missed), color='yellow', linestyle='--', label='Median')
plt.xlabel('Identified as Preventable or Non-Preventable')
plt.ylabel('Days Missed')
plt.title('Days Missed per Preventable or Non-Preventable Injury 2021/22 Season')
plt.legend(loc='upper right', ncol = 3)
plt.show()

# Training Data -> time series 
training_data = pd.read_csv('Training_Data.csv')
print(training_data.head)
print(training_data.columns)
print(training_data.Player.unique())

# 50,545 rows x 34 columns 
# Date
# Week
# Training Day
# Player
# Forward or Back
# Position
# Match Mins
# Activity
# Duration
# Peak Speed
# Peak Speed % of max
# Distance
# Metres Running Fast
# Running Category
# HMLD Metres (High Metabolic Load Distance)
# HMLD category
# HMLD Mins
# High Speed >70% Distance
# VH Speed >80% Distance
# VH Speed
# Sprint >90% Distance
# Effort >80% max speed
# Effort >90% max speed
# Peak Acceleration metres
# Peak Acceleration % of max
# Heavy Acceleration Distance
# Heavy Deceleration Distance
# Heavy Accel & Decel Metres
# Acceleration Efforts Fast
# Peak Heart Rate
# Peak Heart Rate % 
# Average Heart Rate
# Mins >85% max Heart Rate
# Heart Rate Exertion

# Need to create new subset for Players experienced injury
print(injury_dataset_2.Player.unique())
injured_players = training_data[training_data["Player"].isin([69,57,15,19,55,63,17,53,12,62,4,56,7,25,44,51,30,3, 
10,24,5,72,37,28,48,16,64,46,59,2,34,18,41,6,35,43,8,70,65,54,9,22,29])]
print(injured_players.shape)

# 43,265 records for injured players - need to refine by date before injury
injury_dates = injury_dataset_2[['Injury_Date', 'Player']]
injury_dates_count = injury_dates.groupby('Player').count()

# Use Player 63 with 9 separate injuries -> create a if loop to run through Dataframe? 
Player_63_subset = injury_dataset_2[injury_dataset_2['Player']==63]
print(Player_63_subset.Injury_Date)
Player_63_Training_subset = injured_players[injured_players['Player']==63]
Player_63_injury_1 = Player_63_Training_subset.query('Date<=44400')
Player_63_injury_2 = Player_63_Training_subset.query('Date<=44451 & Date>44400')
Player_63_injury_3 = Player_63_Training_subset.query('Date<=44464 & Date>44451')
Player_63_injury_4 = Player_63_Training_subset.query('Date<=44485 & Date>44464')
Player_63_injury_5 = Player_63_Training_subset.query('Date<=44510 & Date>44485')
Player_63_injury_6 = Player_63_Training_subset.query('Date<=44526 & Date>44510')
# 4 records with the same date. Multiple injuries but all with 13 days missed

# Comparing 'heavy' training with Days Missed for Player 63 injuries
# Need to get Days Missed for each injury entry 
plt.scatter(
    x=Player_63_injury_1.Heavy_Accel_Distance,
    y=Player_63_injury_1.Heavy_Decel_Distance,
    s=Player_63_subset.Days_Missed[5]*5,
    alpha=0.5,
    label = "Injury 1"

)
plt.scatter(
    x=Player_63_injury_2.Heavy_Accel_Distance,
    y=Player_63_injury_2.Heavy_Decel_Distance,
    s=Player_63_subset.Days_Missed[18]*5,
    marker="d",
    alpha=0.5,
    label = "Injury 2"
)
plt.scatter(
    x=Player_63_injury_3.Heavy_Accel_Distance,
    y=Player_63_injury_3.Heavy_Decel_Distance,
    s=Player_63_subset.Days_Missed[25]*5,
    alpha=0.5,
    marker='v',
    label = "Injury 3"
)
plt.scatter(
    x=Player_63_injury_4.Heavy_Accel_Distance,
    y=Player_63_injury_4.Heavy_Decel_Distance,
    s=Player_63_subset.Days_Missed[39]*5,
    marker="^",
    alpha=0.5,
    label = "Injury 4"
)
plt.scatter(
    x=Player_63_injury_5.Heavy_Accel_Distance,
    y=Player_63_injury_5.Heavy_Decel_Distance,
    s=Player_63_subset.Days_Missed[52]*5,
    alpha=0.5,
    marker='>',
    label = "Injury 5"
)
plt.scatter(
    x=Player_63_injury_6.Heavy_Accel_Distance,
    y=Player_63_injury_6.Heavy_Decel_Distance,
    s=Player_63_subset.Days_Missed[58]*5,
    marker="<",
    alpha=0.5,
    label = "Injury 6"
)
plt.axhline(y=np.nanmean(training_data.Heavy_Decel_Distance), color='red', linestyle='--')
plt.axvline(x=np.nanmean(training_data.Heavy_Accel_Distance), color='yellow', linestyle='--')
plt.title("Scatterplot showing heavy acceleration and deceleration distance before injury")
plt.legend(loc='upper left')
plt.xlabel("Acceleration Distance")
plt.ylabel("Deceleration Distance")
plt.text(
    580,
    25,
    "Size = days missed",
)

plt.show()

# Size doesn't really show difference in days missed length
# Injuries 4,5 & 6 could be result of overtraining  
# Compared with mean for both measurements for all players 

# Compare with HMLD 
plt.scatter(
    x=Player_63_injury_1.HMLD_metres,
    y=Player_63_injury_1.Heavy_Accel_Decel_metres,
    s=Player_63_subset.Days_Missed[5]*5,
    alpha=0.5,
    label = "Injury 1"

)
plt.scatter(
    x=Player_63_injury_2.HMLD_metres,
    y=Player_63_injury_2.Heavy_Accel_Decel_metres,
    s=Player_63_subset.Days_Missed[18]*5,
    marker="d",
    alpha=0.5,
    label = "Injury 2"
)
plt.scatter(
    x=Player_63_injury_3.HMLD_metres,
    y=Player_63_injury_3.Heavy_Accel_Decel_metres,
    s=Player_63_subset.Days_Missed[25]*5,
    alpha=0.5,
    marker='v',
    label = "Injury 3"
)
plt.scatter(
    x=Player_63_injury_4.HMLD_metres,
    y=Player_63_injury_4.Heavy_Accel_Decel_metres,
    s=Player_63_subset.Days_Missed[39]*5,
    marker="^",
    alpha=0.5,
    label = "Injury 4"
)
plt.scatter(
    x=Player_63_injury_5.HMLD_metres,
    y=Player_63_injury_5.Heavy_Accel_Decel_metres,
    s=Player_63_subset.Days_Missed[52]*5,
    alpha=0.5,
    marker='>',
    label = "Injury 5"
)
plt.scatter(
    x=Player_63_injury_6.HMLD_metres,
    y=Player_63_injury_6.Heavy_Accel_Decel_metres,
    s=Player_63_subset.Days_Missed[58]*5,
    marker="<",
    alpha=0.5,
    label = "Injury 6"
)

plt.axhline(y=np.nanmean(training_data.Heavy_Accel_Decel_metres), color='red', linestyle='--')
plt.axvline(x=np.nanmean(training_data.HMLD_metres), color='yellow', linestyle='--')
plt.title("Scatterplot showing HMLD metres and Heavy metres before injury")
plt.legend(loc='upper left')
plt.xlabel("HMLD metres")
plt.ylabel("Heavy metres")
plt.text(
    1250,
    25,
    "Size = days missed",
)

plt.show()


# What can machine learning offer? Predict days missed to show us most significant variables?

# Target -> injury v non-injury
X = np.array(dataset.drop(['Target', '??'],1))
print(X)
np.where(np.isnan(X))
y = np.array(dataset['Target'])
print(X.shape)
print(y.shape)

# Training / Test Data split -> 50%-85%

import sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35,
random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Injury v Non-Injury -> do we need over-sampling? what is the desired balance?
# https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
# https://medium.com/grabngoinfo/four-oversampling-and-under-sampling-methods-for-imbalanced-classification-using-python-7304aedf9037

import imblearn
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter

# What is oversampling strategy? Below is half as many minority as majority
oversample = RandomOverSampler(sampling_strategy=0.5)

X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)
print(sorted(Counter(y_train_over).items()))

# Feature Scaling -> split first then scale. What does review say?
# Robust versus Standard -> outliers
from sklearn import preprocessing
# Scale X_train
scaler = preprocessing.RobustScaler().fit(X_train_over)
print(scaler)
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled)
# Scale X_test
scaler = preprocessing.RobustScaler().fit(X_test)
print(scaler)
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled)
# y is class label so doesn't need scaling

# Feature Selection / Dimensionality Reduction
# PCA -> not ideal for this? what are the variables?
from sklearn.decomposition import PCA
pca = PCA()
X_train_scaled = pca.fit_transform(X_train_scaled)
X_test_scaled = pca.transform(X_test_scaled)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
print(sum(explained_variance))

# Screeplot -> to help choose number of components
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.grid()
plt.show()

pca = PCA(n_components=25)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Evaluation -> is weighting of FP and FN equal?
# Confusion Matrix / F1 Score
# FP better than FN
# High sensitivity 



