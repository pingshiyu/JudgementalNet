'''
Created on 7 Dec 2017

@author: pingshiyu
'''
'''
Uses Linear Regression on the VGG-encoded faces dataset, and evaluates the trained model.
'''
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv

# location of data stored
data_root = './data/vgg_encoded/'
male_csv = data_root + 'male.csv'
female_csv = data_root + 'female.csv'

# reading in the CSV file as an numpy array
female_data = read_csv(female_csv).values
X_female, rating_female, age_female = female_data[:, :-2], female_data[:, -2], female_data[:, -1]

# perform a train/test split. 90% used for training and 10% for testing
X_female_train, X_female_test, rating_female_train, rating_female_test = train_test_split(X_female, rating_female,
                                                                                          test_size=0.1)

# first we plot the rating distribution of the ratings, as well as the age
plt.hist(rating_female_train[:1000], bins=10, range=(0,10)); plt.show(); plt.clf()
plt.hist(age_female[:1000], bins=80, range=(0,80)); plt.show(); plt.clf()

