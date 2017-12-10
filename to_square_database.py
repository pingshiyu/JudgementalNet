'''
Created on 6 Dec 2017

@author: pingshiyu
'''
'''
Given a database (folder) of images and its features (.csv file of format [index, gender, rating, age])
Save into a new folder of structure
- database_square
    - male 
    - female
    
In minibatches, where each minibatch is a pickled file of a list of tuples, in form [image, feature]
Where each image has dimension 224x224x3, its features are [rating, age]. Its contents are [7.2, 19] for a 7.2 rated 
19 year old's face.
'''
# data reading / writing
import os, glob
import json, pickle
import logging

# create logger
logging.basicConfig(filename = './logs/to_square_database.log',
                    level = logging.DEBUG,
                    filemode = 'w+',
                    format = '%(asctime)s %(message)s')

# processing
import numpy as np
from image_processing import resize_to_square, average_intensity
from scipy.misc import imread

# specify folders to save in
root = './images/database_square/'
male_folder = root + 'male/'
female_folder = root + 'female/'

# keeping track of the male and female datasets
male_num, female_num = 0, 0
male_total_pixelvals, female_total_pixelvals = np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0])
male_minibatch, female_minibatch = [], []
male_batchnum, female_batchnum = 0, 0

for file in glob.glob('./images/raw/*.png'):
    file_name, ext = os.path.splitext(file)
    # reads in the image and the features
    logging.info(('reading file:', file_name))
    image = imread(file);
    with open(file_name + '.csv') as f:
        features = json.load(f)
        
    # if both loads then proceed with analysis
    if features:
        logging.info(('successfully read:', file_name))
        _, gender, rating, age = features
        age = int(age) # age saved as string, we convert it first
        
        # form the database entry
        feature = [rating, age]
        square_im = resize_to_square(image, dim=224)
        database_entry = (square_im, feature)
        
        # for calculating the average pixel value
        avg_pixelval = average_intensity(image)
        if gender == 'M':
            logging.info(('male found', male_num))
            male_total_pixelvals += avg_pixelval
            male_minibatch.append(database_entry); male_num += 1
            
            # save the minibatch once we have 100 entries
            if (male_num > 0) and (male_num%100 == 0):
                with open(male_folder + str(male_batchnum), 'wb') as f:
                    pickle.dump(male_minibatch, f)
                    logging.info('male minibatch number {} saved!'.format(male_batchnum))
                    male_minibatch = []; male_batchnum += 1
                    
        elif gender == 'F':
            logging.info(('female found', female_num))
            female_total_pixelvals += avg_pixelval
            female_minibatch.append(database_entry); female_num += 1
            
            # save the minibatch once we have 100 entries
            if (female_num > 0) and (female_num%100 == 0):
                with open(female_folder + str(female_batchnum), 'wb') as f:
                    pickle.dump(female_minibatch, f)
                    logging.info('female minibatch number {} saved!'.format(female_batchnum))
                    female_minibatch = []; female_batchnum += 1
        else:
            logging.warning(('neither male nor female found'))
            logging.warning((file_name, 'male num', male_num, 'female num', female_num))
    else:
        logging.warning(('reading failure - no .csv file found for:', file_name))
            
# Save remainder of the minibatch
with open(male_folder + str(male_batchnum), 'wb') as f:
    pickle.dump(male_minibatch, f)
    logging.info('male minibatch number {} saved!'.format(male_batchnum))
    male_minibatch = []; male_batchnum += 1
with open(female_folder + str(female_batchnum), 'wb') as f:
    pickle.dump(female_minibatch, f)
    logging.info('female minibatch number {} saved!'.format(female_batchnum))
    female_minibatch = []; female_batchnum += 1
    
# calculate the average pixel intensity
male_avg_pixelvals = male_total_pixelvals / male_num
female_avg_pixelvals = female_total_pixelvals / female_num
logging.info(('male average pixel values', male_avg_pixelvals))
logging.info(('female average pixel values', female_avg_pixelvals))

'''
Male average: [149, 112, 98] ~36,400 samples
Female average: [151, 116, 103] ~13,500 samples
Interesting - females on average use slightly brighter pictures and use around 5% more blue-light.
'''
