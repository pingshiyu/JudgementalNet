'''
Created on 6 Dec 2017

@author: pingshiyu
'''
import glob
import pickle
from vgg_encoder import VGG_Encoder

from pandas import DataFrame
import numpy as np

# create logger
import logging
logging.basicConfig(filename = './logs/encode_faces.log',
                    level = logging.DEBUG,
                    filemode = 'w+',
                    format = '%(asctime)s %(message)s')

# locations male and female batches are stored
images_root = './images/database_square/'
male_location = images_root + 'male/'
female_location = images_root + 'female/'

# location to store the data
data_root = './data/vgg_encoded/'
male_csv_location = data_root + 'male.csv'
female_csv_location = data_root + 'female.csv'

# grab the batch of images for testing
mean_male = np.array([149, 112, 98])
mean_female = np.array([151, 116, 103])

def encode_faces_to_csv(path, mean, csvpath):
    '''
    Encode faces in the specified ``path`` with initialising the network to ``mean``
    Faces will be stored in batch files, which are Pickled lists of (image, [rating, age]) tuples
    path, csvpath: String
    mean: np array, 1x3 in shape (for VGG)
    '''
    # initialise the VGG-encoder
    vgg = VGG_Encoder(mean)
    
    # loop through files stored in path and save to .csv part by part
    for filepath in glob.glob(path + '*'):
        encoded_data = DataFrame(_vgg_encode_file(vgg, filepath))
        encoded_data.to_csv(csvpath, mode='a')
        logging.info('successfully saved batch {} to {}'.format(filepath, csvpath))
    
def _vgg_encode_file(vgg, filepath):
    '''
    Given a pickled filepath in the form that VGG-encoder accepts, encodes the faces stored in the file through the
    VGG network. The Pickled file will be a list of tuples: [(face_image, [rating, age]), ...].
    Note ``face_image`` must have dimension 224x224x3, as per the dimension VGG was trained on.
    vgg: VGG_Encoder() object
    filepath: String
    
    Returns a numpy array where the first columns are the the 4096 features; the last columns are the face's rating
    and age respectively.
    '''
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
    # unzipping the list of tuples
    images, features = list(zip(*data))
    images = np.array(images); features = np.array(features)
    batch_size = images.shape[0]
    logging.info('batch size for {} is {}'.format(filepath, batch_size))
    
    # passing all images through VGG (batching to manage RAM)
    vgg_encoding = np.vstack([vgg.encode_batch(images[i:i+25]) 
                              for i in range(0, batch_size, 25)])
    
    return np.hstack([vgg_encoding, features])
    
if __name__ == '__main__':
    encode_faces_to_csv(female_location, mean_female, female_csv_location)
    encode_faces_to_csv(male_location, mean_male, male_csv_location)
