'''
Created on 21 Aug 2017

@author: pings
'''

'''
    This script will look through the databse with imgur links saved from earlier and 
    then find faces in each image - creating a new database which contains the faces
    alongside its ratings, age and gender
'''
# find faces
import face_recognition

# image processing
from PIL import Image
from scipy.misc import toimage 

# web
import requests
from io import BytesIO
import timeout_decorator

# file management
import os
import json

# for debugging / optimizing
import timeit
import pprint as pp

# general misc. use
import numpy as np

# shrink the image down to ``MAX_DIM`` faster processing. Chosen as at this size,
# the face is still visible
# 400 in general results in faces found to be ~50-150px in dimension
MAX_DIM = 400

def get_face(img_url):
    '''
        Returns images as numpy arrays of the faces found in the image link stored in
        ``img_url``
        
        Note to save CPU time the image is resized to be smaller
    '''
    raw_image = _url_to_image(img_url)
    if not raw_image: return None # check link is live
    
    image = _shrink_image(raw_image)
    if not image: return None # check image is not corrupted
    
    image_arr = _to_numpy(image)
    
    # face_locations returns the locations of the faces on the image
    start = timeit.default_timer()
    try: # some images cause format errors
        face_locations = face_recognition.face_locations(image_arr)
    except:
        face_locations = []
    end = timeit.default_timer()
    print('time taken: %.3f' % (end-start))
    
    if face_locations:
        # if faces were found, then
        t, r, b, l = _find_largest_face(face_locations)
        try:
            found_face = image_arr[t:b, l:r, :]
        except: # array error
            return None
        
        print('face & image size:', found_face.shape, image_arr.shape)

        return toimage(found_face)
    else:
        # no faces found
        return None
    
@timeout_decorator.timeout(10)
def _url_to_image(url):
    '''
        Reads in the image from the url given. Returns an image object.
        Returns None otherwise
    '''
    # request.get(url).content returns in bytes, and BytesIO allows it to be read by
    # Image.open
    try :
        print('Reading link:', url)
        image = Image.open(BytesIO(requests.get(url).content))
        print('Image read!')
        return image
    except: # no image found
        print('No image found on', url)
        return None
    
def _to_numpy(image):
    '''
        Takes in an ``image`` and turns it into a numpy array.
    '''
    image_arr = np.asarray(image)
    image_arr.setflags(write = True) # make image writable for face-recognition
    return image_arr

def _shrink_image(image):
    '''
        Takes in an image object and shrinks it, maintaining its aspect ratio. The
        shrink is guaranteed to happen, with the largest dimension capped to
        ``MAX_DIM``
        
        Returns the resized image.
    '''
    w, h = image.size
    shrink_factor = max(w/MAX_DIM, h/MAX_DIM)
    if shrink_factor > 1:
        try:
            return image.resize((int(w/shrink_factor), int(h/shrink_factor)))
        except: # bad image - (GIFs, corrupted etc)
            print('Image is corrupted')
            return None
    else:
        return image

def _find_largest_face(face_locations):
    '''
        Given a list of face locations, find the largest face present within the list
        ``face_locations`` is a list of face locations, represented in tuples of css
        format, which is (top, right, bottom, left)
        
        Returns the tuple which represents the location of the largest face
    '''
    # here 'size' is defined by its area, i.e. (bottom-top)*(right-left)
    size_fn = lambda trbl: (trbl[2]-trbl[0])*(trbl[1]-trbl[3])
    face_sizes = np.array(list(map(size_fn, face_locations)))
    return face_locations[np.argmax(face_sizes)]

def save_post(post):
    '''
        Input: a ``post`` (structure [links_list, gender, rating, age])
        Grabs the links and saves information locally
    '''
    img_links = post[0]
    gender = post[1]; rating = post[2]; age = post[3]
    
    # go through the image links
    if not img_links: # make sure there are links to go through
        print('No links found on this post.')
        return None
    for im in img_links:
        face = get_face(im)
        if face:
            # face is found in the image link
            save_name = save_dir + str(settings['img_num'])
            # save the image
            face.save(save_name + '.png')
            # save associated information
            img_info = [settings['img_num'], gender, rating, age]
            with open(save_name + '.csv', 'w+') as f:
                json.dump(img_info, f)
            
            settings['img_num'] += 1

def save_data():
    '''
        Saves dictionary ``settings`` to disk
    '''
    with open('./config/progress.config', 'w+') as f:
        json.dump(settings, f)
        
    print('DATA SAVED!')
    print(settings)

if __name__ == '__main__':
    # reads links from ./data/image_links/ (json files); save the images to ./image/raw
    # json files are a list of lists, with each element corresponding to a 'post'. Each 
    # post has a few links, and the first element represents the links.
    # to store relational data, the data is indexed by ``filenum``
    save_dir = './images/raw2/'
    
    # initial settings and database
    settings = {'img_num': 0,
                'file_path': 'images-10',
                'progress_through_file': 0}
    
    # load the settings and database saved previously
    with open('./config/progress.config') as f:
        settings = json.load(f)
        
    # go through the links; save the images to file
    try:            
        for root, dirs, files in os.walk('./data/image_links'):
            # sort files so we always go through the same order of files
            files.sort()
            
            # start from last saved checkpoint
            checkpoint = settings['file_path']
            if checkpoint in files:
                # trim the contents before checkpoint
                files = files[files.index(checkpoint):] 
            
            for file in files:
                # update the settings for the most recent accessed file
                settings['file_path'] = file
                
                # file's full path
                fpath = os.path.join(root, file)
                with open(fpath) as f:
                    link_list = json.load(f)
                    
                    # load saved progress:
                    progress = settings['progress_through_file']
                    link_list = link_list[progress:]
                    
                    for i, post in enumerate(link_list):
                        print('Processing post {}'.format(i))
                        save_post(post)
                        settings['progress_through_file'] += 1
                    
                # done with the current file - reset progress to 0
                settings['progress_through_file'] = 0
                save_data()
                
    except Exception as e:
        print('Ran into exception', e)
    finally: # save data if we run into any errors
        save_data()