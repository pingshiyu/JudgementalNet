'''
Created on 6 Dec 2017

@author: pingshiyu
'''

'''
Image processing functions. Functions in this file include:
- square_resize(image, dim): resizes image to square of dimension ``dim``
- average_intensity(image): calculates the average value for each channel in the image
'''
# image loading
from scipy.ndimage import imread
import numpy as np
from PIL import Image

def average_intensity(image):
    '''
    Takes in an image and calculates the average values for each channel.
    Assumes that the channel is the last dimension.
    image: np representation of image
    
    Returns a 1xC numpy array where ``c`` is the number of channels in the image
    '''
    return image.mean(0).mean(0)

def resize_to_square(image, dim=224, background=(129, 105, 94)):
    '''
    Resizes ``image`` to a square of dimension ``dim``x``dim``. Fills in any empty space with the specified 
    ``background`` colour.
    image: numpy array representation of image
    
    Returns the resized image, (In whatever the channel of the background is)
    '''
    height, width, _ = image.shape
    resize_ratio = dim / max(width, height)
    # convert to PIL object for manipulation with PIL
    image_pil = Image.fromarray(image)
    # resize so largest dimension is size ``dim``
    new_w, new_h = int(width * resize_ratio), int(height * resize_ratio)
    image_resized = image_pil.resize((new_w, new_h), Image.ANTIALIAS)
        
    # put image onto the background
    image_square = Image.new('RGB', (dim, dim), background)
    offset = ((dim-new_w) // 2, (dim-new_h) // 2) # for pasting to the centre of the square
    image_square.paste(image_resized, offset)
    
    return np.asarray(image_square)

if __name__ == '__main__':
    image = imread('./ak.png')
    image_resized = resize_to_square(image, dim=331)
    