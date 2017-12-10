'''
Created on 6 Dec 2017

@author: pingshiyu
'''
'''
The class will make use of the VGG face encoder to encode image batches.

The method ``encode_batch()`` will take in a batch of images of the standard form, 
i.e. [imgnum, height, width, channels],
and will output the corresponding encoded feature array. (dimension [imgnum, 4096])
'''
import caffe

class VGG_Encoder():
    def __init__(self,
                 data_mean,
                 model_path='./vgg_face_caffe/VGG_FACE_deploy.prototxt',
                 weights_path='./vgg_face_caffe/VGG_FACE.caffemodel'):
        '''
        As the network only takes in normalised data we may supply a ``data_mean`` so that we don't have to
        calculate the mean each time. (data_mean: np array)
        If the data is already normalised then simply put np.array([0,0,0]_, for example, for an RGB image, or 
        np.array([0]) for a greyscale image.
        '''
        self.net = caffe.Net(model_path, weights_path, caffe.TEST)
        
        # create transformer for the input called 'data'
        self.transformer = caffe.io.Transformer(inputs={'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
        self.transformer.set_mean('data', data_mean)     # subtract the dataset-mean value in each channel
        self.transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        self.transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    def encode_batch(self, image_batch):
        '''
        Takes in a batch of images (of the same dimensions) and encodes the batch of images.
        The input array ``image_batch`` will have dimensions (`size of batch`, height, width, channels)
        Returns an array of shape (`size of batch`, 4096). Note 4096 is the number of features that VGG encodes.
        
        Note that only 224x224x3 images will be taken - as that is the dimension VGG was trained on.
        (To do: add option to not use the fully connected layers, this will allow any dimensions to be used)
        '''
        # gets the dimensions of this batch
        batch_size, height, width, channels = image_batch.shape
        # specify the dimension to our network.
        # channels has been moved to the first dimension as per caffe's requirements for networks
        self.net.blobs['data'].reshape(batch_size, channels, height, width)
        
        # load the batch into the network
        for i in range(batch_size):
            self.net.blobs['data'].data[i, ...] = self.transformer.preprocess('data', image_batch[i])
            
        # feed the data through the network
        self.net.forward()
        # the activation on the penultimate layer is our encoded features.
        return self.net.blobs['fc7'].data
        