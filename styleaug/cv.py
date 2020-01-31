import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImgGreyscale():

    def run(self, img_arr):
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        return img_arr



class ImgCanny():

    #def __init__(self, low_threshold=60, high_threshold=110):
    def __init__(self, low_threshold=50, high_threshold=150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        
    def run(self, img_arr):
        return cv2.Canny(img_arr, 
                         self.low_threshold, 
                         self.high_threshold)

    

class ImgGaussianBlur():

    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size
        
    def run(self, img_arr):
        return cv2.GaussianBlur(img_arr, 
                                (self.kernel_size, self.kernel_size), 0)



class ImgCrop:
    """
    Crop an image to an area of interest. 
    """
    def __init__(self, top=0, bottom=0, left=0, right=0):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        
    def run(self, img_arr):
        #height, width, _ = img_arr.shape
        height, width = img_arr.shape
        img_arr = img_arr[self.top:height-self.bottom, 
                          self.left: width-self.right]
        return img_arr


class ImgMask:
    """
    Mask out background noise
    """
    def __init__(self, vertices):
        self.vertices = vertices

    def region_of_interest(image, vertices):
        # defining a blank mask to start with mask = np.zeros_like(image)
        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        mask = np.zeros_like(image)
        if len(image.shape) > 2: 
            channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are non-zero
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def run(self, img_arr):
        img_arr = region_of_interest(img_arr, self.vertices)
        return img_arr

#class ImgThreshold:
#
#    def __init__(self, space='hsv'):
#        self.space = space
#
#    def img_threshold(self, image):
#        hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#        sensitivity = 51
#        lower_white = np.array([0,0,255-sensitivity])
#        upper_white = np.array([255,sensitivity,255])
#        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
#        filtered_image = cv2.bitwise_and(image, image, mask=white_mask)
#        return filtered_image
#
#    def run(self, img_arr):
#        img_arr = img_threshold(img_arr)
#        return img_arr

class ImgThreshold:
    """
    Mask out the light glare in HSV space and fill the up the space with mean pixel
    """

    def __init__(self, space='hsv'):
        self.space = space

    def img_threshold(self, image):
        hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        sensitivity = 100
        lower_white = np.array([0,0,255-sensitivity])
        upper_white = np.array([255,sensitivity,255])
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
        
        mean_pixel = np.mean(image, axis=(0, 1))
        
        white_mask_inv = cv2.bitwise_not(white_mask)
        
        no_light_image = cv2.bitwise_and(image, image, mask=white_mask_inv)
        r_layer = np.full((image.shape[0], image.shape[1], 1), np.uint8(mean_pixel[0]))
        g_layer = np.full((image.shape[0], image.shape[1], 1), np.uint8(mean_pixel[1]))
        b_layer = np.full((image.shape[0], image.shape[1], 1), np.uint8(mean_pixel[2]))
        mode_image = np.concatenate((r_layer, g_layer, b_layer), axis=2)
        filled_image = cv2.bitwise_and(mode_image, mode_image, mask=white_mask)
        filtered_image = cv2.add(no_light_image, filled_image)
        
        return filtered_image

    def run(self, img_arr):
        img_arr = self.img_threshold(img_arr)
        return img_arr

class ImgStack:
    """
    Stack N previous images into a single N channel image, after converting each to grayscale.
    The most recent image is the last channel, and pushes previous images towards the front.
    """
    def __init__(self, num_channels=4):
        self.img_arr = None
        self.num_channels = num_channels

    def rgb2gray(self, rgb):
        '''
        take a numpy rgb image return a new single channel image converted to greyscale
        '''
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        
    def run(self, img_arr):
        gray = self.rgb2gray(img_arr)
        width, height = gray.shape        
        
        if self.img_arr is None:
            self.img_arr = np.zeros([width, height, self.num_channels], dtype=np.dtype('B'))

        for ch in range(self.num_channels - 1):
            self.img_arr[...,ch] = self.img_arr[...,ch+1]

        self.img_arr[...,self.num_channels - 1:] = np.reshape(gray, (width, height, 1))

        return self.img_arr

class StackedFrame:
    """
    Input to Stacked Frame Model

    Return frame dimension (1,120,160,4)
    """
    def __init__(self, num_frames=4):
        self.stacked_img = None
        self.num_frames = num_frames

    def run(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.stacked_img is None:
            self.stacked_img = np.stack(([img]*self.num_frames),axis=2)
        else:
            img = img.reshape(img.shape[0], img.shape[1], 1)
            self.stacked_img = np.append(img, self.stacked_img[:, :, :(self.num_frames-1)], axis=2)

        return self.stacked_img


class TimeSequenceFrames:
    """
    Input to LSTM

    Return frame dimension (1,7,120,160,4)
    """
    def __init__(self, num_states=7):
        self.rnn_input = None
        self.num_states = num_states # Number of States for RNNN

    def run(self, img):

        if self.rnn_input is None:
            self.rnn_input = np.stack(([img]*self.num_states),axis=0)
        else:
            img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
            self.rnn_input = np.append(img, self.rnn_input[:(self.num_states-1), :, :, :], axis=0)

        return self.rnn_input
        
        
class Pipeline():
    def __init__(self, steps):
        self.steps = steps
    
    def run(self, val):
        for step in self.steps:
            f = step['f']
            args = step['args']
            kwargs = step['kwargs']
            
            val = f(val, *args, **kwargs)
        return val


"""
Custom Code added by Felix on lane segmentation
"""

def remove_noise(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def discard_colors(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detect_edges(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def region_of_interest(image, vertices):
    # defining a blank mask to start with mask = np.zeros_like(image)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    mask = np.zeros_like(image)
    if len(image.shape) > 2: 
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are non-zero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
    
def segment_lane(img):
    """
    Transform raw image to lane segment
    """

    #img = cv2.imread(img_path)

    # Gray Scale Histogram
    #img = remove_noise(img, 5) # Gaussian Blur
    img = discard_colors(img) # Grayscale

    #ret,thresh = cv2.threshold(img,90,255,cv2.THRESH_BINARY)
    #img = cv2.bitwise_and(img, thresh)

    img = detect_edges(img, low_threshold=50, high_threshold=150) # Detect Edges with Canny Edge Detector

    ysize = img.shape[0]
    xsize = img.shape[1]
	
    vertices = np.array([[(0,48),(xsize,48),(xsize,ysize),(0,ysize)]], dtype=np.int32)
    img = region_of_interest(img, vertices)

    img = np.expand_dims(img, axis=-1)

    return img

if __name__ == '__main__':

    img = cv2.imread('augmented_data_18_01_20/_original/1000_cam-image_array_.jpg')
    img_orig = img

    #threshold = ImgThreshold()
    #img = threshold.img_threshold(img)

    #  works
    #gauss= ImgGaussianBlur()
    #img = gauss.run(img)

    # works not
    #crop = ImgCrop()
    #img = crop.run(img)

	
    # works not
    #mask = ImgMask(50)
    #img = mask.run(img)

    img = segment_lane(img)

    plt.imshow(img)
    plt.show()
