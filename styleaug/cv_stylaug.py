from styleaug import StyleAugmentor

import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np 



class ImgStyleAug():

    def img_style(self, im, alpha=0.1):
        toTensor = ToTensor()
        toPIL = ToPILImage()
        im_torch = toTensor(im).unsqueeze(0) # 1 x 3 x 256 x 256
        im_torch = im_torch.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        # create style augmentor:
        augmentor = StyleAugmentor()

        # randomize style:
        im_restyled = augmentor.forward(im_torch, alpha)
        #return im_restyled
        
        out_img = toPIL(im_restyled.squeeze().cpu())
        return np.float32(out_img)



'''
# PyTorch Tensor <-> PIL Image transforms:
toTensor = ToTensor()
toPIL = ToPILImage()

# load image:
im = Image.open('augmented_data_18_01_20/_original/1000_cam-image_array_.jpg')
im_torch = toTensor(im).unsqueeze(0) # 1 x 3 x 256 x 256
im_torch = im_torch.to('cuda:0' if torch.cuda.is_available() else 'cpu')

# create style augmentor:
augmentor = StyleAugmentor()

# randomize style:
im_restyled = augmentor.forward(im_torch, alpha = 0.1)

# display:
plt.imshow(toPIL(im_restyled.squeeze().cpu()))
plt.show()
'''


if __name__ == '__main__':
    img = Image.open('augmented_data_18_01_20/_original/1000_cam-image_array_.jpg')

    style_aug = ImgStyleAug()
    img = style_aug.img_style(img)
    
    plt.imshow(img)
    plt.show()