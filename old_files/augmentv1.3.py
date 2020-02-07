"""
Script to augment teaching data
version 1.3
-adding parser and using args
Usage:
    augment.py --path=<records_dir> --out=<target_dir> [--method=all|classic|gaussian|threshold|canny|style_aug] --gpu_enabled=1

Options:
    TODO: change names
    -h --help        Show this screen.
    --path TUBPATHS   Path of the record directory
    --out MODELPATH  Path of the model file


Todo:
- add parser
- change function's input to args
- dont copy origin images
- create folder with tub name: example: tub_10-22-12_candy
- FUTURE have both options of creating folder OR giving array
- multiple folder arrays with subcomand, example of options
        -all: do everything
        -trad: traditional augmentation
        -candy: cady transd
        ...
"""

from docopt import docopt
from PIL import Image

import numpy as np

import cv2

import glob
import json
import re
import copy
import shutil
import os
from collections import deque
from os import sys

# user defined imports
import styleaug.cv
from styleaug.cv import ImgGaussianBlur
from styleaug.cv import ImgThreshold
from styleaug.cv import ImgCanny
from styleaug import neural_style

from styleaug.cv_stylaug import ImgStyleAug


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def print_progress(count, total, name='', bar_length=20):
    if count % 10 == 0 or count == total:
        percent = 100 * (count / total)
        filled_length = int(bar_length * count / total)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        print('\r  %s\t |%s| %.1f%% %s' % (name, bar, percent, 'done'), end='\r')
    if count == total:
        print()

def initialize_records(records, path, out, target_dir):
    sum = 0

    if path is not out:
        target_path = '%s/%s' % (out, target_dir)
        ensure_directory(target_path)
        shutil.copy('%s/meta.json' % path, target_path)
    else:
        target_path = path

    for _, record in records:
        sum = sum + 1
        if path is not out:
            with open(record, 'r') as record_file:
                data = json.load(record_file)
                img_path = data['cam/image_array']
            shutil.copy(record, target_path)
            shutil.copy('%s/%s' % (path, img_path), target_path)

    return (sum, target_path)

# TODO: better place for global stuff
round_number = 0

def augmentation_round(in_path, out, total, name, augment_function, meta_function=None, args=None):
    global round_number
    round_number += 1
    target = '%s/%s_%s' % (out, round_number, name)
    records = glob.glob('%s/record*.json' % in_path)
    records = ((int(re.search('.+_(\d+).json', path).group(1)), path) for path in records)

    ensure_directory(target)
    if (meta_function is not None):
        with open('%s/meta.json' % in_path, 'r') as meta_file:
            raw_data = json.load(meta_file)
            new_data = meta_function(raw_data)
            with open('%s/meta.json' % target, 'w') as outfile:
                json.dump(new_data, outfile)
    else:
        shutil.copy('%s/meta.json' % in_path, target)

    count = 0

    for _, record in sorted(records):
        with open(record, 'r') as record_file:
            data = json.load(record_file)
            img_path = data['cam/image_array']
        if not args:
            print("In Img_path: " + '%s/%s' % (in_path, img_path))
            img = Image.open('%s/%s' % (in_path, img_path))
            img = np.array(img)
        else:
            img = '%s/%s' % (in_path, img_path)
        
        write(target, _, img, data, name, augment_function, args)
        count = count + 1
        print_progress(count, total, name)

    return (count, target)


def write(out, id, img, data, name, augment_function, args =None):

    if not args:
        new_img, new_data = augment_function(img, data)
        if (new_img is None or new_data is None):
             return
    else:
        new_data = data
        new_img = img
    # Augment function can return None if this item should be skipped in the return set

    record_path = '%s/record_%d.json' % (out, id)
    image_name = '%d_%s.jpg' % (id, name)
    image_path = '%s/%s' % (out, image_name)

    new_data['cam/image_array'] = image_name
    if not args:
        cv2.imwrite(image_path, new_img)
    else:
        augment_function(img, image_path, args)

    with open(record_path, 'w') as outfile:
        json.dump(new_data, outfile)

# TODO: better place for global stuff
HISTORY_LENGTH = 50
current_history_length = 0
history_buffer = {}

def gen_history_meta(old_meta):
    meta_with_history = copy.deepcopy(old_meta)
    for input_key in old_meta['inputs']:
        meta_with_history['inputs'].append('history/%s' % input_key)
    for type_key in old_meta['types']:
        meta_with_history['types'].append('%s_array' % type_key)
    return meta_with_history

def augment_history(img, data):
    global current_history_length
    global history_buffer
    data_with_history = copy.deepcopy(data)
    data_keys = data.keys()
    for key in data_keys:
        if (key not in history_buffer):
            history_buffer[key] = deque(maxlen=HISTORY_LENGTH)
        history_buffer[key].append(data[key])
    current_history_length += 1
    if (current_history_length < HISTORY_LENGTH):
        return (None, None)

    # TODO: this includes also the current value
    for key in data_keys:
        history_key = 'history/%s' % key
        data_with_history[history_key] = list(history_buffer[key])

    return (img, data_with_history)

def aug_flip(inputs, outputs):
    img = inputs[0]
    img = cv2.flip(img, 1)

    augmented_outputs = [-outputs[0], outputs[1]]
    augmented_inputs = copy.deepcopy(inputs)
    augmented_inputs[0] = img
    return augmented_inputs, augmented_outputs

def aug_brightness(inputs, outputs):
    img = inputs[0]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img = np.array(img, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    img[:,:,2] = img[:,:,2]*random_bright
    img[:,:,2][img[:,:,2]>255]  = 255
    img = np.array(img, dtype = np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    augmented_inputs = copy.deepcopy(inputs)
    augmented_inputs[0] = img
    return augmented_inputs, outputs

def aug_shadow(inputs, outputs):
    img = inputs[0]

    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][0]
    Y_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][1]

    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    # random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    img = cv2.cvtColor(image_hls, cv2.COLOR_HLS2BGR)

    augmented_inputs = copy.deepcopy(inputs)
    augmented_inputs[0] = img
    return augmented_inputs, outputs

def aug_shadow2(inputs, outputs):
    img = cv2.cvtColor(inputs[0],cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    shadow_mask = image_hls[:, :, 1] * 0
    X_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][0]
    Y_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][1]

    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    # random_bright = .25+.7*np.random.uniform()
    #if np.random.randint(2) == 1:
    random_bright = .4
    random_bright2 = .2
    cond = shadow_mask == np.random.randint(2)
    image_hls[:, :, 0][cond] = image_hls[:, :, 0][cond] * random_bright
    image_hls[:, :, 1][cond] = image_hls[:, :, 1][cond] * random_bright
    image_hls[:, :, 2][cond] = image_hls[:, :, 2][cond] * random_bright2
    img = cv2.cvtColor(image_hls, cv2.COLOR_HLS2BGR)

    augmented_inputs = copy.deepcopy(inputs)
    augmented_inputs[0] = img
    return augmented_inputs, outputs

def augment_flip(img, data):
    data = copy.deepcopy(data)
    img = cv2.flip(img, 1)

    flip_keys = [
        'user/angle',
        #'acceleration/y',
        #'gyro/y',
        'history/user/angle',
        #'history/acceleration/y',
        #'history/gyro/y'
    ]

    for key in flip_keys:
        if (isinstance(data[key], list)):
            flipped_list = list(map(lambda value: 0 - value, data[key]))
            data[key] = flipped_list
        else:
            data[key] = 0 - data[key]

    # Sonar values have to be switched
    #old_sonar_left = data['sonar/left']
    #data['sonar/left'] = data['sonar/right']
    #data['sonar/right'] = old_sonar_left

    #old_sonar_history_left = data['history/sonar/left']
    #data['history/sonar/left'] = data['history/sonar/right']
    #data['history/sonar/right'] = old_sonar_history_left

    return (img, data)

def augment_brightness(img, data):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img = np.array(img, dtype = np.float64)
    random_bright = .2+np.random.uniform()
    img[:,:,2] = img[:,:,2]*random_bright
    img[:,:,2][img[:,:,2]>255]  = 255
    img = np.array(img, dtype = np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

    return (img, data)

def augment_shadow(img, data):
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][0]
    Y_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][1]

    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    # random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    img = cv2.cvtColor(image_hls, cv2.COLOR_HLS2BGR)
    return (img, data)


#
# customer defined functions
#

# gaussian blur
def augment_gaussian_blur(img, data):
    gauss= ImgGaussianBlur()
    img = gauss.run(img)
    return (img, data)


# threshold
def augment_threshold(img, data):
    threshold = ImgThreshold()
    img = threshold.img_threshold(img)
    return (img, data)


#
#augment_style_alpha = 1
#augment_style_gpu_enabled = 0
# style augmentation
def augment_style(img, data):
    style = ImgStyleAug()
    img = style.img_style(img,augment_style_alpha)
    return (img, data)

def augment_style_neural(inImg, outImg, args):
    args.export_onnx = False
    args.output_image = outImg
    args.content_image= inImg
    args.content_scale = 1
    neural_style.stylize(args)


# canny
def augment_canny(img, data):
    canny = ImgCanny()
    img = canny.run(img)
    return (img, data)

def augment(target, out = None, method_args='all', args = None):

    print('Start augmentation')

    records = glob.glob('%s/record*.json' % target)
    records = ((int(re.search('.+_(\d+).json', path).group(1)), path) for path in records)

    # Directories starting with underscore are skipped in training. Originals have no history augmented so have to be skipped
    size, init_path = initialize_records(records, target, out, "_original")

    count = size

    if not out:
        out = target
    print('  Augmenting %d records from "%s". Target folder: "%s"' % (count, target, out))
    if target is not out:
        print('  Original files copies to "%s"', init_path)
    print('  -------------------------------------------------')
    
    if 'all' in method_args or 'classic' in method_args:
        size, history_path = augmentation_round(init_path, out, count, 'history', augment_history, gen_history_meta)
        count = count + size
        size, flipped_path = augmentation_round(history_path, out, count, 'flipped', augment_flip)
        count = count + size
        size, bright_path = augmentation_round(flipped_path, out, count, 'bright', augment_brightness)
        count = count + size
        size, shadow_path = augmentation_round(bright_path, out, count, 'shadow', augment_shadow)
        count = count + size
    
    
    if 'all' in method_args or 'gaussian' in method_args:
        size, gaussian_path = augmentation_round(init_path, out, count, 'gaussian_blur', augment_gaussian_blur)
        count = count + size
       
    if 'all' in method_args or 'threshold' in method_args:
        size, threshold = augmentation_round(init_path, out, count, 'threshold', augment_threshold)
        count = count + size
        
    if 'all' in method_args or 'canny' in method_args:
        size, canny = augmentation_round(init_path, out, count, 'canny', augment_canny)
        count = count + size
    
    if 'all' in method_args or 'style_aug' in method_args:
        global augment_style_alpha
    #    global augment_style_gpu_enabled
   #     if gpu_enabled:
  #          augment_style_gpu_enabled = 1
        
        augment_style_alpha = args.style_alpha
        size, style = augmentation_round(init_path, out, count, "".join(['style_aug_',str(augment_style_alpha) ]) , augment_style)
        count = count + size

    if 'all' in method_args or 'style_neural' in method_args:

        size, style = augmentation_round(init_path, out, count, "".join(['style_neural_']),
                                         augment_style_neural, args=args)
        count = count + size



    print('  -------------------------------------------------')
    print('Augmentation done. Total records %s.' % count)


def is_empty(dir):
    return not os.listdir(dir)


if __name__ == '__main__':
    """
    Example of use
    python augment.py -t mycar/data/tub_2_20-02-02 -o  mycar/data/test -m styleaug/saved_models/candy.pth -a style_neural 
    
    """
    import argparse
    parser = argparse.ArgumentParser(description="Parser for creating augmented data")

    parser.add_argument('-t', '--target-path', help='Images dir', default= 'mycar/data/*.jpg', type=str, required=True,)
    parser.add_argument('-o', '--out-path', help='Path to save generated stylized images', default='mycar/data/', type=str, required=True,)
    parser.add_argument('-m', '--model', help='Path to a model',
                        default='styleaug/saved_models/mosaic.pth', type=str)
    parser.add_argument('-a', '--method-args', help='Method to be choosen all, traditional, etc..',
                        default='all', type=str)
    parser.add_argument("-s", "--style-alpha",  default=1, type=float,
                        help="set it to 1 for running on GPU, 0 for CPU")


    parser.add_argument("-c", "--cuda", type=int, default=False,
                        help="set it to 1 for running on GPU, 0 for CPU")

    args = parser.parse_args()
    ensure_directory(args.out_path)

    #TODO: cuda beeing load

    if args.out_path and args.target_path is not args.out_path and not is_empty(args.out_path):
        print(' Target folder "%s" must be empty' % args.out_path)
    else:
        # TODO: add args
        augment(args.target_path, args.out_path, args.method_args, args)
