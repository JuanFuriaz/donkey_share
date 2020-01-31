"""
Script to augment teaching data

Usage:
    augment.py --path=<records_dir> --out=<target_dir>

Options:
    -h --help        Show this screen.
    --path TUBPATHS   Path of the record directory
    --out MODELPATH  Path of the model file
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

def augmentation_round(in_path, out, total, name, augment_function, meta_function=None):
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
        img = Image.open('%s/%s' % (in_path, img_path))
        img = np.array(img)
        write(target, _, img, data, name, augment_function)
        count = count + 1
        print_progress(count, total, name)

    return (count, target)


def write(out, id, img, data, name, augment_function):

    new_img, new_data = augment_function(img, data)

    # Augment function can return None if this item should be skipped in the return set
    if (new_img is None or new_data is None):
        return
    record_path = '%s/record_%d.json' % (out, id)
    image_name = '%d_%s.jpg' % (id, name)
    image_path = '%s/%s' % (out, image_name)

    new_data['cam/image_array'] = image_name

    cv2.imwrite(image_path, new_img)

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


def augment(target, out = None):

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
    size, history_path = augmentation_round(init_path, out, count, 'history', augment_history, gen_history_meta)
    count = count + size
    size, flipped_path = augmentation_round(history_path, out, count, 'flipped', augment_flip)
    count = count + size
    size, bright_path = augmentation_round(flipped_path, out, count, 'bright', augment_brightness)
    count = count + size
    size, shadow_path = augmentation_round(bright_path, out, count, 'shadow', augment_shadow)
    count = count + size
    print('  -------------------------------------------------')
    print('Augmentation done. Total records %s.' % count)


def is_empty(dir):
    return not os.listdir(dir)


if __name__ == '__main__':
    args = docopt(__doc__)

    target_path = args['--path']
    out_path = args['--out']

    if out_path:
        ensure_directory(out_path)

    if out_path and target_path is not out_path and not is_empty(out_path):
        print(' Target folder "%s" must be empty' % out_path)
    else:
        augment(target_path, out_path)
