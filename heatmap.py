import matplotlib.pyplot as plt
from matplotlib import animation

try:
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Convolution2D
    from keras import backend as K
    plt.rcParams['animation.ffmpeg_path'] = '/home/jm/bin/ffmpeg' # explicit path for finding ffmpeg in my computer
except ImportError:
    from tensorflow.python.keras.layers import Input
    from tensorflow.python.keras.models import Model, load_model
    from tensorflow.python.keras.layers import Convolution2D
    from tensorflow.python.keras import backend as K


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disabling TF warnings
import tensorflow as tf
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import animation
from glob import glob
#from keras import backend as K
import argparse
#plt.rcParams['animation.ffmpeg_path'] = '/home/jm/bin/ffmpeg' # explicit path for finding ffmpeg in my computer


def compute_visualisation_mask(img, functor, layers_kernels, layers_strides):
    activations = functor([np.array([img])])
    upscaled_activation = np.ones((3, 6))
    for layer in [4, 3, 2, 1, 0]:
        averaged_activation = np.mean(activations[layer], axis=3).squeeze(axis=0) * upscaled_activation
        if layer > 0:
            output_shape = (activations[layer - 1].shape[1], activations[layer - 1].shape[2])
        else:
            output_shape = (120, 160)
        x = tf.constant(
            np.reshape(averaged_activation, (1,averaged_activation.shape[0],averaged_activation.shape[1],1)),
            tf.float32
        )
        conv = tf.nn.conv2d_transpose(
            x, layers_kernels[layer],
            output_shape=(1,output_shape[0],output_shape[1], 1),
            strides=layers_strides[layer],
            padding='VALID'
        )
        with tf.Session() as session:
            result = session.run(conv)
        upscaled_activation = np.reshape(result, output_shape)
    final_visualisation_mask = upscaled_activation
    return (final_visualisation_mask - np.min(final_visualisation_mask))/(np.max(final_visualisation_mask) - np.min(final_visualisation_mask))


def save_movie_mp4(image_array, video_name = "example.mp4"):
    writer = animation.FFMpegFileWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    dpi = 72.0
    xpixels, ypixels = image_array[0].shape[0], image_array[0].shape[1]
    fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)
    im = plt.figimage(image_array[0])


    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    plt.show()
    ani = animation.FuncAnimation(fig, animate, frames=len(image_array))
    ani.save(video_name, writer=writer)


def get_video_array(video_limit=500, data_path = 'my/path/to/imgs/*.jpg', functor= None, layers_kernels = None, layers_strides = None):

    def numericalSort(value):
        parts = value.split("/")[-1]
        parts = int(parts.split("_")[0])
        return parts

    imgs = []
    alpha = 0.004
    beta = 1.0 - alpha
    counter = 0
    for path in sorted(glob(data_path), key=numericalSort):
        img = cv2.imread(path)
        salient_mask = compute_visualisation_mask(img, functor, layers_kernels, layers_strides)
        salient_mask_stacked = np.dstack((salient_mask,salient_mask))
        salient_mask_stacked = np.dstack((salient_mask_stacked,salient_mask))
        blend = cv2.addWeighted(img.astype('float32'), alpha, salient_mask_stacked, beta, 0.0)
        imgs.append(blend)
        counter += 1
        if video_limit is not None:
            if counter >= video_limit:
                return imgs
    return imgs


def get_keras_functor(model_path="my/path/to/model.h5"):
    """
    Create CNN-model structure for Heatmap
    """
    custom_objects = {"GlorotUniform": tf.keras.initializers.glorot_uniform}
    model = load_model(model_path, custom_objects)

    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu', name='conv2d_1')(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu', name='conv2d_2')(x)
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu', name='conv2d_3')(x)
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu', name='conv2d_4')(x)
    conv_5 = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu', name='conv2d_5')(x)
    convolution_part = Model(inputs=[img_in], outputs=[conv_5])

    for layer_num in ('1', '2', '3', '4', '5'):
        convolution_part.get_layer('conv2d_' + layer_num).set_weights(
            model.get_layer('conv2d_' + layer_num).get_weights())
    inp = convolution_part.input  # input placeholder
    outputs = [layer.output for layer in convolution_part.layers][1:]  # all layer outputs
    functor = K.function([inp], outputs)
    return functor


def main(video_limit = 100, data_path = 'my/path/to/imgs/*.jpg', model_path="my/path/to/model.h5", video_name = "example.mp4"):
    functor = get_keras_functor(model_path= model_path)
    kernel_3x3 = tf.constant(np.array([
        [[[1]], [[1]], [[1]]],
        [[[1]], [[1]], [[1]]],
        [[[1]], [[1]], [[1]]]
    ]), tf.float32)
    kernel_5x5 = tf.constant(np.array([
        [[[1]], [[1]], [[1]], [[1]], [[1]]],
        [[[1]], [[1]], [[1]], [[1]], [[1]]],
        [[[1]], [[1]], [[1]], [[1]], [[1]]],
        [[[1]], [[1]], [[1]], [[1]], [[1]]],
        [[[1]], [[1]], [[1]], [[1]], [[1]]]
    ]), tf.float32)
    layers_kernels = {4: kernel_3x3, 3: kernel_3x3, 2: kernel_5x5, 1: kernel_5x5, 0: kernel_5x5}
    layers_strides = {4: [1, 1, 1, 1], 3: [1, 2, 2, 1], 2: [1, 2, 2, 1], 1: [1, 2, 2, 1], 0: [1, 2, 2, 1]}
    imgs = get_video_array(video_limit= video_limit, data_path = data_path, functor= functor, layers_kernels = layers_kernels, layers_strides = layers_strides)
    save_movie_mp4(imgs,  video_name)


if __name__ == '__main__':
    """
    Example use
    python3 heatmap.py -d "mycar/data/tub_4_19-12-22/*.jpg" -m "mycar/models/mod_lin_1.h5" -v "lin_mod_19-12-22-tub4_500.mp4" -c 500
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', help='Images dir', default= 'my/path/to/imgs/*.jpg', type=str)
    parser.add_argument('-m', '--model-path', help='Path to a model',
                        default='my/path/to/model.h5', type=str)
    parser.add_argument('-v', '--video-name', help='Video Name',
                        default='example.mp4', type=str)
    parser.add_argument('-c', '--number-images', help='number of images for creating video', default=100,
                        type=int)
    args = parser.parse_args()
    #Without paser use this:
    #main(200, "mycar/data/tub_4_19-12-22/*.jpg" ,"mycar/models/mod_lin_aug_1.h5", "lin_mod_aug_tub1_200.mp4" )
    main(args.number_images, args.data_path, args.model_path, args.video_name)
