"""The main code for:
    * creating the training data,
    * building and training the neural network model, 
    * and image generation.
"""

from __future__ import print_function
from scipy.misc import imsave, imresize
import numpy as np
import time
from time import sleep
import os
import random

from keras import backend as K
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import SGD, RMSprop

import scipy
from scipy import ndimage
from scipy.ndimage import imread
from scipy.stats import pearsonr

from helper import boxify_top_left, boxify_bottom_right
from helper import lineify_top_left, lineify_bottom_right
from helper import circleify_top_left, circleify_bottom_right
from helper import triangulify_top_left, triangulify_bottom_right
from helper import boxify_center, lineify_center, circleify_center, triangulify_center
from helper import boxify_center_hollow, lineify_center_horizontal, circleify_center_hollow, triangulify_center_hollow
from helper import noiseify, noiseify_blur
from helper import normalize

import matplotlib.pyplot as plt


def preprocess(training_data_indicies):
    """Builds the dataset. 
        Args:
            training_data_indicies: an array of 0s and 1s, where 1s indicate selected training images to include.  

        Returns:
           X: the dataset.
           Y: the dataset labels.
   """

    x_data = []
    y_data = []

    num_of_pictures = 3

    blank = np.zeros([1,28,28])
    
    num_total_training_images = len(training_data_indicies)
   
    # 0 = do not include in training data
    # 1 = include in training data
    for i in range(num_of_pictures):
    	counter = 0
        
        # row 1
        if training_data_indicies[0%num_total_training_images] == 1:
            x_data.append(boxify_center(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1

        if training_data_indicies[1%num_total_training_images] == 1:
            x_data.append(boxify_center_hollow(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1
            
        if training_data_indicies[2%num_total_training_images] == 1:
            x_data.append(lineify_center(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1
            
        if training_data_indicies[3%num_total_training_images] == 1:
            x_data.append(lineify_center_horizontal(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1
            
        if training_data_indicies[4%num_total_training_images] == 1:
            x_data.append(circleify_center(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1
                        
        if training_data_indicies[5%num_total_training_images] == 1:
            x_data.append(circleify_center_hollow(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1
            
        if training_data_indicies[6%num_total_training_images] == 1:
            x_data.append(triangulify_center(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1
            
        if training_data_indicies[7%num_total_training_images] == 1:
            x_data.append(triangulify_center_hollow(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1
            
        # row 2
        if training_data_indicies[8%num_total_training_images] == 1:
            x_data.append(boxify_top_left(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1

        if training_data_indicies[9%num_total_training_images] == 1:
            x_data.append(boxify_bottom_right(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1
            
        if training_data_indicies[10%num_total_training_images] == 1:
            x_data.append(lineify_top_left(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1

        if training_data_indicies[11%num_total_training_images] == 1:
            x_data.append(lineify_bottom_right(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1
            
        if training_data_indicies[12%num_total_training_images] == 1:
            x_data.append(circleify_top_left(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1

        if training_data_indicies[13%num_total_training_images] == 1:
            x_data.append(circleify_bottom_right(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1
            
        if training_data_indicies[14%num_total_training_images] == 1:
            x_data.append(triangulify_top_left(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1

        if training_data_indicies[15%num_total_training_images] == 1:
            x_data.append(triangulify_bottom_right(np.copy(blank)))
            y_data.append(counter)
            counter = counter + 1
            
        # row 3
        if training_data_indicies[16%num_total_training_images] == 1:
            x_data.append(noiseify())
            y_data.append(counter)
            counter = counter + 1
            
        if training_data_indicies[17%num_total_training_images] == 1:
            x_data.append(noiseify_blur())
            y_data.append(counter)
            counter = counter + 1
        
    nb_classes = np.sum(training_data_indicies)
    print(nb_classes)

    X_temp = np.array(x_data)
    y_temp = np.array(y_data)

    print(X_temp.shape)
    print(y_temp.shape)
    y_temp_2 = np_utils.to_categorical(y_temp, nb_classes)

    s = list(range(X_temp.shape[0]))
    random.shuffle(s) 
    X = X_temp[s]+np.random.random(X_temp.shape)*0.01
    Y = y_temp_2[s]
    
    return X, Y

def build_and_train_model(X, Y, nb_classes, model_type, epoch):
    """Builds and trains the neural network image classifier model. 
        Args:
            X: the dataset.
            Y: the labels.
            nb_classes: number of classes in the image classifier.
            model_type: delineating between multilayer perceptron and convolutional neural network.
            epoch: number of epochs for training.

        Returns:
           model: the trained model.
           input: the input layer of the model. 
   """
    batch_size = 4 
    nb_epoch = epoch
    img_rows, img_cols = 28, 28
    WIDTH = 64*2

    input = Input(batch_shape=(batch_size, 1, img_rows, img_cols))
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)
    
    print(str(model_type))
    print(type(str(model_type)))
    print(len(str(model_type)))

    if str(model_type).strip() == "MLP":
        m = Flatten()(input)
        m = Dense(WIDTH, activation='tanh')(m)
        # m = Dropout(0.2)(m)
        m = Dense(WIDTH, activation='tanh')(m)
        m = Dense(nb_classes, activation='softmax')(m)

    if str(model_type).strip() == "CNN":
        m = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid')(input)
        m = Activation('relu')(m)
        m = Convolution2D(nb_filters, kernel_size[0], kernel_size[1])(m)
        m = Activation('relu')(m)
        m = MaxPooling2D(pool_size=pool_size)(m)
        m = Dropout(0.25)(m)

        m = Flatten()(m)
        m = Dense(128)(m)
        m = Activation('relu')(m)
        m = Dropout(0.5)(m)
        m = Dense(nb_classes)(m)
        m = Activation('softmax')(m)

    model = Model(input=input, output=[m])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta', #'sgd'
                  metrics=['accuracy'])

    model.fit(X,Y,batch_size=batch_size,nb_epoch=nb_epoch,validation_split=0.2,shuffle=True, verbose=2)
    sleep(0.1)
    return model, input

def draw_images(img_num, model, input, initial_image_indicies, step_size):
    """Performs the class activation maximization image drawing/generation process.
        Args:
            img_num: iterator for drawing multiple images.
            model: the trained model.
            input: the input layer of the trained model.
            initial_image_indicies: specifies which image to initialize the image generation process.
            step_size: the step_size used for gradient ascent.

        Returns:
           If success: True, the generated image.
           If failure: False.
   """
    
    # we build a loss function 
    loss = model.output[0,img_num]
    print(loss)

    img_width = 28
    img_height = 28

    # we compute the gradient of the input picture with respect to this loss
    grads = K.gradients(loss, input)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)
    
    # this function returns the loss and grads given the input picture
    iterate = K.function([input, K.learning_phase()], [loss, grads])
    
    # create initial image
    if initial_image_indicies[0] == 1:
        input_img_data = np.zeros([1, 1, img_width, img_height])
        print("initial image is zeros")
    if initial_image_indicies[1] == 1:
        input_img_data = np.ones([1, 1, img_width, img_height])
        print("initial image is ones")
    if initial_image_indicies[2] == 1:
        input_img_data = np.random.random((1, 1, img_width, img_height))*1.0
        print("initial image is random")
    if initial_image_indicies[3] == 1: 
        input_img_data = ndimage.gaussian_filter(np.random.random((1, 1, img_width, img_height))*1.0, 1)
        print("initial image is random blur")

    temp_time = time.time()
#     print('Time after initialization:' , temp_time - start_time)
    
    # we run gradient ascent    
    step = step_size
    switched_on = True # should be False for drawing VGG pictures
    NUM_ITERS = 2
    INIT_STEP = 300
    L_PHASE = 0
    DROPOUT_RATE = 0.5
    nsteps = 2
    loss_value = None
    idx = 0
#     for idx in range(NUM_ITERS):
    while loss_value < 0.99:

        if not switched_on:        
            image2 = scipy.misc.imresize(input_img_data[0],2.0).transpose((2,0,1))            
            d,w,h = image2.shape
            m = np.mean(image2[:, (w/2 - img_width/2):(w/2 + img_width/2),
                                  (h/2 - img_height/2):(h/2 + img_height/2)])
            input_img_data[0] = image2[:, (w/2 - img_width/2):(w/2 + img_width/2),
                                        (h/2 - img_height/2):(h/2 + img_height/2)]/m

        for rep in range(0,INIT_STEP):
            for j in range(1,(nsteps+1 + idx)):#/((i+1))):
                
                loss_value, grads_value = iterate([input_img_data,L_PHASE])

                # temp = np.copy(grads_value[:, (img_width*(0.5-0.5*j/nsteps)):(img_width*(0.5+0.5*j/nsteps)),
                #                               (img_height*(0.5-0.5*j/nsteps)):(img_height*(0.5+0.5*j/nsteps))])
                # grads_value[:] = 0.0
                # grads_value[:, img_width*(0.5-0.5*j/nsteps):img_width*(0.5+0.5*j/nsteps),
                #                img_height*(0.5-0.5*j/nsteps):img_width*(0.5+0.5*j/nsteps)] = temp
                
                input_img_data += grads_value * step

                if loss_value > 0.999: #+1 #for multiple
                    # img = deprocess_image(np.copy(input_img_data[0]))
                    img = 1-input_img_data[0,0]
                    loss_value, grads_value = iterate([input_img_data, L_PHASE])
                    print('Current loss value:', loss_value,'- Current intensity:', np.mean(input_img_data))
                    
                    # plt.imshow(input_img_data[0,0], cmap='gray')
                    # plt.show()

                    return True, img   # draw an image
            # if INIT_STEP % 300 == 0: 
                # print(loss_value)
            
        idx += idx

                
    if loss_value < 0.99:
        print('Current loss value:', loss_value, '- Current intensity:', np.mean(input_img_data))
        print('Did not make it to 0.99')
        return False    # did not draw an image

def compute_error(training_data_indicies, results):
    """Computes the correlation coefficient for generated images.
        Args:
            training_data_indicies: an array of 0s and 1s, where 1s indicate selected training images to include.
            results: the generated images.

        Returns:
           errors: correlation coefficients for each generated image.
    """
    x_data = []
    blank = np.zeros([1,28,28])

    # row 1
    x_data.append(boxify_center(np.copy(blank)))
    x_data.append(boxify_center_hollow(np.copy(blank)))
    x_data.append(lineify_center(np.copy(blank)))
    x_data.append(lineify_center_horizontal(np.copy(blank)))
    x_data.append(circleify_center(np.copy(blank)))
    x_data.append(circleify_center_hollow(np.copy(blank)))
    x_data.append(triangulify_center(np.copy(blank)))
    x_data.append(triangulify_center_hollow(np.copy(blank)))
    # row 2
    x_data.append(boxify_top_left(np.copy(blank)))
    x_data.append(boxify_bottom_right(np.copy(blank)))
    x_data.append(lineify_top_left(np.copy(blank)))
    x_data.append(lineify_bottom_right(np.copy(blank)))
    x_data.append(circleify_top_left(np.copy(blank)))
    x_data.append(circleify_bottom_right(np.copy(blank)))
    x_data.append(triangulify_top_left(np.copy(blank)))
    x_data.append(triangulify_bottom_right(np.copy(blank)))
    # row 3
    x_data.append(noiseify())
    x_data.append(noiseify_blur())

    training_data_indicies_nonzero = np.nonzero(training_data_indicies)[0]
    errors = []

    for i in range(results.shape[0]):

        # print(training_data_indicies)
        # print(training_data_indicies_nonzero)
        # print(training_data_indicies_nonzero[i])
        org = x_data[training_data_indicies_nonzero[i]].flatten()
        gen = results[i].flatten()

        error = pearsonr(org, gen)
        errors.append(error)

    errors = np.array(np.abs(errors))

    return errors[:,0]

def save_image(data, cm, fn, dpi):
    """Saves a generated image to disk.
        Args:
            data: the image to save.
            cm = the colormap used when saving.
            fn: file name.
            dpi: resolution of saved image.

        Returns:
           None.
   """
   
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.matshow(data, cmap=cm)
    plt.savefig(fn, dpi = dpi) 
    plt.close()

    return None

def model(training_data_indicies, initial_image_indicies, number_of_times_clicked, step_size, model_type, epoch):
    """Computes the correlation coefficient for generated images.
        Args:
            training_data_indicies: an array of 0s and 1s, where 1s indicate selected training images to include.
            initial_image_indicies: specifies which image to initialize the image generation process.
            number_of_times_clicked: the experiment number.
            step_size: the step_size used for gradient ascent.
            model_type: delineating between multilayer perceptron and convolutional neural network.
            epoch: number of epochs for training.

        Returns:
           results: the generated images.
           errors: correlation coefficients for each generated image.

    """
    img_width = 28
    img_height = 28

    num_of_pictures = np.sum(training_data_indicies)
    nb_classes = num_of_pictures
    X, Y = preprocess(training_data_indicies)
    print(X.shape)
    print(Y.shape)

    model, input = build_and_train_model(X, Y, nb_classes, model_type, epoch)

    img_num = 0
    results = []
    errors = np.zeros(num_of_pictures)

    while img_num < num_of_pictures:
           
        start_time = time.time()
        print('START image', str(img_num))

        result_bool, img = draw_images(img_num, model, input, initial_image_indicies, step_size)
                   
        end_time = time.time()
        print('END image', str(img_num) + ":", end_time - start_time)

        if result_bool == True:
            img_num += 1

        save_image(1-img,'gray','static/results/' + str(number_of_times_clicked) + '_' + str(img_num) + '.png', 500)
        results.append(1-img)

    results = np.array(results)
    errors = compute_error(training_data_indicies, results)

    return results, errors

	    