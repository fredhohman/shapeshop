from __future__ import print_function
from scipy.misc import imsave, imresize
import numpy as np
import time
import os

from keras.models import Sequential
from keras.layers import *
from keras.layers.advanced_activations import ELU
from keras import backend as K

from keras.models import Model

from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

import scipy
from scipy import ndimage
from scipy.ndimage import imread

from keras.datasets import mnist
from keras.models import model_from_json
import keras

from time import sleep

import random

from keras.initializations import normal, identity
from keras.optimizers import SGD, RMSprop
from random import randint

from helper import boxify_top_left, boxify_top_right
from helper import lineify_top_left, lineify_bottom_right
from helper import circleify_top_left, circleify_bottom_right
from helper import triangulify_top_left, triangulify_bottom_right
from helper import boxify_center, lineify_center, circleify_center, triangulify_center
from helper import boxify_center_hollow, lineify_center_horizontal, circleify_center_hollow, triangulify_center_hollow
from helper import noiseify, noiseify_blur

from helper import normalize

# 0 = do not include in training data
# 1 = include in training data

def preprocess(training_data_indicies):
    x_data = []
    y_data = []

    num_of_pictures = 3 #5000*2

    blank = np.zeros([1,28,28])
    
    num_total_training_images = len(training_data_indicies)
    
    for i in range(num_of_pictures):
        
        # row 1
        if training_data_indicies[0%num_total_training_images] == 1:
            x_data.append(boxify_center(np.copy(blank)))
            y_data.append(0)

        if training_data_indicies[1%num_total_training_images] == 1:
            x_data.append(boxify_center_hollow(np.copy(blank)))
            y_data.append(1)
            
        if training_data_indicies[2%num_total_training_images] == 1:
            x_data.append(lineify_center(np.copy(blank)))
            y_data.append(2)
            
        if training_data_indicies[3%num_total_training_images] == 1:
            x_data.append(lineify_center_horizontal(np.copy(blank)))
            y_data.append(3)
            
        if training_data_indicies[4%num_total_training_images] == 1:
            x_data.append(circleify_center(np.copy(blank)))
            y_data.append(4)
                        
        if training_data_indicies[5%num_total_training_images] == 1:
            x_data.append(circleify_center_hollow(np.copy(blank)))
            y_data.append(5)
            
        if training_data_indicies[6%num_total_training_images] == 1:
            x_data.append(triangulify_center(np.copy(blank)))
            y_data.append(6)
            
        if training_data_indicies[7%num_total_training_images] == 1:
            x_data.append(triangulify_center_hollow(np.copy(blank)))
            y_data.append(7)
            
        # row 2
        if training_data_indicies[8%num_total_training_images] == 1:
            x_data.append(boxify_top_left(np.copy(blank)))
            y_data.append(8)

        if training_data_indicies[9%num_total_training_images] == 1:
            x_data.append(boxify_bottom_right(np.copy(blank)))
            y_data.append(9)
            
        if training_data_indicies[10%num_total_training_images] == 1:
            x_data.append(lineify_top_left(np.copy(blank)))
            y_data.append(10)

        if training_data_indicies[11%num_total_training_images] == 1:
            x_data.append(lineify_bottom_right(np.copy(blank)))
            y_data.append(11)
            
        if training_data_indicies[12%num_total_training_images] == 1:
            x_data.append(circleify_top_left(np.copy(blank)))
            y_data.append(12)

        if training_data_indicies[13%num_total_training_images] == 1:
            x_data.append(circleify_bottom_right(np.copy(blank)))
            y_data.append(13)
            
        if training_data_indicies[14%num_total_training_images] == 1:
            x_data.append(triangulify_top_left(np.copy(blank)))
            y_data.append(14)

        if training_data_indicies[15%num_total_training_images] == 1:
            x_data.append(triangulify_bottom_right(np.copy(blank)))
            y_data.append(15)
            
        # row 3
        if training_data_indicies[16%num_total_training_images] == 1:
            x_data.append(noiseify())
            y_data.append(16)
            
        if training_data_indicies[17%num_total_training_images] == 1:
            x_data.append(noiseify_blur())
            y_data.append(17)
        
    nb_classes = np.sum(training_data_indicies)
    print(nb_classes)
    X_train_new = np.array(x_data)
    y_train_new = np.array(y_data)

    print(X_train_new.shape)
    print(y_train_new.shape)
    Y_train_new = np_utils.to_categorical(y_train_new, nb_classes)

    import random
    s=list(range(X_train_new.shape[0]))
    random.shuffle(s) 
    X2 = X_train_new[s] #+np.random.random(X_train_new.shape)*0.01
    Y2 = Y_train_new[s]
    
    return X2, Y2


def build_and_train_model(X2, Y2):
	batch_size = 4 #32
	nb_epoch = 50
	img_rows, img_cols = 28, 28
	WIDTH = 64*2
	num_layers = 8

	input = Input(batch_shape=(batch_size, 1, img_rows, img_cols))

	m = Flatten()(input)
	m = Dense(WIDTH, activation='tanh')(m)
	# m = Dropout(0.2)(m)
	m = Dense(WIDTH, activation='tanh')(m)
	m = Dense(nb_classes, activation='softmax')(m)

	model = Model(input=input, output=[m])

	model.compile(loss='categorical_crossentropy',
	              optimizer='sgd',
	              metrics=['accuracy'])

	model.fit(X2,Y2,batch_size=batch_size,nb_epoch=nb_epoch,validation_split=0.2,shuffle=True, verbose=2)
	sleep(0.1)
	return model, input

def draw_images(img_num, model, input):
    # we build a loss function 
    loss = model.output[0,img_num]
    print(loss)
#     loss = model.output[0][0,img_num]+model.output[1][0,1]
    
    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)
    
    # this function returns the loss and grads given the input picture
    iterate = K.function([input, K.learning_phase()], [loss, grads])
    
    # random
    # input_img_data = np.copy(X_train[img_num][None,]) # starting with image in X_train
#     input_img_data = np.copy(X_train[15][None,]) # starting with image in X_train
#     input_img_data = ndimage.gaussian_filter(np.random.random((1, 1, img_width, img_height))*1.0, 1)
#     input_img_data = np.random.random((1, 1, img_width, img_height))*0.5
    input_img_data = np.zeros([1,1,28,28])
#     input_img_data = boxify_top_left_half(np.zeros([1, img_width, img_height]))[None,]
    
    temp_time = time.time()
#     print('Time after initialization:' , temp_time - start_time)
    
    # we run gradient ascent    
    step = 0.005 #0.05
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

                temp = np.copy(grads_value[:, (img_width*(0.5-0.5*j/nsteps)):(img_width*(0.5+0.5*j/nsteps)),
                                              (img_height*(0.5-0.5*j/nsteps)):(img_height*(0.5+0.5*j/nsteps))])
                grads_value[:] = 0.0
                grads_value[:, img_width*(0.5-0.5*j/nsteps):img_width*(0.5+0.5*j/nsteps),
                               img_height*(0.5-0.5*j/nsteps):img_width*(0.5+0.5*j/nsteps)] = temp
                
                input_img_data += grads_value * step

                if loss_value > 0.999: #+1 #for multiple
#                     img = deprocess_image(np.copy(input_img_data[0]))
                    img = 1-input_img_data[0,0]
                    # imsave('/Users/fredhohman/Desktop/pnnl/ubuntu-vm/Desktop/data/binary-net/mnist-box/testing/' + str(loop_num) + '/box_' + str(img_num) + '.png', img)
                    loss_value, grads_value = iterate([input_img_data, L_PHASE])
                    print('Current loss value:', loss_value,'- Current intensity:', np.mean(input_img_data))
                    
#                     plt.imshow(input_img_data[0,0], cmap='gray')
#                     plt.show()

                    return True, img   # draw an image
            if INIT_STEP % 300 == 0: 
                # print(loss_value)
            
        idx += idx

                
    if loss_value < 0.99:
        print('Current loss value:', loss_value, '- Current intensity:', np.mean(input_img_data))
        print('Did not make it to 0.99')
        return False    # did not draw an image


if __name__ == "__main__":
	img_width = 28
	img_height = 28

	test = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
	num_of_pictures = np.sum(test)
	nb_classes = num_of_pictures
	X2, Y2 = preprocess(test)
	print(X2.shape)
	print(Y2.shape)

	model, input = build_and_train_model(X2, Y2)

	img_num = 0
	results = np.zeros([num_of_pictures, img_width, img_height])

	while img_num < num_of_pictures:
	       
	    start_time = time.time()
	    print('START image', str(img_num))

	    result, img = draw_images(img_num, model, input)
	               
	    end_time = time.time()
	    print('END image', str(img_num-1) + ":", end_time - start_time)

	    if result == True:
	        img_num += 1
	    imsave(str(img_num) + '.png', img)


	            
	 




