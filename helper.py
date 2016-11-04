"""Create images with simple shapes for use in ShapeShop. (28x28 pixels)
   The first batch is for VGG16.
   The second batch is for MNIST.

    Most function contain the following inputs and outputs:

    Args:
        mnist_X_train_sample: a 28x28 image, can have MNIST digit information or be blank.

    Returns:
       image: the created image.
"""

from scipy.misc import imsave, imresize
import numpy as np
from scipy.ndimage import imread
from noise import snoise2
from random import randint
from keras import backend as K
from scipy import ndimage


#####
#VGG#
#####

def preprocess_image(image_path):
    img_width = 28
    img_height = 28
    img = imresize(imread(image_path), (img_width, img_height)).astype('float32')
    img = img.transpose((2, 0, 1))
    img = img.astype('float32')
    # img = img[None,]
    return img

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-6)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def perlin_noise(size):
    myarray = np.zeros((size, size), np.int)
    scale = 1/48.0
    oct = 8
    xrep = randint(128, 16535)
    yrep = randint(128, 16535)
    for y in xrange(size):
        for x in xrange(size):
            v = snoise2(x * scale, y * scale, oct, repeatx=xrep, repeaty=yrep, persistence=.25)
            if v < 0: v *= -1
            v = int(v * 255.)
            myarray[x, y] = v
    return myarray

def make_box_img(img_width, img_height):
    img = ndimage.gaussian_filter(np.random.random((1, 3, img_width, img_height))*1.0 + 1, 1)
    
    img[0,0,24:40,24:40] = 0.0
    img[0,1,24:40,24:40] = 0.0
    img[0,2,24:40,24:40] = 0.0
    
    return img[0]

def make_noise_img(img_width, img_height):
    img = ndimage.gaussian_filter(np.random.random((1, 3, img_width, img_height))*1.0 + 1, 1)
    
    return img[0]

def make_box_img_no_blur(img_width, img_height):
    img = np.random.random((1, 3, img_width, img_height))*1.0 + 1
    
    img[0,0,24:40,24:40] = 0.0
    img[0,1,24:40,24:40] = 0.0
    img[0,2,24:40,24:40] = 0.0
    
    return img[0]

def make_noise_img_no_blur(img_width, img_height):
    img = np.random.random((1, 3, img_width, img_height))*1.0 + 1
    
    return img[0]

#######
#MNIST#
#######

def boxify_top_left(mnist_X_train_sample):
    image_with_box = mnist_X_train_sample
    image_with_box[0, 0:6, 0:6] = 1.0
    return image_with_box

def boxify_top_right(mnist_X_train_sample):
    image_with_box = mnist_X_train_sample
    image_with_box[0, 0:6, 22:28] = 1.0
    return image_with_box

def boxify_bottom_left(mnist_X_train_sample):
    image_with_box = mnist_X_train_sample
    image_with_box[0, 22:28, 0:6] = 1.0
    return image_with_box

def boxify_bottom_right(mnist_X_train_sample):
    image_with_box = mnist_X_train_sample
    image_with_box[0, 22:28, 22:28] = 1.0
    return image_with_box

def lineify_top_left(mnist_X_train_sample):
    image_with_line = mnist_X_train_sample
    image_with_line[0, 0:6, 2] = 1.0
    image_with_line[0, 0:6, 3] = 1.0
    return image_with_line

def lineify_bottom_right(mnist_X_train_sample):
    image_with_line = mnist_X_train_sample
    image_with_line[0, 22:28, 28-2] = 1.0
    image_with_line[0, 22:28, 28-3] = 1.0
    return image_with_line

def circleify_top_left(mnist_X_train_sample):
    image_with_circle = mnist_X_train_sample
    
    a, b = 3,3
    n = 7
    r = 2
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r

    circle = np.ones((n, n))
    circle = circle*mask

    image_with_circle[0, 0:7, 0:7] = circle
    return image_with_circle

def circleify_bottom_right(mnist_X_train_sample):
    image_with_circle = mnist_X_train_sample
    
    a, b = 3,3
    n = 7
    r = 2
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r

    circle = np.ones((n, n))
    circle = circle*mask

    image_with_circle[0, 28-7:28, 28-7:28] = circle
    return image_with_circle

def triangulify_top_left(mnist_X_train_sample):
    image_with_tri = mnist_X_train_sample
    
    n = 7
    tri = np.zeros((n, n))
    tri[3, 0:7] = 1.0
    tri[2, 1:6] = 1.0
    tri[1, 2:5] = 1.0
    tri[0, 3] = 1.0
    image_with_tri[0, 0:7, 0:7] = tri
    
    return image_with_tri

def triangulify_bottom_right(mnist_X_train_sample):
    image_with_tri = mnist_X_train_sample
    
    n = 7
    tri = np.zeros((n, n))
    tri[3, 0:7] = 1.0
    tri[2, 1:6] = 1.0
    tri[1, 2:5] = 1.0
    tri[0, 3] = 1.0
    image_with_tri[0, 28-7:28, 28-7:28] = tri
    
    return image_with_tri

def boxify_center(mnist_X_train_sample):
    image_with_box = mnist_X_train_sample
    image_with_box[0, 11-4:17+4, 11-4:17+4] = 1.0
    return image_with_box

def lineify_center(mnist_X_train_sample):
    image_with_line = mnist_X_train_sample
    image_with_line[0, 11-4:17+4, 2+11] = 1.0
    image_with_line[0, 11-4:17+4, 3+11] = 1.0
    return image_with_line

def circleify_center(mnist_X_train_sample):
    image_with_circle = mnist_X_train_sample
    
    a, b = 7,7
    n = 15
    r = 8
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r

    circle = np.ones((n, n))
    circle = circle*mask

    image_with_circle[0, 7:22, 7:22] = circle
    return image_with_circle

def triangulify_center(mnist_X_train_sample):
    image_with_tri = mnist_X_train_sample
    
    a=12
    b=a/2
    n = 7+a
    tri = np.zeros((n, n))
    tri[n-1, 0:n] = 1.0
    tri[n-2, 1:n-1] = 1.0
    tri[n-3, 2:n-2] = 1.0
    tri[n-4, 3:n-3] = 1.0
    tri[n-5, 4:n-4] = 1.0
    tri[n-6, 5:n-5] = 1.0
    tri[n-7, 6:n-6] = 1.0
    tri[n-8, 7:n-7] = 1.0
    tri[n-9, 8:n-8] = 1.0
    tri[n-10, 9:n-9] = 1.0
    
    image_with_tri[0, 11-b-5:18+b-5, 11-b:18+b] = tri
    return image_with_tri

def noiseify():
    image_with_noise = np.random.random([1,28,28])

    return image_with_noise

###################
#ADDITIONAL SHAPES#
###################

def boxify_center_hollow(mnist_X_train_sample):
    image_with_box = mnist_X_train_sample
    border=2
    image_with_box[0, 11-4:17+4, 11-4:17+4] = 1.0
    image_with_box[0, 11-4+border:17+4-border, 11-4+border:17+4-border] = 0.0
    return image_with_box

def lineify_center_horizontal(mnist_X_train_sample):
    image_with_line = mnist_X_train_sample
    image_with_line[0, 2+11, 11-4:17+4] = 1.0
    image_with_line[0, 3+11, 11-4:17+4] = 1.0
    return image_with_line

def circleify_center_hollow(mnist_X_train_sample):
    image_with_circle = mnist_X_train_sample
    image_with_circle_big = np.copy(mnist_X_train_sample)
    
    circleify_center(image_with_circle_big)
    
    a, b = 7,7
    n = 15
    r = 5
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r

    circle = np.ones((n, n))
    circle = circle*mask

    image_with_circle[0, 7:22, 7:22] = circle
    final_image = image_with_circle_big[0]-image_with_circle[0]

    return final_image[None,]

def triangulify_center_hollow(mnist_X_train_sample):
    image_with_tri = mnist_X_train_sample
    image_with_tri_big = np.copy(mnist_X_train_sample)

    triangulify_center(image_with_tri_big)
    
    a=6
    b=a/2
    n = 7+a
    tri = np.zeros((n, n))
    tri[n-1, 0:n] = 1.0
    tri[n-2, 1:n-1] = 1.0
    tri[n-3, 2:n-2] = 1.0
    tri[n-4, 3:n-3] = 1.0
    tri[n-5, 4:n-4] = 1.0
    tri[n-6, 5:n-5] = 1.0
    tri[n-7, 6:n-6] = 1.0
    tri[n-8, 7:n-7] = 1.0
    tri[n-9, 8:n-8] = 1.0
    tri[n-10, 9:n-9] = 1.0
    
    image_with_tri[0, 11-b-3:18+b-3, 11-b:18+b] = tri
    final_image = image_with_tri_big[0] - image_with_tri[0]
    
    return final_image[None,]

def noiseify_blur():
    image_with_noise = ndimage.gaussian_filter(np.random.random((1, 28, 28))*1.0, 1)

    return image_with_noise

###############
#RANDOM SHAPES#
###############

def boxify(mnist_X_train_sample):
    box_diameter = np.random.randint(28)
    i = np.random.randint(28-box_diameter)
    j = np.random.randint(28-box_diameter)
    image_with_box = mnist_X_train_sample
    image_with_box[0, i:i+box_diameter, j:j+box_diameter] = 1.0
    return image_with_box

def lineify(mnist_X_train_sample):
    line_diameter = np.random.randint(28)
    i = np.random.randint(28-line_diameter)
    j = np.random.randint(28-line_diameter)
    image_with_line = mnist_X_train_sample
    image_with_line[0, i:i+line_diameter, j] = 1.0
    image_with_line[0, i:i+line_diameter, j+1] = 1.0
    return image_with_line

def circleify(mnist_X_train_sample):
    image_with_circle = mnist_X_train_sample
    
    n = 28
    r = np.random.randint(1, high=n/2)
    
    a = np.random.randint(r, high=n-r)
    b = np.random.randint(r, high=n-r)
    
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r

    circle = np.ones((n, n))
    circle = circle*mask

    image_with_circle[0, 0:n, 0:n] = circle
    return image_with_circle

######
#HALF#
######

def boxify_top_left_half(mnist_X_train_sample):
    image_with_box = mnist_X_train_sample
    image_with_box[0, 3:6, 3:6] = 1.0
    return image_with_box