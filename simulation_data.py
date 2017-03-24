import csv
import scipy.misc
from random import shuffle
import cv2
from skimage.util import random_noise
from numpy.random import uniform as random
import numpy as np

class data_handler(object):
    def __init__(self, validation_split = 0.2, batch_size = 128, left_and_right_images = False, root_path = '', left_right_offset = 0.2, test_root_path = '', test_left_and_right_images = False):

        # Name of file where metadata is present
        filename = 'driving_log.csv'
        test_filename = 'test_driving_log.csv'

        self.left_and_right_images = left_and_right_images
        self.left_right_offset = left_right_offset

        self.metadata = []

        # loading metadata
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            i = 0
            for row in reader:
                self.metadata.append(row)

        # removing first row if it has column names
        if(self.metadata[0][0]=='center'):
            self.metadata.reverse()
            self.metadata.pop()
            self.metadata.reverse()

        # shuffle the training data
        shuffle(self.metadata)

        self.test_metadata = []

        # loading metadata
        with open(test_filename, 'r') as f:
            reader = csv.reader(f)
            i = 0
            for row in reader:
                self.test_metadata.append(row)

        # removing first row if it has column names
        if(self.test_metadata[0][0]=='center'):
            self.test_metadata.reverse()
            self.test_metadata.pop()
            self.test_metadata.reverse()

        # splitting into training and validation set
        if(validation_split<1.0):
            self.metadata_train = self.metadata[0:int((1-validation_split)*len(self.metadata))]
            if(not validation_split==0):
                self.metadata_val = self.metadata[int((1-validation_split)*len(self.metadata)):]
        else:
            print("Validation split can't be 1.")
            raise Exception("Validation split not valid.")

        # setting batch size
        self.batch_size = batch_size

        # setting current training step (in the beginning we are at the 0th step)
        self.step_train = 0

        # setting current validation step (in the beginning we are at the 0th step)
        self.step_val = 0

        # setting current validation step (in the beginning we are at the 0th test step)
        self.step_test = 0

        # root path of images
        self.root_path = root_path

        # root path of test images
        self.test_root_path = test_root_path

        # left and right images for
        self.test_left_and_right_images = test_left_and_right_images

    def generate_train_batch(self):
        while 1:
            X_train = []
            y_train = []

            # start and end of current batch
            start = self.step_train*self.batch_size
            end = (self.step_train+1)*self.batch_size

            # if number of training samples are not a multiple of batch size
            if(end>=len(self.metadata_train)):
                end = len(self.metadata_train)

                # restart from the beginning
                self.step_train = 0
                shuffle(self.metadata_train)

            # load images and steering angles for current batch
            for j in range(start,end,1):
                if(not self.metadata_train[j][0][0] == 'C'):
                    center_path = self.root_path+self.metadata_train[j][0]
                else:
                    center_path = self.metadata_train[j][0]

                center_steer = [float(self.metadata_train[j][3])]
                # X_train.append(self.get_image(self.root_path+self.metadata_train[j][0]))
                # y_train.append([float(self.metadata_train[j][3])])

                center_image, center_steer[0] = self.get_image_and_steering(center_path,center_steer[0])
                X_train.append(center_image)
                y_train.append(center_steer)

                if(self.left_and_right_images):
                    if(self.metadata_train[j][1][0] == ' ' and not self.metadata_train[j][1][1]=='C'):
                        left_path = self.root_path+self.metadata_train[j][1][1:]
                    elif(self.metadata_train[j][1][0] == ' ' and self.metadata_train[j][1][1]=='C'):
                        left_path = self.metadata_train[j][1][1:]
                    elif(self.metadata_train[j][1][0] == 'C'):
                        left_path = self.metadata_train[j][1]
                    else:
                        left_path = self.root_path + self.metadata_train[j][1]

                    left_steer = [float(self.metadata_train[j][3])+self.left_right_offset]

                    if(self.metadata_train[j][2][0] == ' ' and not self.metadata_train[j][2][1]=='C'):
                        right_path = self.root_path+self.metadata_train[j][2][1:]
                    elif(self.metadata_train[j][2][0] == ' ' and self.metadata_train[j][2][1]=='C'):
                        right_path = self.metadata_train[j][2][1:]
                    elif(self.metadata_train[j][2][0] == 'C'):
                        right_path = self.metadata_train[j][2]
                    else:
                        right_path = self.root_path + self.metadata_train[j][2]

                    right_steer = [float(self.metadata_train[j][3])-self.left_right_offset]

                    left_image, left_steer[0] = self.get_image_and_steering(left_path, left_steer[0])
                    right_image, right_steer[0] = self.get_image_and_steering(right_path, right_steer[0])

                    X_train.append(left_image)
                    y_train.append(left_steer)

                    X_train.append(right_image)
                    y_train.append(right_steer)

                    # X_train.append(self.get_image(self.root_path+self.metadata_train[j][1][1:]))
                    # y_train.append([float(self.metadata_train[j][3])+self.left_right_offset])
                    # X_train.append(self.get_image(self.root_path+self.metadata_train[j][2][1:]))
                    # y_train.append([float(self.metadata_train[j][3])-self.left_right_offset])

            # incrementing step
            self.step_train = self.step_train + 1

            yield (X_train, y_train)

    def generate_validation_batch(self):
        while 1:
            X_val = []
            y_val = []

            # start and end of current batch
            start = self.step_val*self.batch_size
            end = (self.step_val+1)*self.batch_size

            # if number of validation samples are not a multiple of batch size
            if(end>=len(self.metadata_val)):
                end = len(self.metadata_val)

                # restart from the beginning
                self.step_val = 0

                shuffle(self.metadata_val)

            # laod images and steering angles for current batch
            for j in range(start,end):
                if(not self.metadata_val[j][0][0] == 'C'):
                    center_path = self.root_path+self.metadata_val[j][0]
                else:
                    center_path = self.metadata_val[j][0]

                center_steer = [float(self.metadata_val[j][3])]
                # X_val.append(self.get_image(self.root_path+self.metadata_val[j][0]))
                # y_val.append([float(self.metadata_val[j][3])])
                center_image, center_steer[0] = self.get_image_and_steering(center_path, center_steer[0])

                X_val.append(center_image)
                y_val.append(center_steer)

                if(self.left_and_right_images):
                    if(self.metadata_val[j][1][0]==' ' and not self.metadata_val[j][1][1] == 'C'):
                        path_left = self.root_path + self.metadata_val[j][1][1:]
                    elif(self.metadata_val[j][1][0]==' ' and self.metadata_val[j][1][1] == 'C'):
                        path_left = self.metadata_val[j][1][1:]
                    elif(self.metadata_val[j][1][0] == 'C'):
                         path_left = self.metadata_val[j][1]
                    else:
                        path_left = self.root_path + self.metadata_val[j][1]

                    steer_left = [float(self.metadata_val[j][3])+self.left_right_offset]

                    if(self.metadata_val[j][2][0] == ' ' and not self.metadata_val[j][2][1] == 'C'):
                        path_right = self.root_path+self.metadata_val[j][2][1:]
                    elif(self.metadata_val[j][2][0] == ' ' and self.metadata_val[j][2][1] == 'C'):
                        path_right = self.metadata_val[j][2][1:]
                    elif(self.metadata_val[j][2][0] == 'C'):
                        path_right = self.metadata_val[j][2]
                    else:
                        path_right = self.root_path+self.metadata_val[j][2]

                    steer_right = [float(self.metadata_val[j][3])-self.left_right_offset]

                    image_left, steer_left[0] = self.get_image_and_steering(path_left,steer_left[0])
                    image_right, steer_right[0] = self.get_image_and_steering(path_right, steer_right[0])
                    X_val.append(image_left)
                    y_val.append(steer_left)

                    X_val.append(image_right)
                    y_val.append(steer_right)
                    #
                    # X_val.append(self.get_image(self.root_path+self.metadata_train[j][1][1:]))
                    # y_val.append([float(self.metadata_train[j][3])+self.left_right_offset])
                    # X_val.append(self.get_image(self.root_path+self.metadata_train[j][2][1:]))
                    # y_val.append([float(self.metadata_train[j][3])-self.left_right_offset])

            # incrementing step
            self.step_val = self.step_val + 1

            yield (X_val, y_val)

    def generate_test_batch(self):

        while 1:
            X_test = []
            y_test = []

            start = self.step_test*self.batch_size
            end = (self.step_test+1)*self.batch_size

            if(end >= len(self.test_metadata)):
                end = len(self.test_metadata)
                self.step_test = 0
                shuffle(self.test_metadata)

            for j in range(start,end):
                center_path = self.root_path +self.test_metadata[j][0]
                center_steer = [float(self.test_metadata[j][3])]
                # X_val.append(self.get_image(self.root_path+self.metadata_val[j][0]))
                # y_val.append([float(self.metadata_val[j][3])])
                center_image, center_steer[0] = self.get_image_and_steering(center_path, center_steer[0])

                X_test.append(center_image)
                y_test.append(center_steer)

                if(self.test_left_and_right_images):
                    path_left = self.test_root_path + self.test_metadata[j][1][1:]
                    steer_left = [float(self.test_metadata[j][3])+self.left_right_offset]

                    path_right = self.test_root_path + self.test_metadata[j][2][1:]
                    steer_right = [float(self.test_metadata[j][3])-self.left_right_offset]

                    image_left, steer_left[0] = self.get_image_and_steering(path_left,steer_left[0])
                    image_right, steer_right[0] = self.get_image_and_steering(path_right, steer_right[0])
                    X_test.append(image_left)
                    y_test.append(steer_left)

                    X_test.append(image_right)
                    y_test.append(steer_right)

            self.step_test = self.step_test + 1
            yield X_test, y_test, int(len(self.test_metadata)/self.batch_size)

    def set_root_image_path(self,path):
        self.root_path = path

    def move_to_start_train(self):
        self.step_train = 0

    def move_to_start_val(self):
        self.step_val = 0

    def num_train_batches(self):
        return int(len(self.metadata_train) / self.batch_size)

    def num_val_batches(self):
        return int(len(self.metadata_val) / self.batch_size)

    def add_noise(self,x):
        return random_noise(x, mode='gaussian')

    def get_image_and_steering(self,path,steering):
        image = scipy.misc.imresize(scipy.misc.imread(path)[25:135], [66, 200])

        if(self.coin_flip()):
            image = self.random_saturation_change(image)

        if(self.coin_flip()):
            image = self.random_lightness_change(image)

        if(self.coin_flip()):
            image = self.invert_image(image)

        image = self.random_shadow(image)

        image, steering = self.random_translation(image,steering)

        if(self.coin_flip()):
            image, steering = self.horizontal_flip_image(image,steering)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

        return (image/255.0)-0.5, steering

    def coin_flip(self):
        return random()<0.5

    def make_yuv_grey_scale(self,x):
        x = np.array(x)
        x[:,:,1] = 0
        x[:,:,2] = 0
        return x

    def random_gamma_correction_rgb(self,x):
        # Partially taken from http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        gamma = 0.4 + random() * 1.2
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(x, table)

    def random_brightness_change_rgb(self,x):
        brightness_change = 0.4 + random()*1.2
        x = np.array(x)
        x = cv2.cvtColor(x,cv2.COLOR_RGB2HSV)
        x[:,:,2] = x[:,:,2]*brightness_change
        return cv2.cvtColor(x,cv2.COLOR_HSV2RGB)


    def random_saturation_change(self,x):
        saturation_change = 1.5*random()
        x = np.array(x)
        x = cv2.cvtColor(x,cv2.COLOR_RGB2HSV)
        x[:,:,1] = x[:,:,1]*saturation_change
        return cv2.cvtColor(x,cv2.COLOR_HSV2RGB)

    def invert_image(self,x):
        return -x+255

    def random_lightness_change(self,x):
        lightness_change = 0.2 + 1.4*random()
        x = np.array(x)
        x = cv2.cvtColor(x,cv2.COLOR_RGB2HLS)
        x[:,:,1] = x[:,:,1]*lightness_change
        return cv2.cvtColor(x,cv2.COLOR_HLS2RGB)


    def random_translation(self,x,steer):
        x = np.array(x)
        rows,cols,rgb = x.shape

        rand_for_x = random()

        translate_y = -10 + random()*20
        translate_x = -30 + rand_for_x*60

        M = np.float32([[1,0,translate_x],[0,1,translate_y]])
        return cv2.warpAffine(x,M,(cols,rows)), (steer+(rand_for_x-0.5)*0.4)

    # def random_translation(self,x,steer):
    #     x = np.array(x)wwwwwwwwwwwwwwwwwwwwww
    #     rows,cols,rgb = x.shape
    #
    #     rand_for_x = random()
    #     rand_for_y = random()
    #
    #     translate_y = -15 + rand_for_y*30
    #     translate_x = -30 + rand_for_x*60
    #
    #     M = np.float32([[1,0,translate_x],[0,1,translate_y]])
    #
    #     return cv2.warpAffine(x,M,(cols,rows)), ((steer+(rand_for_x-0.5)*0.27))

    def random_rotation_image(self,x):
        x = np.array(x)
        rows,cols,rgb = x.shape

        rand_angle = 3*(random()-0.5)

        M = cv2.getRotationMatrix2D((cols/2,rows/2),rand_angle,1)
        x = cv2.warpAffine(x,M,(cols,rows))
        return x

    def horizontal_flip_image(self,x,steer):
        steer = -steer
        x = np.array(x)
        return cv2.flip(x,1), steer


    def random_shadow(self,x):
        x = cv2.cvtColor(x,cv2.COLOR_RGB2HSV)

        max_x = 200
        max_y = 66

        if(self.coin_flip()):
            i_1 = (0,0)
            i_2 = (0,max_y)
            i_3 = (random()*max_x,max_y)
            i_4 = (random()*max_x,0)
        else:
            i_1 = (random()*max_x,0)
            i_2 = (random()*max_x,max_y)
            i_3 = (max_x,max_y)
            i_4 = (max_x,0)

        vertices = np.array([[i_1,i_2,i_3,i_4]], dtype = np.int32)

        x = self.region_of_interest(x,vertices)

        x = cv2.cvtColor(x,cv2.COLOR_HSV2RGB)
        return x

    def random_blur(self,x):
        kernel_size = 1+int(random()*9)
        if(kernel_size%2 == 0):
            kernel_size = kernel_size + 1
        x = cv2.GaussianBlur(x,(kernel_size,kernel_size),0)
        return x

    def region_of_interest(self,x, vertices):

        random_brightness = 0.20
        mask = np.zeros_like(x)

        ignore_mask_color = [0,0,255]

        cv2.fillPoly(mask, vertices, ignore_mask_color)

        indices = mask[:,:,2] == 255

        x[:,:,2][indices] = x[:,:,2][indices]*random_brightness

        return x

    def cut_top(self,x):
        x = cv2.cvtColor(x,cv2.COLOR_RGB2HSV)
        vertices = np.array([[(0,0),(200,0),(200,33),(0,33)]],np.int32)
        random_brightness = 0
        mask = np.zeros_like(x)

        ignore_mask_color = [0,0,255]

        cv2.fillPoly(mask, vertices, ignore_mask_color)

        indices = mask[:,:,2] == 255

        x[:,:,2][indices] = x[:,:,2][indices]*random_brightness

        x = cv2.cvtColor(x,cv2.COLOR_HSV2RGB)
        return x
