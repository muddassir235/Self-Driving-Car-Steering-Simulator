import csv
import scipy.misc
from random import shuffle

class data_handler(object):
    def __init__(self, validation_split = 0.2, batch_size = 128, left_and_right_images = False, root_path = '', left_right_offset = 0.2):

        # Name of file where metadata is present
        filename = 'driving_log.csv'

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

        # shuffling the metadata
        shuffle(self.metadata)

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

        # root path of images
        self.root_path = root_path

    def load_next_train_batch(self):
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

        # load images and steering angles for current batch
        for j in range(start,end,1):
            X_train.append(scipy.misc.imresize(scipy.misc.imread(self.root_path+self.metadata_train[j][0])[-150:], [66, 200]) / 255.0)
            y_train.append([float(self.metadata_train[j][3])])
            if(self.left_and_right_images):
                X_train.append(scipy.misc.imresize(scipy.misc.imread(self.root_path+self.metadata_train[j][1][1:])[-150:], [66, 200]) / 255.0)
                y_train.append([float(self.metadata_train[j][3])+self.left_right_offset])
                X_train.append(scipy.misc.imresize(scipy.misc.imread(self.root_path+self.metadata_train[j][2][1:])[-150:], [66, 200]) / 255.0)
                y_train.append([float(self.metadata_train[j][3])-self.left_right_offset])

        # incrementing step
        self.step_train = self.step_train + 1

        return (X_train, y_train)

    def load_next_validation_batch(self):
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

        # laod images and steering angles for current batch
        for j in range(start,end):
            X_val.append(scipy.misc.imresize(scipy.misc.imread(self.root_path+self.metadata_val[j][0])[-150:], [66,200]) / 255.0)
            y_val.append([float(self.metadata_val[j][3])])
            if(self.left_and_right_images):
                X_train.append(scipy.misc.imresize(scipy.misc.imread(self.root_path+self.metadata_train[j][1][1:])[-150:], [66, 200]) / 255.0)
                y_train.append([float(self.metadata_train[j][3])+self.left_right_offset])
                X_train.append(scipy.misc.imresize(scipy.misc.imread(self.root_path+self.metadata_train[j][2][1:])[-150:], [66, 200]) / 255.0)
                y_train.append([float(self.metadata_train[j][3])-self.left_right_offset])

        # incrementing step
        self.step_val = self.step_val + 1

        return (X_val, y_val)

    def move_to_start_train(self):
        self.step_train = 0

    def move_to_start_val(self):
        self.step_val = 0

    def num_train_batches(self):
        return int(len(self.metadata_train) / self.batch_size)

    def num_val_batches(self):
        return int(len(self.metadata_val) / self.batch_size)
