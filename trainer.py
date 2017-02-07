import os
import tensorflow as tf
import model
from simulation_data import data_handler
import numpy as np

class trainer(object):

    def __init__(self, epochs = 10, batch_size = 128, validation_split = 0.2, tune_model = True, L2NormConst = 0.001, left_and_right_images = False, left_right_offset = 0.2, root_path = '', test_root_path ='',stop_gradient_at_conv = False, test_left_and_right_images = False):
        self.handler = data_handler(validation_split = validation_split, batch_size = batch_size, root_path = root_path, left_and_right_images = left_and_right_images, left_right_offset = left_right_offset, test_root_path = test_root_path, test_left_and_right_images = False)
        self.validation_split = validation_split

        self.LOGDIR = './save'

        self.sess = tf.InteractiveSession()

        self.L2NormConst = L2NormConst

        self.train_vars = tf.trainable_variables()

        self.loss = tf.reduce_mean(tf.square(tf.sub(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in self.train_vars]) * L2NormConst
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.epochs = epochs
        self.batch_size = batch_size

        self.sess.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver()

        if(tune_model):
            self.saver.restore(self.sess, "save/model_trained_on_game.ckpt")

        if(stop_gradient_at_conv):
            model.h_fc1 = tf.stop_gradient(model.h_fc1)

    def train(self):
        # train over the dataset about 30 times

        for epoch in range(self.epochs):
            batches_handled = 0

            for X_train, y_train in self.handler.generate_train_batch():
                self.train_step.run(feed_dict={model.x: X_train, model.y_: y_train, model.keep_prob: 0.8})

                if not os.path.exists(self.LOGDIR):
                  os.makedirs(self.LOGDIR)
                checkpoint_path = os.path.join(self.LOGDIR, "model_trained_on_game.ckpt")
                filename = self.saver.save(self.sess, checkpoint_path)

                batches_handled = batches_handled + 1
                if(batches_handled>self.handler.num_train_batches()):
                    break

            avg_train_loss = 0
            batches_handled = 0

            for X_train, y_train in self.handler.generate_train_batch():
                avg_train_loss = (avg_train_loss*batches_handled + self.loss.eval(feed_dict={model.x: X_train, model.y_: y_train, model.keep_prob: 1.0}))/(batches_handled+1)

                batches_handled = batches_handled + 1
                if(batches_handled>self.handler.num_train_batches()):
                    break

            avg_val_loss = 0
            batches_handled = 0

            for X_val, y_val in self.handler.generate_validation_batch():
                avg_val_loss = (avg_val_loss*batches_handled + self.loss.eval(feed_dict={model.x: X_val, model.y_: y_val, model.keep_prob: 1.0}))/(batches_handled+1)

                batches_handled = batches_handled + 1
                if(batches_handled>self.handler.num_val_batches()):
                    break

            print("Model saved in %s. Metrics::: Epoch: %d, Loss: %g, Validation_loss: %g" % (filename, epoch, avg_train_loss, avg_val_loss))

        print("Run the command line:\n" \
                      "--> tensorboard --logdir=./logs " \
                      "\nThen open http://0.0.0.0:6006/ into your web browser")

    def test(self):
        avg_test_loss = 0
        batches_tested = 0
        for X_test, y_test, num_batches in self.handler.generate_test_batch():
            avg_test_loss = (avg_test_loss*batches_tested + self.loss.eval(feed_dict={model.x: X_test, model.y_: y_test, model.keep_prob: 1.0}))/(batches_tested+1)

            batches_tested = batches_tested + 1
            if(batches_tested>num_batches):
                break

        print("Test_loss: %g" % (avg_test_loss))

    def set_root_image_path(self,path):
        self.handler.set_root_image_path(path)
