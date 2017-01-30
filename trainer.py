import os
import tensorflow as tf
import model
from simulation_data import data_handler

class trainer(object):

    def __init__(self, epochs = 10, batch_size = 128, validation_split = 0.2, tune_model = True, L2NormConst = 0.001, left_and_right_images = False, left_right_offset = 0.2, root_path = ''):
        self.handler = data_handler(validation_split = validation_split, batch_size = batch_size, root_path = root_path, left_and_right_images = left_and_right_images, left_right_offset = left_right_offset)
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

        model.h_fc1 = tf.stop_gradient(model.h_fc1)

    def train(self):
        # train over the dataset about 30 times
        for epoch in range(self.epochs):
          for i in range(self.handler.num_train_batches()):
            batch = self.handler.load_next_train_batch()

            self.train_step.run(feed_dict={model.x: batch[0], model.y_: batch[1], model.keep_prob: 0.8})

            if not os.path.exists(self.LOGDIR):
              os.makedirs(self.LOGDIR)
            checkpoint_path = os.path.join(self.LOGDIR, "model_trained_on_game.ckpt")
            filename = self.saver.save(self.sess, checkpoint_path)

          avg_train_loss = 0
          for i in range(self.handler.num_train_batches()):
            batch = self.handler.load_next_train_batch()
            avg_train_loss = (avg_train_loss*i + self.loss.eval(feed_dict={model.x: batch[0], model.y_: batch[1], model.keep_prob: 1.0}))/(i+1)

          avg_val_loss = 0

          if(not self.validation_split == 0):
            for i in range(self.handler.num_val_batches()):
              batch = self.handler.load_next_validation_batch()
              avg_val_loss = (avg_val_loss*i + self.loss.eval(feed_dict={model.x: batch[0], model.y_: batch[1], model.keep_prob: 1.0}))/(i+1)

          print("Model saved in %s. Metrics::: Epoch: %d, Loss: %g, Validation_loss: %g" % (filename, epoch, avg_train_loss, avg_val_loss))

        print("Run the command line:\n" \
                  "--> tensorboard --logdir=./logs " \
                  "\nThen open http://0.0.0.0:6006/ into your web browser")
