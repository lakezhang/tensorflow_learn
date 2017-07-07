#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.examples.tutorials.mnist import input_data

class CNNMnist(object) :
    '''
        CNN Model for Mnist Classification
    '''
    def __init__(self, sess, batch_size, learning_rate, is_training, checkpoint_dir, sum_dir, epoch, max_steps) :
        #tensorflow session
        self.sess = sess
        self.is_training = is_training
        #model save path
        self.checkpoint_dir = checkpoint_dir
        #summary save path, for tensorboard
        self.sum_dir = sum_dir
        self.batch_size = batch_size
        self.epoch = epoch
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.build_model()

    def build_model(self) :
        '''
            build cnn network
        '''
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')
        # set drop_rate to 0, when validating or testing
        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')

        images = tf.reshape(self.x, [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(inputs=images, filters=32, kernel_size=(5, 5),
                strides=(1, 1), padding='same', activation=tf.nn.relu,
                trainable=self.is_training, name='conv1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2), name='pool1')

        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=(5, 5),
                strides=(1, 1), padding='same', activation=tf.nn.relu,
                trainable=self.is_training, name='conv2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(2, 2), name='pool2')

        #flatten_pool2 = tf.reshape(pool2, [self.batch_size, -1])
        flatten_pool2 = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense1 = tf.layers.dense(inputs=flatten_pool2, units=1024, activation=tf.nn.relu,
                trainable=self.is_training,name='dense1')

        dropout1 = tf.layers.dropout(inputs=dense1, rate=self.drop_rate, name='drop1')
        self.logits = tf.layers.dense(inputs=dropout1, units=10, trainable=self.is_training,name='logits')
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.logits))
        self.saver = tf.train.Saver()

    def evaluate(self) :
        '''
            calculate accuracy
        '''
        correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.y,1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self) :
        '''
            Train the model
        '''
        mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(self.loss, global_step=global_step, var_list=self.t_vars)

        tf.summary.scalar('loss', self.loss)
        self.sess.run(tf.global_variables_initializer())

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.sum_dir, self.sess.graph)

        while True:
            start_time = time.time()
            batch = mnist.train.next_batch(self.batch_size)
            _, loss, accuracy, step = self.sess.run([train_op, self.loss, self.evaluate(), global_step], feed_dict={self.x : batch[0], self.y : batch[1], self.drop_rate : 0.4})
            duration = time.time() - start_time

            if step % 50 == 0 :
                print('Step %d: loss = %.2f accuracy = %.2f (%.3f sec)' % (step, loss, accuracy, duration))
                # Update the events file.
                summary_str = self.sess.run(summary, feed_dict={self.x : batch[0], self.y : batch[1]})
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if step % 100 == 0 :
                accuracy = self.sess.run(self.evaluate(), feed_dict={self.x : mnist.test.images, self.y : mnist.test.labels, self.drop_rate : 0.0})
                print('TEST %d: accuracy = %.2f' % (step, accuracy))

            if mnist.train.epochs_completed >= self.epoch or step >= self.max_steps :
                self.save(step)
                break

    def predict(self) :
        '''
            Load the model and predict
        '''
        could_load = self.load()
        if could_load:
          print(" Model Load SUCCESS")
          mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

          #logits, accuracy = self.sess.run([self.logits, self.evaluate()], feed_dict={self.x : mnist.test.images, self.y : mnist.test.labels, self.drop_rate : 0.0})
          #print('TEST accuracy = %.2f' % (accuracy,))
          #print np.argmax(logits[:10,:], 1)
          #print np.argmax(mnist.test.labels[:10, :], 1)

          logits = self.sess.run(self.logits, feed_dict={self.x : mnist.test.images, self.drop_rate : 0.0})
          print np.argmax(logits[:10,:], 1)

        else:
          print(" Model Load failed...")

    def save(self, step):
        '''
            Save the model for use later
        '''
        model_name = "CNNMnist.model"

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saver.save(self.sess,
              os.path.join(self.checkpoint_dir, model_name),
              global_step=step)

    def load(self):
        '''
            Load the Model
        '''
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(" Success to load model")
            return True
        else:
            print(" Failed to find a checkpoint")
            return False

    def show_all_variables(self) :
        '''
            Show all variables for training
        '''
        import tensorflow.contrib.slim as slim
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

if __name__ == '__main__' :
    flags = tf.app.flags
    flags.DEFINE_integer("batch_size", 50, "The size of batch images [50]")
    flags.DEFINE_integer("epoch", 10, "Epoch to train [10]")
    flags.DEFINE_float("learning_rate", 0.01, "Learning rate of for SGD [0.01]")
    flags.DEFINE_string("checkpoint_dir", "./model", "Directory name to save the checkpoints [./model]")
    flags.DEFINE_string("sum_dir", "./summary", "Directory name to save the summarys [./summary]")
    flags.DEFINE_boolean("is_training", False, "True for training, False for testing [False]")
    flags.DEFINE_integer("max_steps", 2000, "Max steps to train [2000]")
    FLAGS = flags.FLAGS

    with tf.Session() as sess:
        cnn = CNNMnist(sess=sess, batch_size=FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate,
            is_training=FLAGS.is_training,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sum_dir=FLAGS.sum_dir,
            epoch=FLAGS.epoch,
            max_steps=FLAGS.max_steps)
        #cnn.show_all_variables()
        #cnn.train()

        cnn.predict()
