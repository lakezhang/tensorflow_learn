#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import os
from TFModel import TFModel
from tensorflow.examples.tutorials.mnist import input_data

class RNNMnist(TFModel) :
    def __init__(self, sess, epochs, batch_size, is_training, learning_rate, model_name, checkpoint_dir, sum_dir,
                max_steps) :
        '''
            @param sess : tensorflow session
        '''
        super(RNNMnist, self).__init__(sess, epochs, batch_size, is_training, learning_rate, model_name, checkpoint_dir, sum_dir)
        self.max_steps = max_steps
        self.build_model()

    def build_model(self) :
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')
        self.build_dynamic_model()
        self.saver = tf.train.Saver()

    def build_static_model(self) :
        # Current data input shape: (batch_size, n_steps, n_input)
        images = tf.reshape(self.x, (-1, 28, 28))
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        images = tf.unstack(images, num=28, axis=1)

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=128, forget_bias=1.0, state_is_tuple=True)
        output, state = tf.nn.static_rnn(cell, images, dtype=tf.float32)

        self.logits = tf.layers.dense(inputs=output[-1], units=10, name='logits')
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.logits))

    def build_dynamic_model(self) :
        # Current data input shape: (batch_size, n_steps, n_input)
        images = tf.reshape(self.x, (-1, 28, 28))
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=128, forget_bias=1.0, state_is_tuple=True)
        # 初始化为全0 state
        #init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        # 如果 inputs 为 (batches, steps, inputs) ==> time_major=False
        # 如果 inputs 为 (steps, batches, inputs) ==> time_major=True
        output, state = tf.nn.dynamic_rnn(cell, images, time_major=False, dtype=tf.float32)
        # output shape is as same as inputs
        # reshape to get the result of the last step
        output = tf.unstack(tf.transpose(output, [1, 0, 2]))
        self.logits = tf.layers.dense(inputs=output[-1], units=10, name='logits')
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.logits))


    def evaluate(self) :
        '''
            calculate accuracy
        '''
        return self.softmax_accuracy()

    def train(self) :
        '''
            Train the model
        '''
        mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(self.loss, global_step=global_step)

        tf.summary.scalar('loss', self.loss)
        self.sess.run(tf.global_variables_initializer())

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.sum_dir, self.sess.graph)

        while True:
            start_time = time.time()
            batch = mnist.train.next_batch(self.batch_size)
            _, loss, accuracy, step = self.sess.run([train_op, self.loss, self.evaluate(), global_step], feed_dict={self.x : batch[0], self.y : batch[1]})
            duration = time.time() - start_time

            if step % 50 == 0 :
                print('EPOCH %d Step %d : loss = %.2f accuracy = %.2f (%.3f sec)' % (mnist.train.epochs_completed, step, loss, accuracy, duration))
                # Update the events file.
                summary_str = self.sess.run(summary, feed_dict={self.x : batch[0], self.y : batch[1]})
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if step % 500 == 0 :
                accuracy = self.sess.run(self.evaluate(), feed_dict={self.x : mnist.test.images, self.y : mnist.test.labels})
                print('TEST %d: accuracy = %.2f' % (step, accuracy))
                self.save(step)

            if mnist.train.epochs_completed >= self.epochs or step >= self.max_steps :
                self.save(step)
                break

if __name__ == '__main__' :
    flags = tf.app.flags
    flags.DEFINE_integer("batch_size", 50, "The size of batch images [32]")
    flags.DEFINE_integer("epochs", 20, "Epoch to train [10]")
    flags.DEFINE_float("learning_rate", 0.01, "Learning rate [0.01]")
    flags.DEFINE_string("checkpoint_dir", "./model/rnn_mnist", "Directory name to save the checkpoints [./model/mnist]")
    flags.DEFINE_string("sum_dir", "./summary/rnn_mnist", "Directory name to save the summarys [./summary/mnist]")
    flags.DEFINE_boolean("is_training", False, "True for training, False for testing [False]")
    flags.DEFINE_string("model_name", "rnn_mnist", "")
    flags.DEFINE_integer("max_steps", 20000, "Max steps to train [2000]")

    FLAGS = flags.FLAGS

    with tf.Session() as sess:
        rnn = RNNMnist(sess=sess, batch_size=FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate,
            is_training=FLAGS.is_training,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sum_dir=FLAGS.sum_dir,
            epochs=FLAGS.epochs,
            model_name=FLAGS.model_name,
            max_steps=FLAGS.max_steps)
        rnn.show_all_variables()
        rnn.train()
