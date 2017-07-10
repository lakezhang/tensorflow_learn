#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import os
from keras.datasets import imdb
from keras.preprocessing import sequence
from TFModel import TFModel

class FastText(TFModel) :
    '''
        fast text implemented by tensorflow
        https://arxiv.org/abs/1607.01759
    '''
    def __init__(self, sess, epochs, batch_size, is_training, learning_rate, model_name, checkpoint_dir, sum_dir,
                embedding_dims, vocabulary_size, max_len, ngram) :
        '''
            @param sess : tensorflow session
            @param embedding_dims : embedding dimension
            @param vocabulary_size : vocabulary size
            @param max_len : max length of inputs
            @param ngram : uni-gram bi-gram or tri-gram eg. 1 or 2 or 3
        '''
        super(FastText, self).__init__(sess, epochs, batch_size, is_training, learning_rate, model_name, checkpoint_dir, sum_dir)
        self.embedding_dims = embedding_dims
        self.vocabulary_size = vocabulary_size
        self.max_len = max_len
        self.ngram = ngram

        self.build_model()

    def build_model(self) :
        '''
            build fasttext model
        '''
        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.y = tf.placeholder(tf.float32, [None, 1])

        W = tf.Variable(
            tf.random_uniform([self.vocabulary_size, self.embedding_dims], -1.0, 1.0), name="embedding_weights")
        embedded = tf.nn.embedding_lookup(W, self.x)
        global_pool = tf.reduce_mean(embedded, axis=1)
        self.logits = tf.layers.dense(inputs=global_pool, units=1, activation=None, name='output')

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits))
        self.saver = tf.train.Saver()

    def evaluate(self) :
        '''
            calculate accuracy
        '''
        return self.sigmoid_accuracy()

    def predict(self) :
        if self.load() :
            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.vocabulary_size)
            y_train = np.reshape(y_train, (-1, 1))
            y_test = np.reshape(y_test, (-1, 1))
            x_train = sequence.pad_sequences(x_train, maxlen=self.max_len)
            x_test = sequence.pad_sequences(x_test, maxlen=self.max_len)

            output = tf.cast(tf.round(tf.sigmoid(self.logits)), tf.int32)
            out = self.sess.run(output, feed_dict={self.x : x_test})
            print out[:5], y_test[:5]

    def train(self) :
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.vocabulary_size)
        y_train = np.reshape(y_train, (-1, 1))
        y_test = np.reshape(y_test, (-1, 1))
        x_train = sequence.pad_sequences(x_train, maxlen=self.max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=self.max_len)

        tf.summary.scalar('loss', self.loss)
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.sum_dir, self.sess.graph)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=global_step)

        self.sess.run(tf.global_variables_initializer())

        for i in range(self.epochs) :
            batch_num = int(x_train.shape[0] / self.batch_size)
            for j in range(batch_num) :
                start_time = time.time()
                _, loss, step = self.sess.run([self.train_op, self.loss, global_step], feed_dict={self.x : x_train[j*self.batch_size : j*self.batch_size + self.batch_size], self.y : y_train[j*self.batch_size : j*self.batch_size + self.batch_size]})
                duration = time.time() - start_time

                if step % 50 == 0 :
                    summary_str, accuracy = self.sess.run([summary, self.evaluate()], feed_dict={self.x : x_train[j*self.batch_size : j*self.batch_size + self.batch_size], self.y : y_train[j*self.batch_size : j*self.batch_size + self.batch_size]})
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    print('Step %d: loss = %.2f accuracy = %.2f (%.3f sec)' % (step, loss, accuracy, duration))

            loss, accuracy = self.sess.run([self.loss, self.evaluate()], feed_dict={self.x : x_test , self.y : y_test})
            print('EPOCH: %d TEST loss = %.2f accuracy = %.2f' % (i, loss, accuracy))
            self.save(i)

if __name__ == '__main__' :
    flags = tf.app.flags
    flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
    flags.DEFINE_integer("epochs", 10, "Epoch to train [10]")
    flags.DEFINE_float("learning_rate", 0.001, "Learning rate [0.01]")
    flags.DEFINE_string("checkpoint_dir", "./model/fasttext", "Directory name to save the checkpoints [./model/mnist]")
    flags.DEFINE_string("sum_dir", "./summary/fasttext", "Directory name to save the summarys [./summary/mnist]")
    flags.DEFINE_boolean("is_training", False, "True for training, False for testing [False]")
    flags.DEFINE_string("model_name", "fasttext", "")

    flags.DEFINE_integer("embedding_dims", 50, "")
    flags.DEFINE_integer("vocabulary_size", 20000, "")
    flags.DEFINE_integer("max_len", 400, "")
    flags.DEFINE_integer("ngram", 1, "")

    FLAGS = flags.FLAGS

    with tf.Session() as sess:
        fast = FastText(sess, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
        is_training=FLAGS.is_training, learning_rate=FLAGS.learning_rate,
        model_name=FLAGS.model_name, checkpoint_dir=FLAGS.checkpoint_dir,
        sum_dir=FLAGS.sum_dir, embedding_dims=FLAGS.embedding_dims,
        vocabulary_size=FLAGS.vocabulary_size, max_len=FLAGS.max_len,
        ngram=FLAGS.ngram)

        #fast.train()
        fast.predict()
