#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import os
from keras.datasets import imdb
from keras.preprocessing import sequence

class FastText(object) :
    '''
        fast text implemented by tensorflow
        https://arxiv.org/abs/1607.01759
    '''
    def __init__(self, sess, embedding_dims, vocabulary_size, max_len, ngram, batch_size, epochs, sum_dir) :
        '''
        @param sess : tensorflow session
        @param sess : embedding dimension
        @param vocabulary_size : vocabulary size
        @param max_len : max length of inputs
        @param ngram : uni-gram bi-gram or tri-gram eg. 1 or 2 or 3
        '''
        self.sess = sess
        self.embedding_dims = embedding_dims
        self.vocabulary_size = vocabulary_size
        self.max_len = max_len
        self.ngram = ngram
        self.batch_size = batch_size
        self.epochs = epochs
        self.sum_dir = sum_dir
        self.saver = tf.train.Saver()
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

    def evaluate(self) :
        '''
            calculate accuracy
        '''
        correct_prediction = tf.equal(tf.cast(tf.round(tf.sigmoid(self.logits)), tf.int32), tf.cast(self.y, tf.int32))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self) :

        print('Loading data...')
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.vocabulary_size)
        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')

        print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
        print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))
        y_train = np.reshape(y_train, (-1, 1))
        y_test = np.reshape(y_test, (-1, 1))
        x_train = sequence.pad_sequences(x_train, maxlen=self.max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=self.max_len)
        print('x_train shape:', x_train.shape)
        print('y_train shape:', y_train.shape)
        print('x_test shape:', x_test.shape)

        tf.summary.scalar('loss', self.loss)
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.sum_dir, self.sess.graph)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, global_step=global_step)

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

if __name__ == '__main__' :
    with tf.Session() as sess:
        fast = FastText(sess, embedding_dims=50, vocabulary_size=20000, max_len=400, ngram=1, batch_size=32, epochs=10, sum_dir='./summary/fastext')
        fast.train()
