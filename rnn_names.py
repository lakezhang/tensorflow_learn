#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import os
from TFModel import TFModel

class RNNNames(TFModel) :
    def __init__(self, sess, epochs, batch_size, is_training, learning_rate, model_name, checkpoint_dir, sum_dir,
                num_steps, dropout, rnn_units, rnn_layers_num, embedding_dims) :
        '''
            @param sess : tensorflow session
        '''
        super(RNNNames, self).__init__(sess, epochs, batch_size, is_training, learning_rate, model_name, checkpoint_dir, sum_dir)
        self.num_steps = num_steps
        self.dropout = dropout
        self.embedding_dims = embedding_dims
        self.rnn_units = rnn_units
        self.rnn_layers_num = rnn_layers_num

        self.load_data()
        self.build_model()

    def load_data(self) :
        start_char = chr(1)
        end_char = chr(2)
        oov_char = chr(3)

        input_names = []
        vocab = set([start_char, end_char, oov_char])

        with open('./data/origin_names') as fp :
            for line in fp :
                line = line.strip().decode('utf-8')
                if len(line) > 3 or len(line) < 2 :
                    continue

                input_names.append('%s%s%s' % (start_char, line, end_char))
                vocab |= set(line)

        idchar = dict((i, c) for i, c in enumerate(vocab))
        charid = dict((c, i) for i, c in enumerate(vocab))
        self.vocab_size = len(vocab)
        self.num_steps = 4

        self.x_train = np.zeros((len(input_names), self.num_steps))
        self.y_train = np.zeros((len(input_names), self.num_steps))

        for i, name in enumerate(input_names) :
            for j, char in enumerate(name[:-1]) :
                self.x_train[i][j] = charid[char]
            for j, char in enumerate(name[1:]) :
                self.y_train[i][j] = charid[char]

        '''
        #one-hot representation
        self.x_train = np.zeros((len(input_names), self.num_steps, self.vocab_size), dtype=np.bool)
        self.y_train = np.zeros((len(input_names), self.num_steps, self.vocab_size), dtype=np.bool)
        for i, name in enumerate(input_names) :
            for j, char in enumerate(name[:-1]) :
                self.x_train[i][j][charid[char]] = 1
            for j, char in enumerate(name[1:]) :
                self.y_train[i][j][charid[char]] = 1
        '''
        self.start_char = start_char
        self.end_char = end_char
        self.charid = charid
        self.idchar = idchar

    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def predict(self) :
        if self.load() :
            for i in range(10) :
                x_str = [self.start_char]
                predict = []
                while True :
                    x = np.zeros((1, self.num_steps))
                    for j, c  in enumerate(x_str) :
                        x[0][j] = self.charid[c]

                    pos = len(x_str) - 1
                    logits = self.sess.run(self.logits, feed_dict={self.x : x})
                    probability = logits[0][pos]
                    probability = np.exp(probability) / np.sum(np.exp(probability))
                    id_p = self.sample(probability, 1.0)
                    predict.append(self.idchar[id_p])
                    x_str.append(predict[-1])

                    if predict[-1] == self.end_char :
                        #print "END" ,''.join(predict).encode('utf-8'), len(predict)
                        #break
                        pass

                    if len(x_str) > self.num_steps :
                        print ''.join(predict).encode('utf-8'), len(predict)
                        break

    def build_model(self) :
        self.x = tf.placeholder(tf.int32, [None, self.num_steps], name='x')
        self.y = tf.placeholder(tf.int32, [None, self.num_steps], name='y')

        '''
        with tf.name_scope("Train"):
            initializer = tf.random_uniform_initializer(-1.0, 1.0)
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                self.build_rnn_model(True)
        '''
        self.build_rnn_model(self.is_training)
        self.saver = tf.train.Saver()

    def build_rnn_model(self, is_training) :
        cell = lambda : tf.contrib.rnn.BasicLSTMCell(num_units=self.rnn_units, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        drop_cell = cell
        if is_training :
            drop_cell = lambda : tf.contrib.rnn.DropoutWrapper(cell(), output_keep_prob=1 - self.dropout, input_keep_prob=1.0, state_keep_prob=1.0)

        cells = tf.contrib.rnn.MultiRNNCell([drop_cell() for _ in range(self.rnn_layers_num)], state_is_tuple=True)
        self.init_state = cells.zero_state(self.batch_size, tf.float32)

        embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_dims], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, self.x)
        #inputs :  (self.batch_size, self.num_steps, self.embedding_dims)
        #rnn_output: (self.batch_size, self.num_steps, self.rnn_units)
        rnn_output, state = tf.nn.dynamic_rnn(cells, inputs, time_major=False, initial_state=self.init_state)
        rnn_output = tf.reshape(rnn_output, [-1, self.rnn_units])

        '''
        rnn_output = []
        state = self.init_state
        with tf.variable_scope("RNN"):
          for time_step in range(self.num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cells(inputs[:, time_step, :], state)
            rnn_output.append(cell_output)
        rnn_output = tf.reshape(tf.stack(axis=1, values=rnn_output), [-1, self.rnn_units])
        '''

        dense1 = tf.layers.dense(inputs=rnn_output, units=self.vocab_size, activation=None, name='dense1')
        self.logits = tf.reshape(dense1, [self.batch_size, self.num_steps, self.vocab_size])
        self.loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
            self.logits, self.y,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False, average_across_batch=True))
        self.final_state = state

    def train(self) :
        x_train = self.x_train
        y_train = self.y_train

        tf.summary.scalar('loss', self.loss)
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.sum_dir, self.sess.graph)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=global_step)

        self.sess.run(tf.global_variables_initializer())
        init_state = self.sess.run(self.init_state)

        for i in range(self.epochs) :
            batch_num = int(x_train.shape[0] / self.batch_size)
            for j in range(batch_num) :
                feed_dict = {self.x : x_train[j*self.batch_size : j*self.batch_size + self.batch_size], self.y : y_train[j*self.batch_size : j*self.batch_size + self.batch_size]}
                start_time = time.time()
                _, loss, step = self.sess.run([self.train_op, self.loss, global_step], feed_dict=feed_dict)
                duration = time.time() - start_time

                if step % 50 == 0 :
                    summary_str = self.sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    print('Epoch %d Step %d: loss = %.2f (%.3f sec)' % (i, step, loss, duration))

            if i % 10 == 0 :
                self.save(i)


if __name__ == '__main__' :
    flags = tf.app.flags

    #flags.DEFINE_integer("batch_size", 16, "The size of batch images [16]")
    #flags.DEFINE_boolean("is_training", True, "True for training, False for testing [False]")

    flags.DEFINE_integer("batch_size", 1, "The size of batch images [16]")
    flags.DEFINE_boolean("is_training", False, "True for training, False for testing [False]")

    flags.DEFINE_integer("epochs", 100, "Epoch to train [10]")
    flags.DEFINE_float("learning_rate", 0.001, "Learning rate [0.01]")
    flags.DEFINE_string("checkpoint_dir", "./model/names", "Directory name to save the checkpoints [./model/mnist]")
    flags.DEFINE_string("sum_dir", "./summary/names", "Directory name to save the summarys [./summary/mnist]")
    flags.DEFINE_string("model_name", "names", "")
    flags.DEFINE_integer("embedding_dims", 50, "")
    flags.DEFINE_integer("num_steps", 4, "")
    flags.DEFINE_integer("dropout", 0.2, "")
    flags.DEFINE_integer("rnn_units", 128, "")
    flags.DEFINE_integer("rnn_layers_num", 2, "")

    FLAGS = flags.FLAGS

    with tf.Session() as sess:
        name = RNNNames(sess, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
            is_training=FLAGS.is_training, learning_rate=FLAGS.learning_rate,
            model_name=FLAGS.model_name, checkpoint_dir=FLAGS.checkpoint_dir,
            sum_dir=FLAGS.sum_dir, embedding_dims=FLAGS.embedding_dims,
            num_steps=FLAGS.num_steps, dropout=FLAGS.dropout,
            rnn_units=FLAGS.rnn_units, rnn_layers_num=FLAGS.rnn_layers_num
            )

        #name.train()
        name.predict()
