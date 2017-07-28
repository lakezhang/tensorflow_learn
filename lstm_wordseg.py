#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
数据集 : http://sighan.cs.uchicago.edu/bakeoff2005/

文件列表（File List）
　　在gold目录里包含了测试集标准切分及从训练集中抽取的词表（Contains the gold standard segmentation of the test data along with the training data word lists.）
　　在scripts目录里包含了评分脚本和简单中文分词器（Contains the scoring script and simple segmenter.）
　　在testing目录里包含了未切分的测试数据（Contains the unsegmented test data.）
　　在training目录里包含了已经切分好的标准训练数据（Contains the segmented training data.）
　　在doc目录里包括了bakeoff的一些指南（Contains the instructions used in the bakeoff.）

编码（Encoding Issues）
　　文件包括扩展名”.utf8”则其编码为UTF-8(Files with the extension ".utf8" are encoded in UTF-8 Unicode.)
　　文件包括扩展名”.txt”则其编码分别为（Files with the extension ".txt" are encoded as follows）:
　　前缀为as_，代表的是台湾中央研究院提供，编码为Big Five (CP950)；
　　前缀为hk_，代表的是香港城市大学提供，编码为Big Five/HKSCS；
　　前缀为msr_，代表的是微软亚洲研究院提供，编码为 EUC-CN (CP936)；
　　前缀为pku_，代表的北京大学提供，编码为EUC-CN (CP936)；

以下利用其自带的中文分词工具进行说明。在scripts目录里包含一个基于最大匹配法的中文分词器mwseg.pl，以北京大学提供的人民日报语料库为例，用法如下：
　　./mwseg.pl ../gold/pku_training_words.txt < ../testing/pku_test.txt > pku_test_seg.txt
　　其中第一个参数需提供一个词表文件pku_training_word.txt，输入为pku_test.txt，输出为pku_test_seg.txt。
　　利用score评分的命令如下：
　　./score ../gold/pku_training_words.txt ../gold/pku_test_gold.txt pku_test_seg.txt > score.txt
　　其中前三个参数已介绍，而score.txt则包含了详细的评分结果，不仅有总的评分结果，还包括每一句的对比结果。这里只看最后的总评结果：
'''
import tensorflow as tf
import numpy as np
import time
import os
import random
from keras.datasets import imdb
from keras.preprocessing import sequence
from TFModel2 import TFModel2

class RNNModel(TFModel2) :
    def __init__(self, sess, batch_size, is_training, learning_rate, model_name, checkpoint_dir,
            num_steps, dropout, rnn_units, rnn_layers_num, embedding_dims, vocabulary_size, max_grad_norm) :
        super(RNNModel, self).__init__(sess, batch_size, is_training, learning_rate, model_name, checkpoint_dir)
        #输入长度，input steps
        self.num_steps = num_steps
        #dropout probability
        self.dropout = dropout
        #embedding dimensions
        self.embedding_dims = embedding_dims
        #RNN cell vector length
        self.rnn_units = rnn_units
        #RNN stack Size
        self.rnn_layers_num = rnn_layers_num
        #vocabulary size
        self.vocabulary_size = vocabulary_size
        #max gradients
        self.max_grad_norm = max_grad_norm
        # N S B M E
        self.y_class = 5
        #build model
        self.build_bi_model()

    def build_bi_model(self) :
        self.x = tf.placeholder(tf.int32, [None, self.num_steps])
        self.y = tf.placeholder(tf.int32, [None, self.num_steps])

        W = tf.get_variable("embedding_weights", [self.vocabulary_size, self.embedding_dims], dtype=tf.float32)
        input_embedded = tf.nn.embedding_lookup(W, self.x)

        if self.is_training and self.dropout > 0.0 :
            input_embedded = tf.nn.dropout(input_embedded, keep_prob = 1- self.dropout)

        lstm_cell = lambda : tf.contrib.rnn.BasicLSTMCell(self.rnn_units,
            forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

        drop_cell = lstm_cell
        if self.is_training and self.dropout > 0.0 :
            drop_cell = lambda : tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob = 1 - self.dropout)

        cells_fw = tf.contrib.rnn.MultiRNNCell([drop_cell() for _ in range(self.rnn_layers_num)], state_is_tuple=True)
        self.init_state_fw = cells_fw.zero_state(self.batch_size, tf.float32)

        cells_bw = tf.contrib.rnn.MultiRNNCell([drop_cell() for _ in range(self.rnn_layers_num)], state_is_tuple=True)
        self.init_state_bw = cells_bw.zero_state(self.batch_size, tf.float32)

        #input_embedded :  (self.batch_size, self.num_steps, self.embedding_dims)
        #rnn_output: (self.batch_size, self.num_steps, self.rnn_units)
        rnn_output, state = tf.nn.bidirectional_dynamic_rnn(cells_fw, cells_bw, input_embedded, time_major=False, initial_state_fw=self.init_state_fw, initial_state_bw=self.init_state_bw)
        #get the last step of the rnn_output
        #output = tf.unstack(tf.transpose(rnn_output, [1, 0, 2]))
        output = tf.concat(rnn_output, 2)
        output = tf.reshape(output, [-1, self.rnn_units * 2])
        dense1 = tf.layers.dense(inputs=output, units=self.y_class, name='logits',activation=None, reuse=tf.get_variable_scope().reuse)
        self.logits = tf.reshape(dense1, [self.batch_size, self.num_steps, self.y_class])
        self.loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
            self.logits, self.y,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False, average_across_batch=True))

        if self.is_training :
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_grad_norm)
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            #self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver()

    def build_model(self) :
        self.x = tf.placeholder(tf.int32, [None, self.num_steps])
        self.y = tf.placeholder(tf.int32, [None, self.num_steps])

        W = tf.get_variable("embedding_weights", [self.vocabulary_size, self.embedding_dims], dtype=tf.float32)
        input_embedded = tf.nn.embedding_lookup(W, self.x)

        if self.is_training and self.dropout > 0.0 :
            input_embedded = tf.nn.dropout(input_embedded, keep_prob = 1- self.dropout)

        lstm_cell = lambda : tf.contrib.rnn.BasicLSTMCell(self.rnn_units,
            forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

        drop_cell = lstm_cell
        if self.is_training and self.dropout > 0.0 :
            drop_cell = lambda : tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob = 1 - self.dropout)

        cells = tf.contrib.rnn.MultiRNNCell([drop_cell() for _ in range(self.rnn_layers_num)], state_is_tuple=True)
        self.init_state = cells.zero_state(self.batch_size, tf.float32)

        #input_embedded :  (self.batch_size, self.num_steps, self.embedding_dims)
        #rnn_output: (self.batch_size, self.num_steps, self.rnn_units)
        rnn_output, state = tf.nn.dynamic_rnn(cells, input_embedded, time_major=False, initial_state=self.init_state)
        #get the last step of the rnn_output
        #output = tf.unstack(tf.transpose(rnn_output, [1, 0, 2]))
        output = tf.reshape(rnn_output, [-1, self.rnn_units])
        dense1 = tf.layers.dense(inputs=output, units=self.y_class, name='logits',activation=None, reuse=tf.get_variable_scope().reuse)
        self.logits = tf.reshape(dense1, [self.batch_size, self.num_steps, self.y_class])
        self.loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
            self.logits, self.y,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False, average_across_batch=True))

        if self.is_training :
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            '''
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_grad_norm)
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            '''
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver()

    def evaluate(self) :
        '''
            calculate accuracy
        '''
        return self.softmax_accuracy_rnn()

class WordSeg(object) :
    def __init__(self, sess, epochs, batch_size, is_training, learning_rate, model_name, checkpoint_dir, sum_dir,
                num_steps, dropout, rnn_units, rnn_layers_num, embedding_dims, vocabulary_size, max_grad_norm) :
        self.sess = sess
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.sum_dir = sum_dir
        self.is_training = is_training
        self.model_name = model_name
        self.num_steps = num_steps
        self.vocabulary_size = vocabulary_size
        # prepare data
        self.prepare_data()
        self.test_batch = 200

        initializer = tf.random_uniform_initializer(-1.0, 1.0)
        if is_training :
            with tf.name_scope("Train"):
                with tf.variable_scope("Model", reuse=None, initializer=initializer) :
                    self.train_model = RNNModel(self.sess, batch_size, True, learning_rate, model_name, checkpoint_dir,
                                num_steps, dropout, rnn_units, rnn_layers_num, embedding_dims, self.vocabulary_size, max_grad_norm)
                    self.train_model.show_all_variables()

            with tf.name_scope("Test"):
                with tf.variable_scope("Model", reuse=True, initializer=initializer) :
                    self.test_model = RNNModel(self.sess, self.test_batch, False, learning_rate, model_name, checkpoint_dir,
                                num_steps, dropout, rnn_units, rnn_layers_num, embedding_dims, self.vocabulary_size, max_grad_norm)

        '''
        else :
            with tf.variable_scope("Model", reuse=None, initializer=initializer) :
                self.preidct_model = RNNModel(self.sess, 1, False, learning_rate, model_name, checkpoint_dir,
                            num_steps, dropout, rnn_units, rnn_layers_num, embedding_dims, self.vocabulary_size, max_grad_norm)
        '''

    def split_tokens(self, tokens) :
        if len(''.join(tokens)) <= self.num_steps :
            return [tokens,]

        sep = [u'。', u'；', u'？', '?', ';', u'，']
        pos_list = []
        for s in sep :
            try :
                pos = tokens.index(s)
                pos_list.append((s, pos))
            except :
                continue

        if len(pos_list) == 0 :
            return [tokens,]

        pos_list.sort(key=lambda v : v[1])

        i = 0
        for (s, pos) in pos_list :
            if pos > self.num_steps :
                break
            i += 1

        pos = pos_list[i - 1][1]
        if pos >= len(tokens) - 1 :
            return [tokens,]

        return self.split_tokens(tokens[:pos + 1]) + self.split_tokens(tokens[pos+2 :])

    def show_sample(self, num) :
        sample = range(len(self.x_train))
        random.shuffle(sample)
        for i in sample[:num] :
            print i, '--------------------'
            for j, cid in enumerate(self.x_train[i]) :
                print '%d:%s:%d' % (cid, self.idchar[cid], self.y_train[i][j])

    def show_result(self, x, logits, num=3, y=None) :
        labels = ['N', 'S', 'B', 'M', 'E']
        p = np.argmax(logits, 2)

        for i, v in enumerate(x) :
            o1 = []
            o2 = []
            for j, t in enumerate(v) :
                o1.append(self.idchar.get(t, chr(1)))
                if y is not None and y[i][j] in (1, 4) :
                    o1.append(' | ')
                o2.append(self.idchar.get(t, chr(1)))
                if p[i][j] in (1, 4) :
                    o2.append(' | ')

            print ''.join(o1)
            print ''.join(o2)

            if i >= num -1 :
                break

    def build_vocab(self, input_file) :
        #chr(0) : PADDING  chr(1) : OOV
        vocab = set([chr(0),chr(1)])
        with open(input_file) as fp :
            for line in fp :
                line = line.strip().decode('utf-8')
                if len(line) < 1 :
                    continue

                vocab |= set(line)

        self.idchar = dict((i, c) for i, c in enumerate(vocab))
        self.charid = dict((c, i) for i, c in enumerate(vocab))
        self.vocabulary_size = len(vocab)

    def build_data(self, input_file) :
        oov_id = self.charid[chr(1)]
        data_x = []
        data_y = []

        with open(input_file) as fp :
            for line in fp :
                line = line.strip().decode('utf-8')
                if len(line) < 1 :
                    continue

                tokens = line.split(' ')
                token_list = self.split_tokens(tokens)

                for x in token_list :
                    one_x = []
                    one_y = []
                    for t in x :
                        if len(one_x) + len(t) > self.num_steps :
                            break

                        if len(t) == 1 :
                            one_x.append(self.charid.get(t, oov_id))
                            one_y.append(1) # SINGLE
                        elif len(t) > 1 :
                            one_x.append(self.charid.get(t[0], oov_id))
                            one_y.append(2) # BEGIN

                            for o in t[1:-1] :
                                one_x.append(self.charid.get(o, oov_id))
                                one_y.append(3) # MIDDLE

                            one_x.append(self.charid.get(t[-1], oov_id))
                            one_y.append(4) # END

                    data_x.append(one_x)
                    data_y.append(one_y)

        data_x = sequence.pad_sequences(data_x, maxlen=self.num_steps, value=self.charid[chr(0)])
        # 0 : PADDING
        data_y = sequence.pad_sequences(data_y, maxlen=self.num_steps, value=0)

        return data_x, data_y

    def prepare_data(self) :
        train_file = './data/icwb2/training/pku_training.utf8'
        test_file = './data/icwb2/gold/pku_test_gold.utf8'

        self.build_vocab(train_file)
        self.x_train, self.y_train = self.build_data(train_file)
        self.x_test, self.y_test = self.build_data(test_file)

    def train(self) :
        x_train = self.x_train
        y_train = self.y_train

        tf.summary.scalar('train_loss', self.train_model.loss)
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.sum_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        for i in range(self.epochs) :
            batch_num = int(x_train.shape[0] / self.batch_size)
            for j in range(batch_num) :
                start_time = time.time()
                feed_dict = {self.train_model.x : x_train[j*self.batch_size : j*self.batch_size + self.batch_size], self.train_model.y : y_train[j*self.batch_size : j*self.batch_size + self.batch_size]}
                _, loss, step = self.sess.run([self.train_model.train_op, self.train_model.loss, self.train_model.global_step], feed_dict=feed_dict)
                duration = time.time() - start_time

                if step % 50 == 0 :
                    summary_str, accuracy = self.sess.run([summary, self.train_model.evaluate()], feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    print('EPOCH: %d Step %d: loss = %.4f accuracy = %.4f (%.3f sec)' % (i, step, loss, accuracy, duration))

            test_sample = np.arange(len(self.x_test))
            np.random.shuffle(test_sample)
            test_sample = test_sample[:self.test_batch]

            x_test = self.x_test[test_sample, :]
            y_test = self.y_test[test_sample, :]

            loss, logits, accuracy = self.sess.run([self.test_model.loss, self.test_model.logits, self.test_model.evaluate()], feed_dict={self.test_model.x : x_test , self.test_model.y : y_test})
            print('EPOCH: %d TEST loss = %.4f accuracy = %.4f' % (i, loss, accuracy))
            self.show_result(x_test, logits, 3, y_test)

            self.train_model.save(i)


if __name__ == '__main__' :
    flags = tf.app.flags
    flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
    flags.DEFINE_integer("epochs", 10, "Epoch to train [10]")
    flags.DEFINE_float("learning_rate", 0.001, "Learning rate [0.01]")
    flags.DEFINE_string("checkpoint_dir", "./model/wordseg", "Directory name to save the checkpoints [./model/mnist]")
    flags.DEFINE_string("sum_dir", "./summary/wordseg", "Directory name to save the summarys [./summary/mnist]")
    flags.DEFINE_boolean("is_training", True, "True for training, False for testing [False]")
    flags.DEFINE_string("model_name", "wordseg", "")
    flags.DEFINE_integer("embedding_dims", 32, "")
    flags.DEFINE_integer("vocabulary_size", 20000, "")

    FLAGS = flags.FLAGS

    with tf.Session() as sess:
        obj = WordSeg(sess, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
            is_training=FLAGS.is_training, learning_rate=FLAGS.learning_rate,
            model_name=FLAGS.model_name, checkpoint_dir=FLAGS.checkpoint_dir,
            sum_dir=FLAGS.sum_dir, embedding_dims=FLAGS.embedding_dims,
            vocabulary_size=FLAGS.vocabulary_size,
            num_steps=32,
            dropout=0.2,
            rnn_units=64,
            rnn_layers_num=2,
            max_grad_norm=6.0
            )

        #obj.show_sample(3)

        obj.train()
