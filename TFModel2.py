#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import os

class TFModel2(object) :
    '''
        super class
    '''
    def __init__(self, sess, batch_size, is_training, learning_rate, model_name, checkpoint_dir) :
        '''
            @param sess : tensorflow session
        '''
        self.sess = sess
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.is_training = is_training
        self.model_name = model_name

    def build_model(self) :
        '''
            build modle, should be rewrite
        '''
        # inputs
        self.x = None
        # labels
        self.y = None
        # model ouputs
        self.logits = None
        # loss function
        self.loss = None

    def sigmoid_accuracy(self) :
        '''
            accuracy calculate
            tf.nn.sigmoid_cross_entropy_with_logits
        '''
        correct_prediction = tf.equal(tf.cast(tf.round(tf.sigmoid(self.logits)), tf.int32), tf.cast(self.y, tf.int32))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def softmax_accuracy(self) :
        '''
            accuracy calculate
            tf.losses.softmax_cross_entropy
        '''
        correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.y,1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def softmax_accuracy_rnn(self) :
        '''
            accuracy calculate
            tf.losses.softmax_cross_entropy
            self.logits : [batch_size, num_steps, yclass_num] probability flot64
            self.y :[batch_size, num_steps]  int32
        '''
        correct_prediction = tf.equal(tf.argmax(self.logits, 2), tf.cast(self.y, 'int64'))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def save(self, step):
        '''
            Save the model for use later
        '''
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saver.save(self.sess,
              os.path.join(self.checkpoint_dir, self.model_name),
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
