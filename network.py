#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : network.py
# @Author: nanzhi.wang
# @Date  : 2018/9/19 上午11:08
import tensorflow as tf
import numpy as np


class network:

    def __init__(self,config):

        self.input_dim=config['input_dim']
        self.hidden_dic=config['hidden_dic']
        self.MAX_GRAD_NORM=config['MAX_GRAD_NORM']
        self.LR=config['LR']
        self.build_gragh()

    def build_gragh(self):
        self.input=tf.placeholder(dtype=tf.float32,shape=[None,None],name='input')
        self.y_std=tf.placeholder(dtype=tf.float32,shape=[None],name='y_std')

        with tf.variable_scope('hidden_layer'):
            self.w1=tf.get_variable('w1',shape=[self.input_dim,self.hidden_dic])
            self.b1=tf.get_variable('b1',shape=[self.hidden_dic])
            self.w2 = tf.get_variable('w2', shape=[self.hidden_dic, self.hidden_dic])
            self.b2 = tf.get_variable('b2', shape=[self.hidden_dic])
        with tf.variable_scope('output_layer'):
            self.w3=tf.get_variable('w2', shape=[self.hidden_dic,1])
            self.b3=tf.get_variable('b3',shape=[1])

        self.y_pre=self.forword()
        self.loss=self.computer_loss(self.y_pre,self.y_std)
        self.train_op=self._train(self.loss)

    def forword(self):

        # s1=tf.nn.relu(tf.matmul(self.input,self.w1))+self.b1
        # s2=tf.nn.relu(tf.matmul(s1,self.w2))+self.b2

        output=tf.matmul(self.input,self.w3)

        return output

    def _train(self,loss):
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.MAX_GRAD_NORM)
        opt = tf.train.GradientDescentOptimizer(learning_rate=self.LR)
        train_op = opt.apply_gradients(zip(grads, trainable_variables))
        return train_op

    def computer_loss(self,prediction,truth):
        prediction = tf.reshape(prediction, shape=[-1, 1])
        truth = tf.reshape(truth, shape=[-1, 1])

        # loss = -tf.reduce_sum((truth * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)) + (1 - truth)
        #                        * tf.log(tf.clip_by_value(1 - prediction, 1e-10, 1.0))))

        loss=-tf.reduce_sum(prediction*truth)

        return loss

    def train_on_batch(self,sess,feed_dict_tmp):
        feed_dict = {self.input:feed_dict_tmp['input'],
                     self.y_std:feed_dict_tmp['y_std']
                     }
        loss,_=sess.run([self.loss,self.train_op],feed_dict=feed_dict)
        return loss

    def predict_on_batch(self,sess,feed_dict_tmp):
        feed_dict = {self.input: feed_dict_tmp['input']
                     }
        y_pre, = sess.run([self.y_pre], feed_dict=feed_dict)
        y_pre=np.reshape(y_pre,[len(y_pre)])
        return y_pre


