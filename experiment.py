#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : experiment.py
# @Author: nanzhi.wang
# @Date  : 2018/9/19 上午11:47
'''
这个实验告诉我两个东西，
第一个是基于向量相乘的分类器其实找的也是比较优的决策平面，并不存在分类平面特别偏的情况

第二个是为什么多分类或者二分类要使用softmax或者sigmoid函数，是为了防止梯度爆炸。

还有一个问题，为什么激活函数可以非线性。

'''
from generate_data import data
from network import network
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def dis_data(input,y):
    input_0=[]
    input_1=[]
    for idx,each in enumerate(input):
        if y[idx]==0:
            input_0.append(each)
        else:
            input_1.append(each)
    return np.array(input_0),np.array(input_1)

data=data(10)



config={'input_dim':2,
'hidden_dic':2,
 'MAX_GRAD_NORM':5,
'LR':0.1,
}

network=network(config)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(300):
        input, y_std = data.next_batch(10)
        feed_dic={'input':input,
                  'y_std':y_std}
        print(sess.run(network.w3))
        print(network.train_on_batch(sess,feed_dic))


    input_test, y_test = data.get_test_data(1000)



    y_pre=network.predict_on_batch(sess,feed_dict_tmp={'input':input_test})

    input_0,input_1=dis_data(input_test,y_test)

    plt.scatter(input_0[:,0], input_0[:,1], s=2,c='r')

    plt.scatter(input_1[:,0], input_1[:,1], s=2, c='b')

    plt.show()





