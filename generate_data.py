#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : generate_data.py
# @Author: nanzhi.wang
# @Date  : 2018/9/19 上午11:36
import random as rd
import numpy as np

class data:

    def __init__(self,dis):
        self.dis=dis

    def next_batch(self,size):
        input=[]
        y_std=[]
        while True:
            rd1=(rd.random()-0.5)*2*self.dis
            rd2=(rd.random()-0.5)*2
            if abs(rd1)<self.dis/2:
                continue
            else:
                input.append(np.array([rd1,rd2]))
                if rd1>0:
                    y_std.append(1)
                else:
                    y_std.append(-1)

            if len(y_std)>size:
                break

        return np.array(input),np.array(y_std)

    def get_test_data(self,size):
        input=[]
        y_std=[]
        while True:
            rd1=(rd.random()-0.5)*2*self.dis
            rd2=(rd.random()-0.5)*2
            input.append(np.array([rd1,rd2]))
            if rd1>0:
                y_std.append(1)
            else:
                y_std.append(0)
            if len(y_std)>size:
                break
        return np.array(input),np.array(y_std)


if __name__=='__main__':
    d=data(10)
    print(d.next_batch(1))
