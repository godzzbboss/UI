# -*- coding: utf-8 -*-

"""
__author__ = "BigBrother"

"""
import matlab.engine as engine
import matlab
import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat
import os
# print(os.getcwd())
# exit()
train_data_x_path = "./data/data_train_x.csv"
train_data_y_path = "./data/data_train_y.csv"

with open(train_data_x_path) as f:
    train_x = pd.read_csv(f, header=None, engine="c", float_precision="round_trip")
with open(train_data_y_path) as f:
    train_y = pd.read_csv(f, header=None, engine="c", float_precision="round_trip")

train_x = np.array(train_x)
train_y = np.array(train_y)

# 将numpy数组转换为mat格式
# savemat("./data/train_x.mat", mdict={"train_x":train_x})
# train_x = loadmat("./data/train_x")["train_x"]
# print(type(train_x))

eng = engine.start_matlab()
eng.cd(os.getcwd(), nargout=0)
# a = eng.test1()
train_x_path = eng.char("./data/train_x.mat")
# print(train_x_path)
# exit()
k = eng.getK(train_x_path,train_x_path)
print(np.array(k))