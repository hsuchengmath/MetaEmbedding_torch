



import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os, pickle

# specify the GPU device
# os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'



'''
Config
'''
# batch size per iteration
BATCHSIZE = 200
# mini-batch size for few-shot learning
MINIBATCHSIZE = 20 
# learning rate
LR = 1e-3 
# coefficient to balance `cold-start' and `warm-up'
ALPHA = 0.1
# length of embedding vectors
EMB_SIZE = 128
# model
MODEL = 'deepFM'
# log file
LOG = "logs/{}.csv".format(MODEL)
# path to save the model
saver_path ="saver/model-"+LOG.split("/")[-1][:-4]


minibatchsize = MINIBATCHSIZE
batch_n_ID = 25
batchsize = minibatchsize*batch_n_ID
n_epoch = 3

from data_process import *
from model import Meta_Model
from modeling import Modeling
import torch
from evaluation import Evaluation
#from evaluation import Evaluation
 
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #(or "1" or "2")

'''
ID_col :  MovieID
item_col :  ['Year']
context_col :  ['Age', 'Gender', 'Occupation', 'UserID']
num_words_dict :  {'MovieID': 4000, 'UserID': 6050, 'Age': 7, 'Gender': 2, 'Occupation': 21, 'Year': 83}
'''

# init model 
model = Meta_Model(ID_col, item_col, context_col, num_words_dict, model=MODEL,
                                emb_size=EMB_SIZE, alpha=ALPHA,
                                warm_lr=LR, cold_lr=LR/10., ME_lr=LR)
print(model.deep0_dense_layer)
# modeling 
modeling_obj = Modeling(model, saver_path, BATCHSIZE, test_x_test, test_t_test, test_g_test)
'''
train_x :  <class 'pandas.core.frame.DataFrame'>
   MovieID  Year  Age  Gender  Occupation  UserID
0     1193    57    0       0          10       1
1      661    78    0       0          10       1
2      914    46    0       0          10       1
train_g :  <class 'numpy.ndarray'> (Genres)
[[ 0  0  0  1]
 [ 0 16 10 14]
 [ 0  0 14  5]]
train_y :  <class 'pandas.core.series.Series'>
0         1
1         0 
         ..
765667    1
765668    1
Name: y, Length: 765669, dtype: int8
train_t :  <class 'numpy.ndarray'> (Title)
[[   0    0   53 2415  312    1 2416 2417]
 [   0    0    0  957    5    1  332 1969]
 [   0    0    0    0    0   13  538  217]]
'''

train_MovieID_set = set(train_x['UserID'])
test_MovieID_set_a = set(test_x_a['UserID'])
test_MovieID_set_b = set(test_x_b['UserID'])
test_MovieID_set_c = set(test_x_c['UserID'])
#test_MovieID_set_d = set(test_x_d['UserID'])

print(list(test_MovieID_set_a)[:10])
print(list(test_MovieID_set_b)[:10])
print(list(test_MovieID_set_c)[:10])

print('overlap -a ',len(train_MovieID_set & test_MovieID_set_a))
print('overlap -b ',len(train_MovieID_set & test_MovieID_set_b))
print('overlap -c ',len(train_MovieID_set & test_MovieID_set_c))
#print('overlap -d ',len(train_MovieID_set & test_MovieID_set_d))
 
 
logloss_base_cold, auc_base_cold = modeling_obj.pre_train_base_model(train_x, train_g, train_y, train_t, test_y_test)
modeling_obj.train_meta_embedding_module(n_epoch, test_y_test, ID_col, item_col,context_col)


# evaluation
# test_test_data = \
#    {'test_x_test':test_x_test,
#     'test_t_test':test_t_test,
#     'test_g_test':test_g_test,
#     'test_y_test':test_y_test
#    }

# eval_obj = Evaluation(test_test_data, test_a_data=dict, test_b_data=dict, test_c_data=dict, logloss_base_cold=logloss_base_cold, auc_base_cold=auc_base_cold)
# eval_obj.eval_base_model(model)
# print('-----------------------')
# eval_obj.eval_meta_learning_model(model)


# eval_obj.eval_base_model(save_path, test_x_b, test_t_b, test_g_b, test_y_b, test_t_c, test_g_c, test_y_c)
# eval_obj.eval_meta_learning_model(test_x_b, test_t_b, test_g_b, test_y_b, test_t_c, test_g_c, test_y_c)