#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:40:15 2019

@author: siamak
"""
import numpy as np
# import gym
import sys
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import  time

from utils import *
from ARSM_Letor_v2 import *
from REINFORCE_Letor import *
from LoadData import *

if __name__ == '__main__':
#    tf.enable_eager_execution()
    
    ###########################   Load the parameter ##########################
    Para = yaml.load(open('Para_info.yml', 'r'))

    version = Para['version']
    model = Para['model']
    dataset = Para['dataset']
    n_feature = Para['Nfeature']
    learning_rate = Para['Learningrate']
    n_episode = Para['Nepisode']
    len_episode = Para['Lenepisode']
    n_hidden_unit = 15
    gamma = 1.0        # reward discount factor
    ######################### Load Data #######################################
    fold = 'Fold5'
    file_name = './MQ/MQ2007/' + fold + '/'
    train_data = LoadData(file_name + 'train.txt', dataset)
    vali_data  = LoadData(file_name + 'vali.txt',  dataset)
    test_data  = LoadData(file_name + 'test.txt',  dataset)
    
    nquery = len(train_data)
    result_file = open('Res_' + dataset + '_' + fold + '_' + \
                      time.strftime("%m%d", time.localtime()),'w')
    result_file.write(yaml.dump(Para) + '\n')
    
    # Initialize the object
    learner = ARSM_Letor(n_hidden_unit, n_feature, learning_rate, len_episode,\
                         gamma, result_file)
    #learner = REINFORCE_Letor(n_hidden_unit, n_feature, learning_rate, len_episode,\
    #                     gamma, result_file)
    
    learner.Eval(train_data, 'train')
    learner.Eval(vali_data , 'vali')
    learner.Eval(test_data , 'test')
    
    ######################## Set Parameters ###################################
    seedi = 0
    tf.set_random_seed(seedi)
    np.random.seed(seedi)
    random.seed(seedi)
    num_epoch = 5000
    '''
    MaxPseudoActionSequences = 16
    Num_ARSM_Ref_Episode = 0 
    # if iter<Num_ARSM_Ref_Episode, then use ARSM single reference estimator 
    # with the true action actegory or a random category as ref 
    TrueActionAsRef = False
    entropy_par = 0.0
    optimizer_actor = tf.train.AdamOptimizer(learning_rate=lr_actor)
    nstep = 3000       # maximum number of true + sudo actions
    n_true = 3000      # maximum number of true actions
    Share_Pi = False
    IsPlot = False
    SaveModel = True
    score_record, entropy_record = [],[]
    pseudo_prop = []
    '''
    for iteration in range(num_epoch):
        seed = np.random.randint(0, 1e+9)
        # Pick a batch of queries
        batch = np.random.randint(nquery, size=n_episode)
        Queryids = []
        for i in batch:
            Queryids.append(list(train_data.keys())[i])
        
        learner.GenEpisodes(Queryids, train_data)
        learner.UpPolicy(train_data, seed)
        learner.Eval(train_data, 'train')
        learner.Eval(vali_data, 'vali')
        learner.Eval(test_data, 'test')