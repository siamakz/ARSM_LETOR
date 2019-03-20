#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 08:43:09 2019

@author: siamak
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:34:10 2019

@author: siamak
"""

import numpy as np
# import gym
import sys
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import tensorflow as tf
from arm_util import *
import matplotlib.pyplot as plt
import random
import os
import yaml
from utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Basic object
class REINFORCE_Letor(object):
    
    def __init__(self, n_hidden_unit, n_feature, learningrate, len_episode,\
                 gamma, result_file):
        
        self.n_feature = n_feature
        self.n_hidden_unit = n_hidden_unit
        self.lr = learningrate
        self.len_episode = len_episode
        self.result_file = result_file
        self.n_top = 10
        self.gamma = gamma
        self.memory = []
        self.iteration = 0
    
        
        global scores, input_docs, position, learning_rate, sess, train_step,\
        cross_entropy, grads_vars, prob
        
        input_docs = tf.placeholder(tf.float32, [None, self.n_feature])
        position = tf.placeholder(tf.int64)
        learning_rate = tf.placeholder(tf.float32, shape=[])
        
        # actor model T_{\theta}(s_t)
        ## TODO: MDPRank doesn't have bias term
        '''model_actor = tf.keras.Sequential([tf.keras.layers.Dense(self.n_hidden_unit,\
                    activation = "tanh", input_shape = (self.n_feature,),\
                    kernel_initializer = tf.truncated_normal_initializer(mean=0,\
                    stddev = 0.1/np.sqrt(float(n_feature))),\
                    bias_initializer = tf.constant_initializer(0.0)),\
                    tf.keras.layers.Dense(1,\
                    kernel_initializer = tf.truncated_normal_initializer(mean=0,\
                    stddev = 0.1/np.sqrt(float(self.n_hidden_unit))),\
                    bias_initializer=tf.constant_initializer(0.0))])
        
        #self.optimizer_actor = tf.train.AdamOptimizer(learning_rate = self.lr)
        scores = tf.transpose(model_actor(input_docs))'''
        
        # Generate hidden layer
        W1 = tf.Variable(tf.truncated_normal([self.n_feature, self.n_hidden_unit],\
                                    stddev=0.1 / np.sqrt(float(n_feature))))
        # b1 = tf.Variable(tf.zeros([1, hidden_units]))
        h1 = tf.tanh(tf.matmul(input_docs, W1))
        # Second layer -- linear classifier for action logits
        W2 = tf.Variable(tf.truncated_normal([self.n_hidden_unit, 1],\
                            stddev=0.1 / np.sqrt(float(self.n_hidden_unit))))
        # b2 = tf.Variable(tf.zeros([1]))
        scores = tf.transpose(tf.matmul(h1, W2))  # + b2
        prob = tf.nn.softmax(scores)

        init = tf.global_variables_initializer()
        ## TODO: logits=prob or scores?? 
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=position)  # logits!! 
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        grads_vars = opt.compute_gradients(cross_entropy)
        train_step = opt.apply_gradients(grads_vars)
        
        # Start TF session
        sess = tf.Session()
        sess.run(init)
        
    def GenEpisodes(self, Queryids, Data):
        '''
        This method generates an episode for each query based on current policy
        and calculates the corresponding return.
        '''
        # Average ranking measures for all queries
        thendcg = np.zeros(self.n_top)
        thedcg = np.zeros(self.n_top)
        themap = 0.0
        thealldcg = 0.0

        for queryid in Queryids:
            phi_sequence, pi_sequence = [], []
            QueryInfo = Data[queryid]
            score = sess.run([scores], feed_dict={input_docs: QueryInfo['feature']})[0].reshape([-1])
            #score = self.model_actor(QueryInfo['feature'])#[0]
            # Generate an episode based on current policy for each query
            phi = score#.transpose()[0]
            ndoc = len(phi)
            positions = list(range(ndoc))
            ranklist = np.zeros(ndoc, dtype=np.int32)
            for position in range(ndoc):
                phi_sequence.append(phi)
                
                pi = np.random.exponential(np.ones(len(phi)))
                pi = pi/np.sum(pi)
                pi_sequence.append(pi)
                choice = np.argmin(np.log(pi) - phi)     # action_true
                ranklist[position] = positions[choice]   # actions
                
                phi = np.delete(phi, choice)
                del positions[choice]
            
            rates = QueryInfo['label'][ranklist]
            #reward = GetReturn_DCG(QueryInfo['label'][ranklist])  # reward to-go
            rewards_true = GetReturn_DCG(rates)
            
            self.memory.append({'queryid': queryid, 'score': score,\
                'phi_sequence': phi_sequence, 'pi_sequence': pi_sequence,\
                'reward': rewards_true, 'ranklist': ranklist})
            
            # Accumulate ranking measures
            themap += MAP(rates)
            thendcg += NDCG(self.n_top, rates)
            thedcg += DCG(self.n_top, rates)
            thealldcg += DCG_all(rates)

        nquery = len(Queryids)
        themap = themap / nquery
        thendcg = thendcg / nquery
        thedcg = thedcg / nquery
        thealldcg = thealldcg / nquery
        
        #print('jiayou: ', themap, thealldcg, thendcg[0], thendcg[2], thendcg[4],\
        #      thendcg[9], thedcg[0], thedcg[2], thedcg[4], thedcg[9])

        info = yaml.dump({"ite": self.iteration, "type": type, "MAP": themap,\
                          "Return": thealldcg, "DCG": thedcg, "NDCG": thendcg})+'\n'
        self.result_file.write(info)
        
    def UpPolicy(self, Data, seed):
        '''
        This method performs policy gradient update for each position in the
        ranking, for each query.
        '''
        self.iteration +=1
        
        for item in self.memory:
            queryid = item['queryid']
            rewards_true = item['reward']
            ranklist = item['ranklist']                 # true actions
            pi_sequence = item['pi_sequence']
            phi_sequence = item['phi_sequence']
            QueryInfo = Data[queryid]
            
            total_reward = discount_reward(rewards_true, self.gamma)  # reward to-go

            ## top K
            ndoc = len(ranklist)
            ll = min(self.len_episode, ndoc)
            
            ## TODO: ndoc or length?
            for pos in range(ll):
                loss, _ = sess.run([cross_entropy, train_step], feed_dict={input_docs:\
                                   QueryInfo['feature'][ranklist], position: [0],\
                                   learning_rate: self.lr * rewards_true[pos]})
                ranklist = np.delete(ranklist, 0)
            
        del self.memory[:]

    def Eval(self, Data, typ):
        '''
        This method re-calculates the documents' scores based on updated policy
        and calculates the average ranking measures for all queries based on 
        these scores.
        '''
        thendcg = np.zeros(self.n_top)
        thedcg = np.zeros(self.n_top)
        themap = 0.0
        thealldcg = 0.0
        
        for queryid in Data:
            QueryInfo = Data[queryid]
            #score = self.model_actor(QueryInfo['feature'])[0]
            #score = score.numpy()
            score = sess.run(scores, feed_dict={input_docs: QueryInfo['feature']})[0].reshape([-1])
            # relevance labels sorted based on scores in descending order
            rates = QueryInfo['label'][np.argsort(score)[np.arange(len(score) - 1, -1, -1)]]
            themap += MAP(rates)
            thendcg += NDCG(self.n_top, rates)
            thedcg += DCG(self.n_top, rates)
            thealldcg += DCG_all(rates)

        nquery = len(Data)
        themap = themap / nquery
        thendcg = thendcg / nquery
        thedcg = thedcg /nquery
        thealldcg = thealldcg / nquery

        info = yaml.dump({"ite":self.iteration, "type": typ, "MAP": themap,\
                        "Return": thealldcg, "DCG": thedcg, "NDCG": thendcg})+'\n'
        self.result_file.write(info)

        #print(typ, ':  ', themap, thealldcg, thendcg[0], thendcg[2],\
        #      thendcg[4], thendcg[9], thedcg[0], thedcg[2], thedcg[4], thedcg[9])
        print(typ, ':  ', themap, thendcg[0], thendcg[2], thendcg[4], thendcg[9])


        