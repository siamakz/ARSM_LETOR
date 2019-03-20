
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

# Auxiliary functions

def pseudo_trajectory(pseudo_action, gamma, pseudo_step, score, actions, rates, rand_seed):
    # record the rewards till the pseudo step
    np.random.seed(rand_seed)         
    ndoc = len(score)
    phi = score
    positions = list(range(ndoc))
    pseudo_ranklist = np.zeros(ndoc, dtype=np.int32)
    pseudo_ranklist[:pseudo_step] = actions[:pseudo_step]
    # first remove the assigned docs from scores and positions
    for choice in sorted(actions[:pseudo_step], reverse=True):
        phi = np.delete(phi, choice)
        del positions[choice]
    for position in range(pseudo_step, ndoc):
        pi = np.random.exponential(np.ones(len(phi)))
        pi = pi/np.sum(pi)
        choice = np.argmin(np.log(pi) - phi)
        pseudo_ranklist[position] = positions[choice]
        phi = np.delete(phi, choice)
        del positions[choice]
        
    pseudo_reward = GetReward_DCG(rates[pseudo_ranklist])      
    dr = discount_reward(pseudo_reward, gamma)
    return dr, np.sum(pseudo_reward)


# Basic object
class ARSM_Letor(object):
    
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
        
        self.MaxPseudoActionSequences = 100
        
        global scores, input_docs, position, learning_rate, sess, train_step,\
        loss_fn, grads_vars, prob, f_delta
        
        input_docs = tf.placeholder(tf.float32, [None, self.n_feature])
        position = tf.placeholder(tf.int64)
        learning_rate = tf.placeholder(tf.float32, shape=[])
        f_delta = tf.placeholder(tf.float32, shape=[None, None])
        # actor model T_{\theta}(s_t)
        ## TODO: MDPRank doesn't have bias term
        '''self.model_actor = tf.keras.Sequential([tf.keras.layers.Dense(self.n_hidden_unit,\
                    activation = "tanh", input_shape = (self.n_feature,),\
                    kernel_initializer = tf.truncated_normal_initializer(mean=0,\
                    stddev = 0.1/np.sqrt(float(n_feature))),\
                    bias_initializer = tf.constant_initializer(0.0)),\
                    tf.keras.layers.Dense(1,\
                    kernel_initializer = tf.truncated_normal_initializer(mean=0,\
                    stddev = 0.1/np.sqrt(float(self.n_hidden_unit))),\
                    bias_initializer=tf.constant_initializer(0.0))])'''
        
        #self.optimizer_actor = tf.train.AdamOptimizer(learning_rate = self.lr)
        
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

        ## TODO: logits=prob or scores?? 
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=position)  # logits!! 
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        
        loss_fn = tf.reduce_sum(tf.multiply(scores, f_delta))
        #opt = tf.train.GradientDescentOptimizer(learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate = self.lr)
        grads_vars = opt.compute_gradients(loss_fn)
        train_step = opt.apply_gradients(grads_vars)
        init = tf.global_variables_initializer()
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
            phi = score#.numpy().transpose()[0]
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
            rewards_true = GetReward_DCG(rates)
            
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
            lenghth = min(self.len_episode, ndoc)
            NumPseudoActionSequences = 0
            unique_pseudo_actions = []         # list of vectors
            pseudo_action_sequences = []       # list of matrices
            doc_list = []
            ## TODO: ndoc or length?
            time_permute = np.random.permutation(ndoc)
            time_permute_used = []
            for t in time_permute:
                action_true = ranklist[t]
                pi = pi_sequence[t]
                phi = phi_sequence[t]
                pseudo_actions = pseudo_action_swap_matrix(pi, phi)
                temp = np.unique(pseudo_actions[pseudo_actions != action_true])
                if NumPseudoActionSequences + temp.size > self.MaxPseudoActionSequences:
                    break
                else:
                    NumPseudoActionSequences = NumPseudoActionSequences + temp.size
                if temp.size > 0:
                    pseudo_action_sequences.append(pseudo_actions)
                    temp = np.insert(temp[:,np.newaxis], 0, values=t, axis=1)
                    unique_pseudo_actions.append(temp)
                    time_permute_used.append(t)
                    positions = list(range(ndoc))
                    for choice in sorted(ranklist[:t], reverse=True):
                        del positions[choice]
                    doc_list.append(positions)
            
            if len(unique_pseudo_actions) > 0:
                unique_pseudo_actions = np.vstack(unique_pseudo_actions)
                RandSeed = np.random.randint(0,1e9, NumPseudoActionSequences)
                '''
                ncpu = multiprocessing.cpu_count()
                pool = ProcessPoolExecutor(ncpu)
                '''
                # run pseudo-trajectory for each pseudo-action in parallel
                score = sess.run([scores], feed_dict={input_docs: \
                                 QueryInfo['feature']})[0].reshape([-1])
                #score = self.model_actor(QueryInfo['feature'])#[0] 
                #score = score.numpy().transpose()[0]
                rates = QueryInfo['label']
                '''
                futures = [pool.submit(pseudo_trajectory, pseudo_action, seed,\
                        pseudo_step, score, ranklist, rand_seed)\
                        for pseudo_action, pseudo_step, rand_seed in \
                        zip(unique_pseudo_actions[:NumPseudoActionSequences, 1],\
                        unique_pseudo_actions[:NumPseudoActionSequences, 0], RandSeed)]
                # extract the pseudo-rewards for each pseudo-trajectory        
                pseudo_sequences = [futures[i].result() for i in range(len(futures))]
                '''
                pseudo_sequences = []
                for pseudo_action, pseudo_step, rand_seed in\
                    zip(unique_pseudo_actions[:NumPseudoActionSequences, 1],\
                        unique_pseudo_actions[:NumPseudoActionSequences, 0], RandSeed):
                        pseudo_sequences.append(pseudo_trajectory(pseudo_action, self.gamma,\
                                      pseudo_step, score, ranklist, rates, rand_seed))
                pseudo_sequences = np.vstack(pseudo_sequences)
                pseudo_reward_total_no_discount = pseudo_sequences[:, 1]
                pseudo_reward_total = pseudo_sequences[:, 0]
                '''
                if IsPlot:
                    plt.subplot(321)
                    plt.plot(pseudo_reward_total-total_reward)
                '''
                f = np.zeros((ndoc, ndoc))
                t_cnt = 0
                for t in time_permute_used:
                    nA = len(phi_sequence[t])
                    ft = np.full((nA, nA), total_reward)   # R_{tmj}
                    idxt = np.where(unique_pseudo_actions[:, 0] == t)[0] # index of pseudo-actions happened at time t
                    
                    for idx in idxt:
                        aa = unique_pseudo_actions[idx, 1]   # a pseudo-action that happened at time t
                        # pseudo_reward_total[idx]: reward for pseudo-action aa at time t
                        ft[pseudo_action_sequences[t_cnt] == aa] = pseudo_reward_total[idx]   # line 80, algorithm 3     
                    
                    meanft = np.mean(ft, axis=0)
                    tes = np.matmul(ft - meanft, 1.0/nA - pi_sequence[t])
                        
                    f[t, doc_list[t_cnt]] = tes/np.power(self.gamma, t)  #reward discount starting at time t+1
                    t_cnt += 1
                '''
                if IsPlot:
                    plt.subplot(323)
                    plt.plot(f,'.')
                    
                if e<25 and np.mean(np.abs(f[:n_true_,:])) < 1e-14 :
                    # With a bad initilization, ARSM generates no psudo actions 
                    # and hence has zero gradient; so reinitilize the model
                    self.model_actor = tf.keras.Sequential([tf.keras.layers.Dense(self.n_hidden_unit,\
                                  activation = "tanh", input_shape = (self.n_feature,),\
                                  kernel_initializer = tf.truncated_normal_initializer(mean=0,\
                                  stddev = 0.1/np.sqrt(float(n_feature))),\
                                  bias_initializer = tf.constant_initializer(0.0)),\
                                  tf.keras.layers.Dense(1,\
                                  kernel_initializer = tf.truncated_normal_initializer(mean=0,\
                                  stddev = 0.1/np.sqrt(float(self.n_hidden_unit))),\
                                  bias_initializer=tf.constant_initializer(0.0))])
                
                f_delta = tf.convert_to_tensor(-f, dtype=tf.float32)
                
                entropy_par = 0.0
                grad_actor = gradient_arm(self.model_actor, states, f_delta, entropy_par)
                
                with tf.GradientTape() as tape:
                    #tape.watch(self.model_actor.variables)
                    logit = tf.transpose(self.model_actor(QueryInfo['feature']))
                    loss_fn = tf.reduce_sum(tf.multiply(logit, f_delta))
                grad_actor = tape.gradient(loss_fn, self.model_actor.variables)
                '''
                score, loss, _ = sess.run([scores, loss_fn, train_step], feed_dict={input_docs:\
                               QueryInfo['feature'], f_delta:-f})
                #self.optimizer_actor.apply_gradients(zip(grad_actor, self.model_actor.variables))
            
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

        info=yaml.dump({"ite":self.iteration, "type": typ, "MAP": themap,\
                        "Return": thealldcg, "DCG": thedcg, "NDCG": thendcg})+'\n'
        self.result_file.write(info)

        #print(typ, ':  ', themap, thealldcg, thendcg[0], thendcg[2],\
        #      thendcg[4], thendcg[9], thedcg[0], thedcg[2], thedcg[4], thedcg[9])
        print(typ, ':  ', themap, thendcg[0], thendcg[2], thendcg[4], thendcg[9])


        