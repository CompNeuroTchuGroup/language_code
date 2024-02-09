#@title 9. FUNCTIONS - Optimizer for DQN
'''
Optimization function for the DQN
'''


from fixed_parameters import *
from changeable_parameters import *
from dqn import DQN
from transition_memory import *
from gridworld import *

import gc
import gym
import math as mt
import scipy as sp
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
from matplotlib.font_manager import FontProperties
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import networkx as nx
import pickle as pkl
import itertools
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pycolormap_2d import ColorMap2DBremm, ColorMap2DZiegler
import time
import os #for creating directories
from scipy.optimize import curve_fit
from typing import Union, Optional
import umap.umap_ as umap


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset



def optimize_dqn(network: DQN,optimizer: torch.optim, memory: torch.tensor, batches: 'tuple[torch.tensor, torch.tensor]', goal_found: bool, loss_norm) -> float:
    '''
    Do one optimizer step of the (teacher) DQN
    ---
    INPUT
    network: the DQN we are optimizing
    optimizer: the opimizer for the DQN
    memory: tensor containing the indices of all transitions that are used for this training step (there can be multiple occurences of single indices)
            -> those indices comprise the short- and long-term memory
    batches: contains a concatenation of a) the batch of all states and b) the batch of nextstates (for all possible transitions)
             further it contains the batch of rewards (for all possible transitions)
    goal_found: Indicates if the goal has been found yet
    loss_norm: norm for the loss function (i.e. nn.MSELoss)
    ---
    OUTPUT:
    loss: loss in this optimizer step
    '''

    states_nextstates, rewards = batches

    #Compute all the Q-values for all states, all nextstates and all 4 actions each
    qvalues=network(states_nextstates)
    state_action_values = torch.flatten(qvalues[:int(len(rewards)/4)])
    # Compute max_a' Q(s_{t+1},a') for all next states
    next_state_values = qvalues[int(len(rewards)/4):].max(dim=1,keepdim=True).values

    # Compute the expected Q values
    expected_state_action_values = torch.flatten(gamma_bellman*next_state_values + rewards) #Bellman


    #get the occurences of each transition, square root is necessary for MSELoss
    trans_factors=torch.sqrt(torch.bincount(memory, minlength=len(next_state_values)))
    #train all 4 goal state values (for the 4 actions) only if the goal has been found
    if goal_found:
        expected_state_action_values[-4:]=2.
        trans_factors[-4:]=mt.sqrt(2) #put every goal action twice (long- and short-term memory)
        # Compute loss using the Bellman equation and MSE norm (if this is to change, have to adjust the square root above)
        loss = 1/mt.sqrt(len(memory)+8)*loss_norm(state_action_values*trans_factors, expected_state_action_values*trans_factors) #+8 because of 4 goal actions two times
    if not goal_found:
        expected_state_action_values[-4:]=0.
        state_action_values[-4:]=0.
        # Compute loss using the Bellman equation and MSE norm (if this is to change, have to adjust the square root above)
        loss = 1/mt.sqrt(len(memory))*loss_norm(state_action_values*trans_factors, expected_state_action_values*trans_factors)


    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    '''
    #restricting parameter gradients to lie in the interval [-1,1]
    for param in network.parameters():
        param.grad.data.clamp_(-1, 1)
    '''

    optimizer.step()
    return loss.item()




def transition_memories(init_state:int, goal_state:int, wall_states:'list[int]')->'tuple[ dict[tuple[int,int],int], tuple[torch.tensor, torch.tensor] ]':
    '''
    create practical memory for transitions that allows for savings in Q-network training
    when learning the task
    ---
    INPUT
    init_state, goal_state, wall_states: specifications of the task
    ---
    OUTPUT
    transition_index_dict: dictionary taking as input a state-action combination and giving out the index belonging to this particular transition
    batches_list: list containing two tensor, firstly the concatenated states and next_states and secondly the concatenated rewards.
                  this complicated format makes computations faster.
    '''
    #initialize environment
    env = SquareGridworld(init_state,goal_state,wall_states, lava)
    outcomes=env.get_outcomes()

    transition_memory = ReplayMemory(capacity=n_actions*grid_dim**2)
    transition_index_dict={} #each transition gets an index
    #create the transition batches according to the world setup
    i=0
    for s_int in range(grid_dim**2):
        if s_int in wall_states or s_int==goal_state:
            continue
        s=state_int_to_tuple(s_int)
        for a in range(n_actions):
            transition_index_dict[(s_int,a)]=i
            i+=1
            ns_int, r = outcomes[s_int,a]
            ns=state_int_to_tuple(ns_int)
            transition_memory.push(s, torch.tensor([[a]], device=device), ns, torch.tensor([[r]], device=device))
    #add the goal state transitions at the end
    gs=state_int_to_tuple(goal_state)
    for a in range(n_actions):
        transition_memory.push(gs, torch.tensor([[a]], device=device), gs, torch.tensor([[2.]], device=device))
        transition_index_dict[(goal_state,a)]=i
        i+=1

    transitions = transition_memory.memory
    #create a big transition
    batch = Transition(*zip(*transitions))
    #and now the batches to train on for the network
    nextstates=torch.cat(batch.next_state)
    rewards=torch.cat(batch.reward)


    #second memory for only the states (don't need to run all of them through the network four times, but only one time)
    transition_memory = ReplayMemory(capacity=n_actions*grid_dim**2)
    for s_int in range(grid_dim**2):
        if s_int in wall_states or s_int==goal_state:
            continue
        s=state_int_to_tuple(s_int)
        transition_memory.push(s, torch.tensor([[0]], device=device), s, torch.tensor([[0.]], device=device))
    #add the goal state transitions at the end
    gs=state_int_to_tuple(goal_state)
    transition_memory.push(gs, torch.tensor([[0]], device=device), gs, torch.tensor([[0.]], device=device))

    transitions = transition_memory.memory
    #create a big transition
    batch = Transition(*zip(*transitions))
    states=torch.cat(batch.state)
    batches_list=[torch.cat([states, nextstates]), rewards]

    return transition_index_dict, batches_list