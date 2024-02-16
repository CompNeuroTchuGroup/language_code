from autoencoder_def import ConvAutoEncoder, MatrixDataset
from autoencoder_training import train_autoencoder, task_indices_sorter
from fixed_parameters import *
from changeable_parameters import *
from qmatrix_gridworlds import *
from helpers import *


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


#@title 18. EXECUTION - Language training (with or without student feedback)

#autoencoder: specific properties/parameters
train_order=0 #0: no parameters frozen / 1:student parameters frozen / 2:autoencoder parameters frozen
batch_size=10 #batch the data (mini-batch gradient descent)
train_worlds=range(16) #these are the worlds we are training on (from the wall_state_dict)
#different indices describe different situations where we train on all mazes, but only a subset of the goals
train_goals_dict={0:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],1:[1,4,3,6,9,12,11,14],2:[2,5,8,7,10,13,15],3:[1,4,5,8,9,12,13],4:[2,3,6,7,10,11,14,15],5:[1,2,3,4,5,6,7],6:[8,9,10,11,12,13,14,15]}
train_goals_index=0 #which goal locations are used for training the language?
nonlinear_ae_train, nonlinear_std_train=True, True #do autoencoder/student networks have nonlinear activations in training?

language_number_autoenc=1 #how many languages with the same parameters and different seeds are trained?
language_save_code=f"test_language" #this list and the two following need to have the same length (will be zipped together)


def generate_language(label_dict, q_matrix_dict): #todo: check which other parameters are required here
    train_goals = train_goals_dict[train_goals_index]  # pick the goals we train on
    for language_index in range(language_number_autoenc):  # create several languages to reduce variance

        language_save_codex = language_save_code
        if language_number_autoenc > 0:
            language_save_codex = language_save_code + f"_language{language_index}"

        goal_world_inds, _, _, _ = task_indices_sorter(label_dict, train_worlds, train_goals)
        # create the dataset for the network to train on from the training indices
        q_matrix_dict_train, label_dict_train = {}, {}
        # filter out only the relevant tasks (maze in train_worlds, goal in train_goals)
        for task_counter, ind in enumerate(goal_world_inds):
            q_matrix_dict_train[task_counter] = q_matrix_dict[ind]
            label_dict_train[task_counter] = label_dict[ind]

        # create dataset
        matrix_dataset = MatrixDataset(q_matrix_dict_train, label_dict_train)

        # initialize network and optimizer
        autoencoder = ConvAutoEncoder(data_shape, K, nonlinear_ae_train, nonlinear_std_train).to(device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate_autoenc)

        '''
        #training without student feedback
        losses1, rec_losses1, spar_losses1 = train_autoencoder(autoencoder, optimizer, matrix_dataset, wall_state_dict,
                                                                gamma_sparse,zeta_std, kappa, training_epochs, batch_size, False, train_order)

        #save autoencoder parameters and losses
        torch.save(autoencoder.state_dict(), file_loc+"autoencoder/autoencoder network parameters/"+f"params_autoenc{language_save_codex}.pt")
        torch.save(optimizer.state_dict(), file_loc+"autoencoder/optimizer parameters/"+f"optimizer{language_save_codex}.pt")
        np.savetxt(file_loc+f"autoencoder/losses/losses1_{language_save_codex}.txt", losses1)
        np.savetxt(file_loc+f"autoencoder/losses/rec_losses1_{language_save_codex}.txt",rec_losses1)
        np.savetxt(file_loc+f"autoencoder/losses/spar_losses1_{language_save_codex}.txt",spar_losses1)

        #re-initialize for new training with student feedback
        autoencoder = ConvAutoEncoder(data_shape, K, nonlinear_ae_train, nonlinear_std_train).to(device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate_autoenc)
        '''

        # training with student feedback
        losses2, rec_losses2, spar_losses2 = train_autoencoder(autoencoder, optimizer, matrix_dataset,
                                                               wall_state_dict,
                                                               gamma_sparse, zeta_std, kappa, training_epochs,
                                                               batch_size, True, train_order)

        goal_losses2 = (losses2 - (1 - gamma_sparse) * rec_losses2 - gamma_sparse * spar_losses2) / zeta_std

        # save autoencoder parameters and losses
        torch.save(autoencoder.state_dict(),
                   file_loc + "autoencoder/autoencoder network parameters/" + f"params_autoenc{language_save_codex}.pt")
        torch.save(optimizer.state_dict(),
                   file_loc + "autoencoder/optimizer parameters/" + f"optimizer{language_save_codex}.pt")
        np.savetxt(file_loc + f"autoencoder/losses/losses2_{language_save_codex}.txt", losses2)
        np.savetxt(file_loc + f"autoencoder/losses/rec_losses2_{language_save_codex}.txt", rec_losses2)
        np.savetxt(file_loc + f"autoencoder/losses/spar_losses2_{language_save_codex}.txt", spar_losses2)
        np.savetxt(file_loc + f"autoencoder/losses/goal_losses2_{language_save_codex}.txt", goal_losses2)
    return autoencoder


def train_language(q_matrix_dict, label_dict, language_gen):
    if language_gen:
        #train autoencoder from scratch
        autoencoder = generate_language(label_dict, q_matrix_dict)

    if not language_gen:
        # load stored autoencoder network parameters
        autoencoder = ConvAutoEncoder(data_shape, K, nonlinear_ae_plots, nonlinear_std_plots).to(device) #todo:put this parameters in the function
        autoencoder.load_state_dict(
            torch.load(file_loc + "autoencoder/autoencoder network parameters/" + f"params_autoenc{language_code}.pt"))
        autoencoder.eval()

        '''
        #only load optimizer if necessary, e.g. if training was interrupted - then parameters can be specified
        optimizer=torch.optim.Adam(autoencoder.parameters(), lr=learning_rate_autoenc)
        optimizer.load_state_dict(torch.load(file_loc+"autoencoder/optimizer parameters/"+f"optimizer{language_code}.pt", map_location=torch.device(device)))
        '''
    # create a message dictionary, with indices corresponding to the task indices
    message_dict = {}
    for task_index, q_matrix in q_matrix_dict.items():
        q_matrix = torch.unsqueeze(q_matrix, 0)  # need this because the autoencoder always expects batches of inputs!
        message = autoencoder.encode(q_matrix)[0]
        message_dict[task_index] = message

    return message_dict







