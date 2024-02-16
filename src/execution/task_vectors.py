# @title 24. EXECUTION - create task vectors and do topographical similarity analysis
from autoencoder_def import ConvAutoEncoder
from fixed_parameters import *
from changeable_parameters import *

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
import os  # for creating directories
from scipy.optimize import curve_fit
from typing import Union, Optional

import umap.umap_ as umap

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_goal_wall_indicator(label_dict, goal_reward, wall_reward):
    '''
    #For each task, create a goal/wall indicator vector as follows (dimensionality four):
    #-the first two entries are the goal state's x- and y coordinates
    #-the last two entries are the wall state's x- and y coordinates
    #-the distance between two such vectors is simply the Euclidean distance
    #-we create weightings for goal and wall by simply multiplying the first and last two components (respectively) with a number corresponding to that weighting

    #-the case of no wall state is covered by artificially placing the wall state in the top right corner outside the maze
    #(this way the distances are most meaningful, as this is symmetric w.r.t. action choice up vs. right and also a natural continuation of walls becoming less
    #important overall if they are further away from the initial location)
    '''

    # create dictionary mapping state numbers to coordinates as such:
    # here the no-wall-state situation maps the wall state to s=(4,4)
    #################
    # 3,0 3,1 3,2 3,3#
    # 2,0 2,1 2,2 2,3#
    # 1,0 1,1 1,2 1,3#
    # 0,0 0,1 0,2 0,3#
    #################
    w_goal = goal_reward
    w_wall = wall_reward
    # while the x- and y-axes start in the bottom left corner
    state_coord_dict = {i: [mt.floor(i / grid_dim), i % grid_dim] for i in range(16)}
    state_coord_dict[0] = [4, 4]  # the goal is never at zero, so no issue here
    indicatorvector_dict = {}
    indicatorvector_dictw = {}
    for task, [wall_index, init_state, goal_state, goalsplength] in label_dict.items():
        # get coordinates for wall state and goal state, append them
        wall_coord = np.array(state_coord_dict[wall_index])
        goal_coord = np.array(state_coord_dict[goal_state])
        overall_coord = np.append(goal_coord, wall_coord)
        overall_coordw = np.append(w_goal * goal_coord, w_wall * wall_coord)
        # put entry into dictionary
        indicatorvector_dict[task] = overall_coord
        indicatorvector_dictw[task] = overall_coordw

    return indicatorvector_dict, indicatorvector_dictw


def calculate_distance_between_message_pairs(q_matrix_dict, indicatorvector_dict, indicatorvector_dictw):
    '''
    #We calculate distances between all message pairs in a language and corresponding distances between a related quantity
    #So far this quantity are either the ground truth Q-matrices or the ground truth action probability matrices
    #In a next step, we then bin the meaning pairs (Q matrix or action proba matrix pairs) according to their distance in bins
    #with equal number of entries and show the corresponding distance average and standard deviation in the signal pairs (message pairs)

    #currently: 25200 pairs of signals and meanings are compared
    '''
    # flatten and transfer to numpy
    qdict = {i: torch.flatten(q).detach().numpy() for i, q in q_matrix_dict.items()}
    vdict = indicatorvector_dict
    vwdict = indicatorvector_dictw

    # pairs distances calculation for ground truth Q-matrices and ground truth action probability matrices (see previous block)
    qdist = np.array([np.linalg.norm(qdict[i] - qdict[j], norm_topo) for i in range(225) for j in range(i + 1, 225)])
    vwdist = np.array([np.linalg.norm(vwdict[i] - vwdict[j], norm_topo) for i in range(225) for j in range(i + 1, 225)])

    # sort the meaning distance pairs by distance (Q- and P-matrices) and store the sorted keys
    qdist_dict = {i: q for i, q in enumerate(qdist)}
    vwdist_dict = {i: vw for i, vw in enumerate(vwdist)}
    qkeylist = list(dict(sorted(qdist_dict.items(), key=lambda item: item[1])).keys())
    vwkeylist = list(dict(sorted(vwdist_dict.items(), key=lambda item: item[1])).keys())

    for lcode_topo in [f"nonlinear_nostudent_language{i}" for i in range(5)] + [f"nonlinear_goallocs0_zeta5_language{i}"
                                                                                for i in range(5)]:
        # PREPARATION: first create the message dictionary corresponding to a certain language
        # load stored autoencoder network parameters
        autoencoder_topo = ConvAutoEncoder(data_shape, K, nonlinear_ae_plots, nonlinear_std_plots).to(device)
        autoencoder_topo.load_state_dict(
            torch.load(file_loc + "autoencoder/autoencoder network parameters/" + f"params_autoenc{lcode_topo}.pt"))
        autoencoder_topo.eval()
        # create a message dictionary, with indices corresponding to the task indices
        message_dict2 = {}
        for task_index, q_matrix in q_matrix_dict.items():
            q_matrix = torch.unsqueeze(q_matrix,
                                       0)  # need this because the autoencoder always expects batches of inputs!
            message = autoencoder_topo.encode(q_matrix)[0]
            message_dict2[task_index] = torch.flatten(message).detach().numpy()

        # TOPOGRAHIC SIMILARITY ANALYSIS
        # pairs distances calculation for messages
        mdist = np.array(
            [np.linalg.norm(message_dict2[i] - message_dict2[j]) for i in range(225) for j in range(i + 1, 225)])

        el_per_bin = int(len(qdist) / n_bins_topo)  # divisibility required!!
        # for every language, for every bin store average meaning value and corresponding average signal value
        xqvals, yqvals, xpvals, ypvals, xvvals, yvvals, xvwvals, yvwvals = [], [], [], [], [], [], [], []
        for k in range(n_bins_topo):  # again divisibility of number of pairs by bin number is needed
            qkeylist_chunk = qkeylist[el_per_bin * k:el_per_bin * (k + 1)]
            vwkeylist_chunk = vwkeylist[el_per_bin * k:el_per_bin * (k + 1)]

            mqchunk = mdist[qkeylist_chunk]
            mvwchunk = mdist[vwkeylist_chunk]
            qchunk = qdist[qkeylist_chunk]
            vwchunk = vwdist[vwkeylist_chunk]

            xqvals += [np.average(qchunk)]
            xvwvals += [np.average(vwchunk)]
            yqvals += [np.average(mqchunk)]
            yvwvals += [np.average(mvwchunk)]

        np.savetxt(file_loc + f"topographic similarity/q vs m {n_bins_topo} bins {norm_topo}-norm {lcode_topo}",
                   np.array([xqvals, yqvals]))
        np.savetxt(file_loc + f"topographic similarity/vw vs m {n_bins_topo} bins {norm_topo}-norm {lcode_topo}",
                   np.array([xvwvals, yvwvals]))
