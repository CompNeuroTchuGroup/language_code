#@title 26. EXECUTION - Entropy analysis: binning approach

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

from gridworld import get_state_tensors
from helpers import get_message_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
Calculate entropy of message and other distributions
Always project to 2D-PCA space first (this loses some information, but makes figures comparable and also
reduces noise as we have a relatively small number of total samples)
'''

def get_2D_PCA_entropy(input_list, pca_type, hist_method, n_Hbins):
    #do PCA on all inputs, extract first two coordinates
    input_pca=pca_type.fit_transform(input_list)
    xvals, yvals=input_pca[:,0], input_pca[:,1]
    #two different methods for getting the histogram: A or B
    if hist_method=="A":
        hist, xedges, yedges=np.histogram2d(xvals, yvals, bins=n_Hbins)
    elif hist_method=="B":
        delta = max(max(xvals)-min(xvals), max(yvals)-min(yvals))/2
        xmid, ymid = (max(xvals)+min(xvals))/2, (max(yvals)+min(yvals))/2
        hist, xedges, yedges=np.histogram2d(xvals, yvals, bins=n_Hbins, range=[[xmid-delta, xmid+delta],[ymid-delta, ymid+delta]])
    #get probabilities, calculate entropy and store value
    hist=hist.flatten()/np.sum(hist)
    return sp.stats.entropy(hist)


def compute_all_entropies(label_dict, q_matrix_dict):
    #the number of bins per dimension (so have total of n_bins**2 bins)
    bll=len(H_binlist)

    #for messages we average entropy over 5 languages each, but for tasks and teacher Q-matrices we do not
    H_nofeedback, H_feedback, H_feedback_noregu=np.zeros(shape=(2,bll,5)), np.zeros(shape=(2,bll,5)), np.zeros(shape=(2,bll,5))
    H_qstd, H_pstd, H_qstd_noregu, H_pstd_noregu=np.zeros(shape=(2,bll,5)), np.zeros(shape=(2,bll,5)), np.zeros(shape=(2,bll,5)), np.zeros(shape=(2,bll,5))
    H_tasks, H_qteacher=np.zeros(shape=(2,bll)), np.zeros(shape=(2,bll))

    pca_mssg, pca_mat, pca_task=PCA(K), PCA(grid_dim**2*4), PCA(2)

    #other preparations
    cmap=plt.get_cmap("bwr")
    state_tensors=get_state_tensors(1) #for autoencoder forward function
    softy=nn.Softmax(dim=0)

    #1.Entropy of the original tasks and Q-matrices
    task_list=np.array([[i,k] for [i,j,k,l] in list(label_dict.values())])
    q_matrix_list=np.array([torch.flatten(Q).detach().numpy() for Q in q_matrix_dict.values()])

    for k,n_Hbins in enumerate(H_binlist):
        for input_list, pca_type, H_arr in zip([q_matrix_list, task_list], [pca_mat, pca_task], [H_qteacher, H_tasks]):
            H_arr[0][k]=get_2D_PCA_entropy(input_list, pca_type, "A", n_Hbins)
            H_arr[1][k]=get_2D_PCA_entropy(input_list, pca_type, "B", n_Hbins)


    #2.Entropy of the messages (nofeedback, feedback, feedback without regularization) and student matrices
    nofeedback_codes=[f"nonlinear_nostudent_language{i}" for i in range(5)]
    feedback_codes=[f"nonlinear_goallocs0_zeta5_language{i}" for i in range(5)]
    feedback_noregu_codes=[f"nonlinear_noregularization_language{i}" for i in range(5)]
    H_language_codes=[nofeedback_codes, feedback_codes, feedback_noregu_codes]
    Hm_arrs=[H_nofeedback, H_feedback, H_feedback_noregu]
    Hq_arrs=[None, H_qstd, H_qstd_noregu]
    Hp_arrs=[None, H_pstd, H_pstd_noregu]

    for Hi_language_codes, Hm_arr, Hq_arr, Hp_arr in zip(H_language_codes, Hm_arrs, Hq_arrs, Hp_arrs):
        for l,lcode in enumerate(Hi_language_codes):
            message_list, autoencoder_H=get_message_list(lcode, q_matrix_dict)
            if Hq_arr is not None:
                q_matrixstd_list=torch.stack([torch.flatten(autoencoder_H.student(torch.tensor([m]),state_tensors)[0])for m in message_list]).detach().numpy()
                p_matrixstd_list=torch.stack([torch.flatten(softy(autoencoder_H.student(torch.tensor([m]),state_tensors)[0])) for m in message_list], dim=0).detach().numpy()
            for k,n_Hbins in enumerate(H_binlist):
                Hm_arr[0][k][l]=get_2D_PCA_entropy(message_list, pca_mssg, "A", n_Hbins)
                Hm_arr[1][k][l]=get_2D_PCA_entropy(message_list, pca_mssg, "B", n_Hbins)
                if Hq_arr is not None:
                    for input_list, H_arr in zip([q_matrixstd_list, p_matrixstd_list],[Hq_arr, Hp_arr]):
                        H_arr[0][k][l]=get_2D_PCA_entropy(input_list, pca_mat, "A", n_Hbins)
                        H_arr[1][k][l]=get_2D_PCA_entropy(input_list, pca_mat, "B", n_Hbins)

    return H_language_codes, Hm_arrs, Hq_arrs, Hp_arrs
