#@title 28. EXECUTION - Hyperparameter study (message length, student loss, number of epochs): extract rates
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from autoencoder_def import ConvAutoEncoder, MatrixDataset
from autoencoder_training import train_autoencoder
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

from helpers import read_dict_from_pkl
'''
Hyperparameter analysis - we kept everything constant and varied only one of the three parameters mentioned in the title
'''

#label dictionaries for all trained and unknown mazes respectively


label_dict_train=read_dict_from_pkl(file_loc+"teacher/label dictionaries/"+f"q_matrices_labelstraining_4x4.pkl")
label_dict_test=read_dict_from_pkl(file_loc+"teacher/label dictionaries/"+f"q_matrices_labelstest_4x4.pkl")

for folder_tag, xaxis in zip(["epochs", "zeta", "mlength_K"], [[100,200,400,600,800,1000], [1,2,5,10,20], [1,2,3,4,5,6,7,8]]):

    folders_known=[f"nonlinear_{folder_tag}{i}_known" for i in xaxis]
    folders_unknown=[f"nonlinear_{folder_tag}{i}_unknown" for i in xaxis]

    language_nr_hyp=5
    param=2 #the stepfactor
    #potentially chuck out languages if the performances are bad
    for plot, ldict, known_addon, folders in zip([0,1], [label_dict_train, label_dict_test], ["known","unknown"], [folders_known, folders_unknown]): #iterate over performance on known and unknown tasks

        for j,folder in enumerate(folders):
            rates=[[],[],[],[]]

            for language in range(language_nr_hyp):
                single_rates=[[],[],[],[]]
                #load the solving rates
                for k,student in enumerate(["info","misinfo"]):
                    single_rates[k]=np.loadtxt(file_loc+"student/"+f"{folder}/{folder}_language{language}/"+f"solving_rates_{student}_no_learning_stepfactor2.txt")
                single_rates[2]=np.loadtxt(file_loc+"student/"+f"{folder}/{folder}_language{language}/"+f"solving_rates_rdwalkersmart_stepfactor2.txt")
                single_rates[3]=np.loadtxt(file_loc+"student/"+f"{folder}/{folder}_language{language}/"+f"solving_rates_rdwalker_stepfactor2.txt")

                for index in range(len(single_rates)):
                    rates[index]+=[np.mean(single_rates[index])]

            np.savetxt(file_loc+"hyperparameters/"+f"rates {folder_tag} {j} {known_addon}", np.array(rates))

# #@title 29. PLOTS - Hyperparameter study (Fig. S8)
# '''
# Hyperparameter analysis - we kept everything constant and varied only one of the three parameters mentioned in the title
# '''
from matplotlib.patches import Rectangle

#plot settings
plt.rc('axes', labelsize=28)
plt.rc('xtick', labelsize=28)
plt.rc('ytick', labelsize=28)
plt.rc('axes', titlesize=28)
cmap=plt.get_cmap("bwr")

folder_tags=["epochs", "zeta", "mlength_K"]
rect_xs=[950,4,4.6]
rect_widths=[100,2,0.8]
bboxes=[[0.6,0.84], [1, 0.725], [0.44, 0.725]]
xlabels=["training epochs "+r"$N_{\mathrm{epochs}}$", "loss weighting "+r"$\zeta$", "message length "+r"$K$"]


x_axes=[[100,200,400,600,800,1000], [1,2,5,10,20], [1,2,3,4,5,6,7,8]]
for index, (folder_tag, xaxis, rect_x, rect_width, bbox, xlabel) in enumerate(zip(folder_tags, x_axes, rect_xs, rect_widths, bboxes, xlabels)):
    #plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(17,5))
    ax[0].grid(visible=True, axis="y")
    ax[1].grid(visible=True, axis="y")
    plt.rcParams['svg.fonttype']='none' #"to make later editing of figures easier" (Carlos)
    labels=["informed student","misinformed student","smart random walker","random walker"]
    colors=[cmap(0.15),cmap(0.25),cmap(0.99),cmap(0.75)]

    language_nr_hyp=5
    param=2 #the stepfactor
    #potentially chuck out languages if the performances are bad
    for plot, known_addon in zip([0,1], ["known","unknown"]): #iterate over performance on known and unknown tasks
        avg_rates=[[],[],[],[]]
        std_rates=[[],[],None, None]

        for j,_ in enumerate(xaxis):

            rates=np.loadtxt(file_loc+"hyperparameters/"+f"rates {folder_tag} {j} {known_addon}")
            #print(f"info:            {np.round(100*np.array(rates[0]),1)} - average {round(100*np.mean(rates[0]),1)}")
            #print(f"misinfo:         {np.round(100*np.array(rates[1]),1)} - average {round(100*np.mean(rates[1]),1)}")
            #print(f"smart rd walker: {round(100*rates[2][0],1)}")
            #print(f"rd walker:       {round(100*rates[3][0],1)}")

            #add average solving rate and NOW STANDARD ERROR OF THE MEAN
            for j,rates_student in enumerate(rates):
                avg_rates[j]+=[100*np.mean(rates_student)]
                if j<2:
                    std_rates[j]+=[100*np.std(rates_student, ddof=1)]
                    #sem_rates[j]+=[100*np.std(rates_student, ddof=1)/mt.sqrt(language_nr_hyp)]

        for yrates,std,label,color in zip(avg_rates, std_rates, labels, colors):
            ax[plot].plot(xaxis, yrates, color=color, lw=3)
            ax[plot].scatter(xaxis, yrates, color=color, label=label, marker="X", s=150)
            #ax[plot].errorbar(xaxis, yrates, yerr=std, color=color, label=label, fmt="X", ms=15, lw=3)

        ax[plot].set_ylim(0,100)
        ax[plot].set_ylabel("tasks solved (%)")
        ax[plot].set_xlim(0,1.1*xaxis[-1])

        #Define the coordinates and dimensions of the rectangle
        rect_height = 98  # height of the rectangle (can be adjusted as needed)
        # Create a Rectangle patch
        rect1 = Rectangle((rect_x, 1), rect_width, rect_height, fill=False, edgecolor='black', label="value in main results", lw=3)
        rect2 = Rectangle((rect_x, 1), rect_width, rect_height, fill=False, edgecolor='black', label="value in main results", lw=3)
        # Add the Rectangle patch to the plot
        if plot==0:
            ax[plot].add_patch(rect1)
        elif plot==1:
            ax[plot].add_patch(rect2)

        ax[plot].set_yticks([0,20,40,60,80,100])
        ax[plot].set_xlabel(xlabel)
        if index==0 and plot==0:
            ax[plot].legend(loc='upper right', bbox_to_anchor=bbox, fontsize=15)

        ax[plot].tick_params(width=5, length=10)

    ax[0].set_title(f"trained tasks", y=1.05)
    ax[1].set_title(f"test tasks", y=1.05)

    plt.subplots_adjust(wspace=0.5, hspace=0.6)

    plt.show()
