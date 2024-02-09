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
import os #for creating directories
from scipy.optimize import curve_fit
from typing import Union, Optional
import umap.umap_ as umap

#for jupyter widget progress bar
# from google.colab import output
# output.enable_custom_widget_manager()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
#this is set for the printing of Q-matrices via console
torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)

from tqdm.auto import tqdm, trange


#Uncomment if GPU is to be used - right now use CPU, as we have very small networks and for them, CPU is actually faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#suppress scientific notation in printouts
np.set_printoptions(suppress=True)
# This is a sample Python script.

#@title 6. FUNCTIONS - Transition Memory class



Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    '''
    a class to store and replay previous transitions of states and actions done by the network
    like a memory
    '''

    def __init__(self, capacity: int):
        #Have short- and long-term memory to optimize exploration results
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int)->'list[Transition]':
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)