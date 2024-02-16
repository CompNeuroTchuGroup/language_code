#@title 5. FUNCTIONS - Gridworld class and state representation transformations

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

#Uncomment if GPU is to be used - right now use CPU, as we have very small networks and for them, CPU is actually faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tqdm.auto import tqdm, trange


class SquareGridworld():
    """
    the class for the gridworlds

    n**2 states (n-by-n gridworld) -> n is the grid dimension

    The mapping from state to the grid is as follows:
    n(n-1)  n(n-1)+1  ...  n^2-1
    ...     ...       ...  ...
    n       n+1       ...  2n-1
    0       1         ...  n-1

    Actions 0, 1, 2, 3 correspond to right, up, left, down (always exactly one step)

    -Agent starts at the init_state
    -Landing in the goal_state gets a reward of goal_reward and ends the episode
    -Bumping into wall states or the map border incurs a reward of wall_reward
    -In case of lava=False, the agent bounces back from walls, while in case of lava=True wall states are accessible (the outside boundary can never be crossed)
    -Each step additionally incurs a reward of step_reward
    """
    def __init__(self,init_state: int,goal_state: int,wall_states: 'list[int]', lava: bool):

        self.init_state: int=init_state
        self.goal_state: int=goal_state
        self.wall_states: 'list[int]'=wall_states
        self.n_states: int = grid_dim**2
        self.lava: bool = lava


    def get_outcome(self, state: int, action: int)-> 'tuple[Optional[int], float]':
        '''
        given a state and an action, this returns the next state and the immediate reward for the action
        ---
        INPUT
        state: the state the agent is in
        action: the action taken
        ---
        OUTPUT
        next_state - the next state
        reward - the reward for the action taken
        '''

        #if the goal is reached, we get the goal_reward and the episode ends
        if state == self.goal_state:
                reward: float = goal_reward
                next_state = None #terminates the episode
                return next_state, reward

        #get the next state before taking into account walls or outside boundary
        next_state_dict={0:state+1, 1:state+grid_dim, 2:state-1, 3:state-grid_dim}
        #for all actions, this dictionary stores a boolean indicating whether we cross the map border executing this action in our current state
        cross_boundary_dict={0:state % grid_dim == grid_dim-1, 1:state >= grid_dim*(grid_dim-1), 2:state % grid_dim ==0, 3:state<grid_dim}

        #case that we have lava states that the agent can walk through with large negative reward
        if self.lava:
            reward=step_reward
            next_state=next_state_dict[action]
            #bounce back from map border
            if cross_boundary_dict[action]:
                reward+=wall_reward
                next_state=state
            #entering or exiting a lava state gives negative reward, but we do not bounce back
            elif next_state in self.wall_states or state in self.wall_states:
                reward+=wall_reward

        #case that we have wall states that the agent bounces back from
        else:
            reward=step_reward
            next_state=next_state_dict[action]
            #bounce back from a wall or the map border?
            if next_state in self.wall_states or cross_boundary_dict[action]:
                next_state=state #bounce back
                reward+=wall_reward


        return int(next_state) if next_state is not None else None, reward


    def get_outcomes(self) -> 'dict[ tuple[int,float] , tuple[Optional[int], float] ]':
        '''
        returns a dictionary where for every possible combination of state and action we get the next state and
        corresponding immediate reward
        ---
        OUTPUT
        outcomes - the dictionary keyed by state-action combo, whose values are the next states and rewards
        '''
        outcomes = {(s, a): self.get_outcome(s,a) for s in range(self.n_states) for a in range(n_actions)}
        return outcomes



def state_int_to_tuple(state_int: Optional[int]) -> Optional[torch.tensor]:
    '''
    Gets state tuple representation (coordinates) from integer representation
    ---
    INPUT
    state_int - the state integer representation
    ---
    OUTPUT
    state - the state tuple representation
    '''
    if state_int==None:
        return None
    else:
        cval: float=(grid_dim-1)/2 #center (0,0) is in the middle of the grid
        sx,sy=state_int%grid_dim-cval, mt.floor(state_int/grid_dim)-cval
        state = torch.tensor([[sx,sy]],device=device)
        return state

def state_tuple_to_int(state: Optional[torch.tensor]) -> Optional[int]:
    '''
    Gets state integer representation from tuple representation (coordinates)
    ---
    INPUT
    state - the state tuple representation
    ---
    OUTPUT
    state_int - the state integer representation
    '''
    if state==None:
        return None
    else:
        sx,sy=state
        cval: float=(grid_dim-1)/2
        sx,sy=sx+cval,sy+cval
        state_int=int(sx.item()+grid_dim*sy.item())
        return state_int

def get_state_tensors(m_len: int) -> torch.tensor:
    '''
    Concatenate tensors representing particular states in the gridworld so that fewer network forwards are needed
    ---
    INPUT
    m_len - length of the message batch
    ---
    OUTPUT
    state_tensors - the concatenated state tensors
    '''
    state_tensors=torch.zeros(size=(grid_dim,grid_dim*m_len,2))
    for j in range(grid_dim):
        for i in range(grid_dim):
            s=i*grid_dim+j
            s_tup=state_int_to_tuple(s)[0]
            for b in range(m_len):
                state_tensors[j,i*m_len+b]=s_tup

    return state_tensors
