#@title 19. EXECUTION - Calculate student performance (solving rates)

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

from solving_rates import random_walker_rates, student_performance_evaluator

'''
test the students on how they process new information given a certain message

methods: softmax with no learning and "simple learning", i.e. avoiding actions done before in this state
'''

#Parameters for the evaluations
random_repeat=5 #randomness repetition for misinformed students
rdwalker_base_rates=[] #baseline random walker rates we set for each task (values like 0.1 for 10% and 0.2 for 20%)
stepfactors=[2] #number of steps we allow for the student in terms of shortest path length for each task
methods=["no_learning"] #could choose "no_learning" (action choice by percentages) and/or "simple_learning" (greedy action choices, but if you come back to a state you take the next best action)

#"known" for trained worlds or "unknown" for untrained worlds
if qmat_read_code=="training4x4":
    known_addon="known"
elif qmat_read_code=="test4x4":
    known_addon="unknown"
else:
    known_addon=None

#Parameters specified by the languages we want to evaluate
goal_locs=[0] #list of trained goal locations

language_nr_evaluation=1 #number of languages to evaluate (assume it's the same for all different goal locations)
language_codes=[f"test_language"]
saving_folders=[f"test_language_{known_addon}"]
evalute_nonlinear_ae, evaluate_nonlinear_std = True, True #are the activations in the autoencoder/student nonlinear?

#first choice if we are doing 'closing the loop', i.e. using the student Q-matrices for creating messages (which are different for every language)
#and second choice if we take regular teacher Q-matrices for creating messages
#q_matrix_dict_list=[read_dict_from_pkl(file_loc+f"closing the loop/test_language_language0/studentQmatrices.pkl")]





gc.enable() #garbage collector enabled to free RAM


def evaluate_students(label_dict, q_matrix_dict):
    q_matrix_dict_list = [q_matrix_dict] * len(language_codes) * language_nr_evaluation

    for step_rate_bool in [True,
                           False]:  # evaluation for all stepfactors (True) and for all random walker base rates (False) listed above
        if step_rate_bool:
            rdwalker_base_rate = 0
        elif not step_rate_bool:
            stepfactor = 0

        # a)evaluate solving rates for different stepfactors
        if step_rate_bool:
            for stepfactor in stepfactors:
                i = 0
                for lcode_eval, saving_folder in zip(language_codes, saving_folders):
                    # first calculate random walker rates (identical for all languages, but need to do several times, because of saving procedure..)
                    solving_steps_rd, solving_steps_smartrd = random_walker_rates(saving_folder, language_nr_evaluation,
                                                                                  step_rate_bool, rdwalker_base_rate,
                                                                                  stepfactor, label_dict)
                    gc.collect()
                    for language_nr in range(language_nr_evaluation):
                        for method in methods:
                            # next calculate rates for misinformed and informed students
                            student_performance_evaluator(lcode_eval, saving_folder, language_nr, method, random_repeat,
                                                          step_rate_bool, solving_steps_rd, rdwalker_base_rate,
                                                          stepfactor, label_dict, q_matrix_dict_list[i],
                                                          evalute_nonlinear_ae, evaluate_nonlinear_std)
                            gc.collect()
                        i += 1

        # a)evaluate solving rates for different baseline random walker baseline solving rates
        elif not step_rate_bool:
            for rdwalker_base_rate in rdwalker_base_rates:
                i = 0
                for lcode_eval, saving_folder in zip(language_codes, saving_folders):
                    # first calculate random walker steps for the baseline solving rate (identical for all languages, but need to do several times, because of saving procedure..)
                    solving_steps_rd, solving_steps_smartrd = random_walker_rates(saving_folder, language_nr_evaluation,
                                                                                  step_rate_bool, rdwalker_base_rate,
                                                                                  stepfactor, label_dict)
                    gc.collect()
                    for language_nr in range(language_nr_evaluation):
                        for method in methods:
                            # next calculate rates for misinformed and informed students and give them the steps the random walkers needed
                            student_performance_evaluator(lcode_eval, saving_folder, language_nr, method, random_repeat,
                                                          step_rate_bool, solving_steps_rd, rdwalker_base_rate,
                                                          stepfactor, label_dict, q_matrix_dict_list[i],
                                                          evalute_nonlinear_ae, evaluate_nonlinear_std)
                            gc.collect()
                        i += 1



