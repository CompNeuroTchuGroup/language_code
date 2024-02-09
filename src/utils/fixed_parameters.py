import math as mt

import numpy as np
import torch

from changeable_parameters import *

#-----------------------------------------------------------------------------------------------------------------------
#@title 3. INIT - Fixed parameters
#-----------------------------------------------------------------------------------------------------------------------

file_loc = '/home/pietro/Documents/Mainz/Project_9_language/language_code/src/data/'

grid_dim: int=4 #side length of the square gridworld
n_actions: int=4 #how many actions are possible in each state
student_init: int=0 #initial state where the student is always starting - states are indexed as follows (example for the 4x4 mazes):
# 12 13 14 15
#  8  9 10 11
#  4  5  6  7
#  0  1  2  3
lava: bool=False #do we use lava states - i.e. accessible wall states - (True) or wall states (False)? (we now always use "False")
data_shape: 'tuple[int,int,int]'=(n_actions,grid_dim,grid_dim) #Q matrix shape


#rewards in the gridworld
step_reward: float=-0.1 #for taking a step
goal_reward: float=2. #for reaching the goal
wall_reward: float=-0.5 #for bumping into a wall


#teacher network parameters for learning Q-matrices
qmat_gen: bool=False #generate new Q-matrices?
accuracy=20 #how accurate do the final Q-matrices have to be compared to the "perfect" ones, which can be calculated by table lookup? (in terms of vector 1-norm)
max_attempts=3 #limit the attempts we give the teacher to calculate a Q-matrix that lies within the accuracy bound
wall_state_dict={0:[]} #only in case of new Q-matrix generation, we index all the mazes, and calculate a Q-matrix for every possible goal location in each maze
gamma_bellman: float=0.99 #temporal discount factor used in the Bellman equation of the teacher networks
L: int=50 #short term memory size of teacher learning Q-matrices
lr_teacher: float=3e-3 #learning rate for the teachers


#autoencoder network parameters for generating the language
language_gen: bool=False #generate a new language?
K: int=5 #length of the message vectors
gamma_sparse=1/20*mt.sqrt(grid_dim**2*n_actions/K) #how important is the sparsity loss compared to reconstruction loss
                                                   #we have the square root here to compare individual entries (in both cases take the overall 2-norm)
kappa=1/500 #balances how much focus is put on regularization, compared to the probability of finding the goal (for the student)
learning_rate_autoenc=5e-4 #learning rate of the autoencoder-student network
training_epochs=50 #number of training epochs for the autoencoder
zeta_std=5 #the zeta parameter (std: "student") for the loss function

#message space PCA (and t-SNE) plots
do_tsne_message_plots=True #do t-SNE plots as well?
do_umap_message_plots=True #do UMAP plots as well?
plot_worlds=range(16) #all these worlds are included in the main plot - can not be more than 20 at the moment due to the color cycler -> range(16) are all training worlds
plot_worlds_single=[6] #create individual plots for all of the worlds included in this list
nonlinear_ae_plots=True if language_code.__contains__("nonlinear") else False #does the autoencoder have nonlinear activations?
nonlinear_std_plots=True if (language_code.__contains__("nonlinear") or language_code.__contains__("linear_ae")) else False #does the student have nonlinear activations?
save_message_plots=False #save the plots?

#evaluate the student agent's performance (calculate goal finding success rates)
student_evaluate: bool=False
save_rates: bool=True #save the goal finding success rates

#autoencoder loss plots
epskip=50 #skip first few epochs to have narrower loss range and see details better
save_autoenc_lossplots=False #save the autoencoder plots?

#plot the student performances for different amounts of steps allowed for the students ("stepfactors")
folder_stepfactor_plots="nonlinear_goallocs0_zeta5_factor1"
language_nr_stepfactor_plots=5
stepfactor_list_plots=[1,1.5,2,2.5,3,3.5,4]
rdrates_list_plots=[]
save_stepfactor_plots=False #save the plots?

#plot the student performances for different groups of goal locations trained
goal_groups_plots=np.array([0,1,2,3,4,5,6]) #include those groups in the plot
stepfactor_goalloc_plots=2
rdrate_goalloc_plots=0.25
save_goalloc_plots=False #save the plots?

#closing the loop
closingloop_nonlinear_ae=True #do the neurons in the autoencoder have nonlinear activations?
closingloop_nonlinear_std=True #do the neurons in the student have nonlinear activations?
save_closingloop_plots=False #save the plots?

#topographic similarity analysis (Fig. 3b,c)
norm_topo=2 #which norm to use for meaning distances, i.e. task vectors and Q-matrices
n_bins_topo=20 #how many bins on the x-axis of the plot?

#entropy analysis through binning (Fig. 3e)
H_binlist=np.linspace(8,62,10).astype(int) #list of number of bins to calculate entropy from