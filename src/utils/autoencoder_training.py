#@title 14. FUNCTIONS - Autoencoder training


import math as mt
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from autoencoder_def import MatrixDataset
from dqn_optimizers import *

'''
function to train the autoencoder network (includes several sub-loss functions)
'''

def train_autoencoder(autoencoder: nn.Module, optim: torch.optim, dataset: MatrixDataset, wall_state_dict:  'dict[int,list[int]]', gamma_sparse: float, zeta_std: float, kappa: float,
                      epochs: int, batch_size: Optional[int], student_incl: bool, train_order: int)-> torch.tensor:
    '''
    Function to train the autoencoder network (=language proxy)
    ---
    INPUT
    autoencoder - Autoencoder instance
    optim - the optimizer for the autoencoder network
    dataset - the q-matrix dataset used for training
    wall_state_dict - dictionary with indices as keys and lists representing wall states of a particular gridworld as values
    gamma_sparse - balances how much focus is put on sparsity loss compared to mse loss
    zeta_std - balances how much focus is put on student loss (finding the goal), compared to raw autoencoder loss
    kappa - balances how much focus is put on regularization, compared to the probability of finding the goal
    epochs - Number of training epochs
    batch_size - Batch size
    student_incl - Is the student network included in autoencoder training?
    train_order - 1: train only the messages with fixed student / 2: train only the student with fixed messages / 0:train both at the same time
    ---
    OUTPUT
    overall_losses: tensor with floats documenting training losses over time (one value per epoch)
    '''

    epoch_printout: int=25 #after how many do we print out the current losses
    big_next_states_dict={}
    probas_transformer_dict={}



    #some modules have to be frozen in case we only train the messaging (autoencoder) or the understanding (student)
    if student_incl:
        for i,module in enumerate(autoencoder.modules()):
            if i>0:
                if i<11:
                    module.requires_grad_(not (train_order==2))
                else:
                    module.requires_grad_(not (train_order==1))

    #initialize data
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=False) #drop_last=False means we include all data in training, the last batch is smaller each epoch

    #one loop through all the data to save some computing in every iteration later
    for matrix_batch, labels_batch in loader:
        for t in range(len(matrix_batch)):
            wall_label, init_state, goal_state=labels_batch[0][t].item(), labels_batch[1][t].item(), labels_batch[2][t].item()
            #get dictionaries
            env = SquareGridworld(init_state,goal_state,wall_state_dict[wall_label], lava)
            outcomes=env.get_outcomes()
            next_states_dict={s:[outcomes[s,a][0] for a in [0,1,2,3]]  for s in range(grid_dim**2)}
            #dictionary to retreive next state and reward given current (s,a)
            big_next_states_dict[(wall_label, goal_state)]=next_states_dict

            probas_transformer=torch.zeros(size=(n_actions*grid_dim**2, grid_dim**2*grid_dim**2))
            for s in range(grid_dim**2):
                if s==goal_state:
                    for a in range(n_actions):
                        probas_transformer[a*grid_dim**2+grid_dim*mt.floor(s/grid_dim)+s%grid_dim,s*grid_dim**2+s]=1
                else:
                    for a, ns in enumerate(next_states_dict[s]):
                        probas_transformer[a*grid_dim**2+grid_dim*mt.floor(s/grid_dim)+s%grid_dim,s*grid_dim**2+ns]=1
            probas_transformer_dict[(wall_label, goal_state)]=probas_transformer

    #now the actual training loop
    overall_losses, reconstruction_losses, sparsity_losses = [], [], []
    for epoch in trange(epochs, desc='Epoch'):
        loss_total, student_loss_total, reconstruction_loss_total, sparse_loss_total = 0,0,0,0
        for matrix_batch, labels_batch in loader:
            state_tensors=get_state_tensors(len(matrix_batch))



            optim.zero_grad(set_to_none=True) #set to none is very important as otherwise often some small gradients remain!!
            matrix_batch=matrix_batch.to(device)
            if student_incl:
                '''
                lp = LineProfiler()
                lp_wrapper = lp(autoencoder.forward_student)
                lp_wrapper(matrix_batch, state_tensors)
                lp.print_stats()
                '''
                messages, matrices_dec, matrices_std=autoencoder.forward_student(matrix_batch, state_tensors) #get decoded matrices by autoencoder
            else:
                messages, matrices_dec = autoencoder.forward(matrix_batch)

            # Loss calculation for the autoencoder - "batch" stands for the loss in this batch, "total" for the overall loss in this epoch (sum of all batches)
            reconstruction_loss_batch=reconstruction_loss(matrix_batch, matrices_dec)
            sparse_loss_batch = sparsity_loss(messages)
            if student_incl:
                '''
                lp = LineProfiler()
                lp_wrapper = lp(goal_finding_loss)
                lp_wrapper(matrices_std, labels_batch, wall_state_dict, kappa, big_next_states_dict, probas_transformer_dict)
                lp.print_stats()
                '''
                student_loss_batch=goal_finding_loss(matrices_std, labels_batch, wall_state_dict, kappa, big_next_states_dict, probas_transformer_dict)
                #student_loss_batch=reward_loss(matrices_std, labels_batch, wall_state_dict, kappa)
            #overall loss is a combination
            if student_incl:
                loss_batch=(1-gamma_sparse)*reconstruction_loss_batch+gamma_sparse*sparse_loss_batch+zeta_std*student_loss_batch
                #loss_batch=gamma_sparse*sparse_loss_batch+zeta_std*student_loss_batch
            if not student_incl:
                loss_batch=(1-gamma_sparse)*reconstruction_loss_batch+gamma_sparse*sparse_loss_batch

            #add losses of the current batch to the total of this epoch
            loss_total+=loss_batch.detach()
            reconstruction_loss_total+=reconstruction_loss_batch.detach()
            sparse_loss_total+=sparse_loss_batch.detach()
            if student_incl:
                student_loss_total+=student_loss_batch.detach()

            loss_batch.backward() #changing of the gradients according to the loss
            optim.step()

        overall_losses.append(loss_total) #list of autoencoder losses per epoch
        reconstruction_losses.append(reconstruction_loss_total)
        sparsity_losses.append(sparse_loss_total)
        #printouts of intermediate losses
        if epoch %epoch_printout==0:
            print(f"losses in epoch {epoch}:")
            if student_incl:
                print(f"losses are sparse:{gamma_sparse*sparse_loss_total}, reconstruction:{(1-gamma_sparse)*reconstruction_loss_total}, student:{zeta_std*student_loss_total}")
            if not student_incl:
                print(f"losses are sparse:{gamma_sparse*sparse_loss_total}, reconstruction:{(1-gamma_sparse)*reconstruction_loss_total}")
    #transfer to cpu -Why?
    autoencoder.to('cpu')
    return torch.tensor(overall_losses).cpu(), torch.tensor(reconstruction_losses).cpu(), torch.tensor(sparsity_losses).cpu()


def reconstruction_loss(matrices: torch.tensor, matrices_dec: torch.tensor)->torch.tensor:
    '''
    Loss for reproduction accuracy of input for the autoencoder
    ---
    INPUT
    matrices - torch.tensor. input q-matrices
    matrices_dec - torch.tensor. reconstructed q-matrices (from the messages)
    ---
    OUTPUT
    loss - the loss
    '''
    loss = torch.norm(matrices_dec-matrices,2)
    return loss

def sparsity_loss(messages: torch.tensor)->torch.tensor:
    '''
    Loss for sparsity of the messages
    ---
    INPUT
    messages - torch.tensor. the messages
    ---
    OUTPUT
    loss - the loss
    '''
    loss = torch.norm(messages,1)
    return loss




def goal_finding_loss(matrices_std_batch: torch.tensor, labels_batch: torch.tensor, wall_state_dict: 'dict[int,list[int]]', kappa: float, big_next_states_dict, probas_transformer_dict)->float:
    '''
    Loss for how accurately the student finds the goal given the message
    ---
    INPUT
    matrices_std - batch of q-matrices of the student
    labels - batch of labels from the labels dictionary corresponding to the batch of student matrices
    wall_state_dict - dictionary with indices as keys and lists representing wall states of a particular gridworld as values
    kappa - balances how much focus is put on regularization, compared to the probability of finding the goal
    ---
    OUTPUT
    loss - the loss
    '''
    student_loss=torch.zeros(size=(1,), dtype=float).to(device)[0]
    #iterate through all the tasks
    print_output=[] #this is what we print out each iteration
    softy=nn.Softmax(dim=0)
    for t in range(len(matrices_std_batch)):
        #load data
        Q=matrices_std_batch[t]
        wall_label, init_state, goal_state, opt_steps=labels_batch[0][t].item(), labels_batch[1][t].item(), labels_batch[2][t].item(), labels_batch[3][t].item()
        wall_states=wall_state_dict[wall_label]
        #first build the network architecture
        next_states_dict=big_next_states_dict[(wall_label, goal_state)]
        matrix_big=torch.zeros(size=(grid_dim**2,grid_dim**2), device=device) #create a big transition matrix
        #get action probabilities as softmax of the Q-values
        action_probas=torch.flatten(softy(Q)).unsqueeze(dim=0)

        #print(wall_states, goal_state)
        matrix_big=action_probas@probas_transformer_dict[(wall_label, goal_state)]
        '''
        for i in range(64):
            for j in range(256):
                hi=probas_transformer_dict[(wall_label, goal_state)][i][j]
                if hi>0:
                    print(i,mt.floor(j/16), j%16)

        sys.exit()
        '''
        matrix_big=torch.transpose(torch.unflatten(input=matrix_big, dim=1, sizes=(grid_dim**2, grid_dim**2)).squeeze(),0,1)

        #For each student step, Apply the transition matrix to the initial probability distribution to get the new probability distribution
        goal_proba=torch.linalg.matrix_power(matrix_big,opt_steps)[goal_state,init_state]

        student_loss+=(1-kappa)*(1-goal_proba)**4 #first part - probability to find the goal (difference to 1)
        student_loss+=kappa/mt.sqrt(grid_dim**2*n_actions)*torch.norm(Q,2) #second part - keep overall Q-values low to avoid some local minima and enhance overall stability

    return student_loss



def reward_loss(matrices_std_batch: torch.tensor, labels_batch: torch.tensor, wall_state_dict: 'dict[int,list[int]]', kappa: float)->float:
    '''
    Loss for how accurately the student finds the goal given the message
    ---
    INPUT
    matrices_std - batch of q-matrices of the student
    labels - batch of labels from the labels dictionary corresponding to the batch of student matrices
    wall_state_dict - dictionary with indices as keys and lists representing wall states of a particular gridworld as values
    kappa - balances how much focus is put on regularization, compared to the probability of finding the goal
    ---
    OUTPUT
    loss - the loss
    '''
    reward_loss=torch.zeros(size=(1,), dtype=float).to(device)[0]
    #iterate through all the tasks
    print_output=[] #this is what we print out each iteration
    softy=nn.Softmax(dim=0)
    for t in range(len(matrices_std_batch)):
        #load data
        Q=matrices_std_batch[t]
        wall_label, init_state, goal_state, opt_steps=labels_batch[0][t].item(), labels_batch[1][t].item(), labels_batch[2][t].item(), labels_batch[3][t].item()
        wall_states=wall_state_dict[wall_label]
        #first build the network architecture
        env = SquareGridworld(init_state,goal_state,wall_states, lava)
        outcomes = env.get_outcomes()
        #dictionary to retreive next state and reward given current (s,a)
        next_states_rewards_dict={s:[outcomes[s,a] for a in [0,1,2,3]]  for s in range(grid_dim**2)}
        matrix_big=torch.zeros(size=(grid_dim**2,grid_dim**2), device=device) #create a big transition matrix
        #get action probabilities as softmax of the Q-values
        action_probas=softy(Q)
        state_rewards=torch.zeros(grid_dim**2, device=device)
        for s in range(grid_dim**2):
            if s!=goal_state:
                for i,[ns,r] in enumerate(next_states_rewards_dict[s]):
                    current_action_probas=action_probas[i,mt.floor(s/grid_dim),s%grid_dim]
                    state_rewards[s]+=current_action_probas*r
                    matrix_big[ns,s]+=current_action_probas #need to have the "+=" here because there can be multiple identical transitions, e.g. for corner states
            else:
                matrix_big[s,s]=1

        #initialize state occupancy probabilities
        probas=torch.zeros(grid_dim**2, device=device)
        probas[init_state]=1
        reward=0

        #For each student step, Apply the transition matrix to the initial probability distribution to get the new probability distribution
        for rep in range(opt_steps): #strict: use the opt_steps here as well!
            reward+=torch.dot(probas,state_rewards) #for each step, add the average obtained reward in this step
            probas=matrix_big@probas
        reward+=probas[goal_state]*goal_reward #finally add the goal reward

        #put worst and best possible reward for the student and from the actual obtained reward create a "probability", like in the goal probability loss
        '''
        best_reward=opt_steps*step_reward+reward
        worst_reward=opt_steps*(wall_reward+step_reward)
        reward_proba=(reward-worst_reward)/(best_reward-worst_reward)
        reward_loss+=(1-kappa)*(1-reward_proba)**4 #same method as for goal finding probability
        '''
        reward_loss+=(1-kappa)*(goal_reward+opt_steps*step_reward-reward)**2 #first part - maximise obtained reward
        reward_loss+=kappa/mt.sqrt(grid_dim**2*n_actions)*torch.norm(Q,2) #second part - keep overall Q-values low to avoid some local minima and enhance overall stability
        print_output+=[round(goal_reward+opt_steps*step_reward-reward.item(),2)] #optimize on probability but still print the lost reward output
    print(print_output)
    return reward_loss


def plot_losses(losses: list):
    '''
    plotting function for the autoencoder training losses (per epoch)
    losses are plotted on a log scale
    ---
    INPUT
    losses: Log of overall losses during the process of the model
    ---
    OUTPUT
    Nothing but the plot
    '''
    plt.figure()
    plt.plot(losses)
    plt.legend(['Autoencoder training loss progression'])
    plt.xlabel('Training batch number')
    plt.ylabel('Losses')
    plt.show()


def task_indices_sorter(label_dict, train_worlds: 'list[int]', train_goals:  'list[int]')->'list[list[int]]':
    '''
    Sorts indices of the tasks according to whether the agent knows the world or the goal
    ---
    INPUT
    label_dict - dictionary with indices (of gridworld "tasks") as keys and corresponding tuple (wall_index, initial_state, goal_state)
                 as value (wall index comes from the wall_state_dict)
    train_worlds - indices of worlds that the network should train on
    train_goals - indices of goal locations the network should train on (only in the worlds specified by train_worlds!)
    ---
    OUTPUT
    goal_world_indices - indices of tasks that the network was trained on
    goal_noworld_indices - indices of tasks where the network was trained on the goal location, but not on the world
    nogoal_world_indices - indices of tasks where the network was trained on the world, but not on the goal location
    nogoal_noworld_indices - indices of tasks where the network was trained on neither goal location nor world
    '''

    #indices marking tasks where the agent has seen the goal location or the world before (or both or neither)
    goal_world_inds=[]
    goal_noworld_inds=[]
    nogoal_world_inds=[]
    nogoal_noworld_inds=[]
    #iterate over all possible tasks
    for task_index, [world,init,goal,mindist] in label_dict.items():
        if goal==student_init: #do not want to train or evaluate student performance on tasks that are always solved trivially
                continue
        if world in train_worlds:
            if goal in train_goals:
                goal_world_inds+=[task_index]
            else:
                nogoal_world_inds+=[task_index]
        else:
            if goal in train_goals:
                goal_noworld_inds+=[task_index]
            else:
                nogoal_noworld_inds+=[task_index]

    return [goal_world_inds, goal_noworld_inds, nogoal_world_inds, nogoal_noworld_inds]