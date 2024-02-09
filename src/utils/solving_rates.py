#@title 16. FUNCTIONS - Student solving rates


import math as mt
import os  # for creating directories

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from autoencoder_def import ConvAutoEncoder
from dqn_optimizers import *
from graph_operations import *


def student_performance_evaluator(language_code, saving_folder, language_nr, method, random_repeat: int, step_rate_bool: bool, solving_steps, rdwalker_base_rate, stepfactor, label_dict, q_matrix_dict, evaluate_nonlinear_ae, evaluate_nonlinear_std):
    '''
    Evaluates a language on the performance of the student by generating solving rates for all the possible tasks
    Three students are investigates:
    -informed student that was trained and gets the correct message
    -misinformed student that was trained and gets a random message from the message dictionary
    -untrained student that was not trained, but gets the correct message
    ---
    INPUT
    language_code - String. Identifier of the autoencoder network (language proxy) in the stored data
    saving_folder - String. Identifier of the folder where the solving rates should be saved
    language_nr - integer. How many languages were generated for this set of trained worlds and trained tasks (to reduce variance)
    method - either "no_learning" or "simple_learning"
    stepfactor - integer. how many times the number of steps in the optimal solution do we give the student before the episode is ended
    random_repeat - how many times is the random procedure for uninformed/misinformed students repeated?
    solving_steps - list of how many steps the random walker needed to obtain a certain solving rate - use this as baseline
    step_rate_bool - which method do we do, either the stepfactor method (True) or the random walker rate method (False) (for determining how many steps are allowed)
    evaluate_nonlinear_ae, evaluate_nonlinear_std - do the neurons in the autoencoder/the student network have nonlinear ReLU activations?
    ---
    OUTPUT
    None. (Store data of solving rates for all three students in a corresponding file.
    '''
    softy=nn.Softmax(dim=0)
    probas=torch.zeros(grid_dim**2, device=device)
    language_code=f"{language_code}_language{language_nr}"
    saving_folder=f"{saving_folder}/{saving_folder}_language{language_nr}"
    #iterate over languages that should be analysed


    #Load the autoencoder network
    autoencoder = ConvAutoEncoder(data_shape, K, evaluate_nonlinear_ae, evaluate_nonlinear_std)
    autoencoder.load_state_dict(torch.load(file_loc+"autoencoder/autoencoder network parameters/"f"params_autoenc{language_code}.pt"))
    autoencoder.eval()

    #create a message dictionary, with indices corresponding to the task indices
    message_dict={}
    for task_index,q_matrix in enumerate(q_matrix_dict.values()):
        q_matrix=torch.unsqueeze(q_matrix,0) #need this because the autoencoder always expects batches of inputs!
        message=autoencoder.encode(q_matrix)[0]
        message_dict[task_index]=message

    #state int-to-tuple dictionary
    state_int_to_tuple_dict={}
    for s in range(grid_dim**2):
        state_int_to_tuple_dict[s]=state_int_to_tuple(s)

    #get state tensors
    state_tensors=get_state_tensors(1)

    #create a dictionary containing all the student action probabilities (for all possible messages)
    action_probas_dict={}
    for message_index, message in message_dict.items():
        message=message_dict[message_index].to(device).detach()
        #create student Q-matrix and corresponding action probabilities
        Q=autoencoder.student(message, state_tensors)[0]
        action_probas=softy(Q)
        action_probas_dict[message_index]=action_probas


    #solving rates for the informed and misinformed students are saved in the corresponding arrays
    solving_rates = [np.zeros(len(label_dict)), np.zeros(len(label_dict))]
    solving_rates_worlds_avg=[[],[]]
    #iterate over gridworlds
    for world_index, wall_states in wall_state_dict.items():
        world_task_indices=[i for i,label in label_dict.items() if label[0]==world_index and label[2]!=student_init] #all tasks in this particular world
        world_quotas_info, world_quotas_random, world_quotas_misinfo =[], [], []
        G=graph_from_walls(wall_states)
        solving_rates_world = [np.zeros(len(world_task_indices)), np.zeros(len(world_task_indices)), np.zeros(len(world_task_indices))] #solving rates in this particular world
        #iterate over goal locations
        for j,task_index in enumerate(world_task_indices):
            goal_state=label_dict[task_index][2]
            #initialize environment
            env = SquareGridworld(student_init,goal_state,wall_states, lava)
            outcomes=env.get_outcomes()
            #dictionary to retreive next state and reward given current (s,a)
            next_states_dict={s:[outcomes[s,a][0] for a in [0,1,2,3]]  for s in range(grid_dim**2)}
            #print(wall_states, student_init, goal_state)
            goal_distance=nx.dijkstra_path_length(G,student_init,goal_state, weight=None)
            #allowed steps are the optimum steps times some factor (if factor is 0.5 need to try both and average)
            if not step_rate_bool:
                max_steps_list=[round(solving_steps[task_index]-0.500001), round(solving_steps[task_index]+0.499999)]
            elif step_rate_bool:
                if (stepfactor*goal_distance)%1<0.01:
                    max_steps_list=[round(stepfactor*goal_distance)]
                else:
                    max_steps_list=[round(stepfactor*goal_distance-0.1), round(stepfactor*goal_distance+0.1)]

            #iterate over informed (l=0) and misinformed (l=1) students
            for l in range(2):

                rdrep_fraction=random_repeat if l==1 else 1 #informed student does not need several repetitions, as it gets one fixed message

                for g in range(rdrep_fraction):
                    if l==0 and g>0:
                        continue

                    #random message for misinformed student, correct message for random and informed students
                    message_index=task_index if l==0 else np.random.randint(0,len(message_dict))
                    action_probas=action_probas_dict[message_index]

                    if method=="no_learning": #a stochastic method, actions have certain probabilities
                        matrix_big=torch.zeros(size=(grid_dim**2,grid_dim**2), device=device) #create a big transition matrix
                        for s in range(grid_dim**2):
                            if s!=goal_state:
                                for i,ns in enumerate(next_states_dict[s]):
                                    matrix_big[ns,s]+=action_probas[i,mt.floor(s/grid_dim),s%grid_dim] #need to have the "+=" here because there can be multiple identical transitions, e.g. for corner states
                            else:
                                matrix_big[s,s]=1

                    #iterate over the three students
                    for max_steps in max_steps_list:
                        max_steps=int(max_steps)

                        if method=="no_learning":
                            #initialize state occupancy probabilities
                            probas=0*probas
                            probas[student_init]=1
                            #For each student step, apply the transition matrix to the initial probability distribution to get the new probability distribution
                            for rep in range(max_steps):
                                probas=matrix_big@probas
                            if not step_rate_bool:
                                solving_rates_world[l][j]+=probas[goal_state]/rdrep_fraction*(1-abs(solving_steps[task_index]-max_steps))
                                solving_rates[l][task_index]+=probas[goal_state]/rdrep_fraction*(1-abs(solving_steps[task_index]-max_steps))
                            elif step_rate_bool:
                                solving_rates_world[l][j]+=probas[goal_state]/(rdrep_fraction*len(max_steps_list))
                                solving_rates[l][task_index]+=probas[goal_state]/(rdrep_fraction*len(max_steps_list))



                        elif method=="simple_learning": #a deterministic method (given the Q-matrix, we know exactly what the student's path will look like)
                            state=student_init
                            for step in range(max_steps):
                                #take the action with maximum Q-value
                                action=torch.argmax(action_probas[:,mt.floor(state/grid_dim),state%grid_dim])
                                #shift its Q-value so that it is now lowest of all the four actions
                                action_probas[action,mt.floor(state/grid_dim),state%grid_dim]=torch.min(action_probas[:,mt.floor(state/grid_dim),state%grid_dim])-1e-5
                                state=next_states_dict[state][action]
                                if state==goal_state:
                                    if not step_rate_bool:
                                        solving_rates_world[l][j]+=1/rdrep_fraction*(1-abs(solving_steps[task_index]-max_steps))
                                        solving_rates[l][task_index]+=1/rdrep_fraction*(1-abs(solving_steps[task_index]-max_steps))
                                    elif step_rate_bool:
                                        solving_rates_world[l][j]+=1/(rdrep_fraction*len(max_steps_list))
                                        solving_rates[l][task_index]+=1/(rdrep_fraction*len(max_steps_list))
                                    break

            #print(f"goal {goal_state}: solving rates info:{round(100*solving_rates[0][task_index])}%, misinfo:{round(100*solving_rates[1][task_index])}%")
        for i in [0,1]:
            if len(solving_rates_world[i]>0): #assume that task numbers per world are approx. equal and then do average
                solving_rates_worlds_avg[i]+=[np.mean(solving_rates_world[i])]
        print(f"in world {world_index} with walls {wall_states} and {len(world_task_indices)} tasks the solving rates are info:{round(100*np.mean(solving_rates_world[0]),2)}%, misinfo:{round(100*np.mean(solving_rates_world[1]),2)}%")

    print("We have the overall results:")
    print(f"Average solving rate for informed: ({round(100*np.mean(solving_rates[0]),2)} +/- {round(100*np.std(solving_rates[0])/mt.sqrt(len(solving_rates[0])),2)})%")
    print(f"Average solving rate for misinformed: ({round(100*np.mean(solving_rates[1]),2)} +/- {100*round(np.std(solving_rates[1])/mt.sqrt(len(solving_rates[1])),2)})%")
    print("")

    #save the results
    if save_rates:
        if not os.path.exists(file_loc+"student/"+saving_folder):
            os.mkdir(file_loc+"student/"+saving_folder)
        if not step_rate_bool:
            np.savetxt(file_loc+"student/"+saving_folder+f"/solving_rates_info_{method}_rate{rdwalker_base_rate}.txt", solving_rates[0])
            np.savetxt(file_loc+"student/"+saving_folder+f"/solving_rates_misinfo_{method}_rate{rdwalker_base_rate}.txt", solving_rates[1])
            np.savetxt(file_loc+"student/"+saving_folder+f"/solving_rates_worlds_info_{method}_rate{rdwalker_base_rate}.txt", solving_rates_worlds_avg[0])
            np.savetxt(file_loc+"student/"+saving_folder+f"/solving_rates_worlds_misinfo_{method}_rate{rdwalker_base_rate}.txt", solving_rates_worlds_avg[1])
        elif step_rate_bool:
            np.savetxt(file_loc+"student/"+saving_folder+f"/solving_rates_info_{method}_stepfactor{stepfactor}.txt", solving_rates[0])
            np.savetxt(file_loc+"student/"+saving_folder+f"/solving_rates_misinfo_{method}_stepfactor{stepfactor}.txt", solving_rates[1])
            np.savetxt(file_loc+"student/"+saving_folder+f"/solving_rates_worlds_info_{method}_stepfactor{stepfactor}.txt", solving_rates_worlds_avg[0])
            np.savetxt(file_loc+"student/"+saving_folder+f"/solving_rates_worlds_misinfo_{method}_stepfactor{stepfactor}.txt", solving_rates_worlds_avg[1])


def random_walker_rates(saving_folder, language_number, step_rate_bool, rdwalker_base_rate, stepfactor, label_dict):
    '''
    Evaluates a language on the performance of the student by generating solving rates for all the possible tasks
    Two students are investigates:
    -random walker that takes completely random actions
    -smart random walker that takes completely random actions, but never runs into a wall
    ---
    INPUT
    saving_folder - String. Identifier of the folder where the solving rates should be saved
    language_number - integer. How many languages were generated for this set of trained worlds and trained tasks (to reduce variance)
    method - either "no_learning" or "simple_learning"
    stepfactor - integer. how many times the number of steps in the optimal solution do we give the student before the episode is ended
    step_rate_bool - boolean. which method do we do, either the stepfactor method (True) or the random walker rate method (False) (for determining how many steps are allowed)
    rdwalker_base_rate - float. we calculate how many steps the random walkers need to achieve the solving rate of rdwalker_base_rate for each task
    ---
    OUTPUT
    None. (Store data of solving rates for the two students in a corresponding file.
    '''

    #state int-to-tuple dictionary
    state_int_to_tuple_dict={}
    probas=torch.zeros(grid_dim**2, device=device)
    for s in range(grid_dim**2):
        state_int_to_tuple_dict[s]=state_int_to_tuple(s)

    #solving rates for the three students are saved in the corresponding arrays
    solving_steps_rd, solving_steps_smartrd = np.zeros(len(label_dict)), np.zeros(len(label_dict))
    solving_rates_rd, solving_rates_smartrd = np.zeros(len(label_dict)), np.zeros(len(label_dict))
    solving_ratesworlds_rd, solving_ratesworlds_smartrd = np.zeros(len(wall_state_dict)), np.zeros(len(wall_state_dict))

    #iterate over gridworlds
    for world_index, wall_states in wall_state_dict.items():
        world_task_indices=[i for i,label in label_dict.items() if label[0]==world_index and label[2]!=student_init] #all tasks in this particular world
        solving_rates_rd_world, solving_rates_smartrd_world = np.zeros(len(world_task_indices)), np.zeros(len(world_task_indices))
        world_quotas_info, world_quotas_random, world_quotas_misinfo =[], [], []
        G=graph_from_walls(wall_states)
        #iterate over goal locations
        for j,task_index in enumerate(world_task_indices):
            goal_state=label_dict[task_index][2]
            #initialize environment
            env = SquareGridworld(student_init,goal_state,wall_states, lava)
            outcomes=env.get_outcomes()
            #dictionary to retreive next state and reward given current (s,a)
            next_states_dict={s:[outcomes[s,a][0] for a in [0,1,2,3]]  for s in range(grid_dim**2)}
            goal_distance=nx.dijkstra_path_length(G,student_init,goal_state, weight=None)

            #create matrices with random action probabilities
            matrix_big=torch.zeros(size=(grid_dim**2,grid_dim**2), device=device) #create a big transition matrix, each action has same likelihood
            matrix_big_nowall=torch.zeros(size=(grid_dim**2,grid_dim**2), device=device) #create a big transition matrix, wall actions are likelihood zero
            for s in range(grid_dim**2):
                if s!=goal_state and not (s in wall_states):
                    next_states=next_states_dict[s]
                    nowall_options=[ns for ns in next_states if ns!=s]
                    for ns in next_states:
                        matrix_big[ns,s]+=0.25 #need to have the "+=" here because there can be multiple identical transitions, e.g. for corner states
                    for ns in nowall_options:
                        matrix_big_nowall[ns,s]=1/len(nowall_options)

                else:
                    matrix_big[s,s]=1
                    matrix_big_nowall[s,s]=1

            #a) stepfactor method -> we give a stepfactor times shortest path length number of steps to solve the task.
            if step_rate_bool:
                #allowed steps are the optimum steps times some factor (if factor is 0.5 need to try both and average)
                if (stepfactor*goal_distance)%1<0.01:
                    max_steps_list=[round(stepfactor*goal_distance)]
                else:
                    max_steps_list=[round(stepfactor*goal_distance-0.1), round(stepfactor*goal_distance+0.1)]
                for max_steps in max_steps_list:
                    #initialize state occupancy probabilities
                    probas=0*probas
                    probas[student_init]=1
                    probas_nowall=torch.zeros(grid_dim**2, device=device)
                    probas_nowall[student_init]=1
                    for rep in range(max_steps):
                        #1. regular random walker
                        probas=matrix_big@probas
                        #2. smart random walker
                        probas_nowall=matrix_big_nowall@probas_nowall

                    solving_rates_rd[task_index]+=probas[goal_state]/len(max_steps_list)
                    solving_rates_rd_world[j]+=probas[goal_state]/len(max_steps_list)
                    solving_rates_smartrd[task_index]+=probas_nowall[goal_state]/len(max_steps_list)
                    solving_rates_smartrd_world[j]+=probas_nowall[goal_state]/len(max_steps_list)

            #b) random walker rate method -> we calculate the steps needed for the random walker to surpass a certain goal reaching probability
            if not step_rate_bool:
                #1.regular random walker
                prev_rate1, rate1, steps1=0,0,0
                while rate1 < rdwalker_base_rate:
                    steps1+=1
                    prev_rate1=rate1
                    #initialize state occupancy probabilities
                    probas=0*probas
                    probas[student_init]=1
                    #For each student step, apply the transition matrix to the initial probability distribution to get the new probability distribution
                    for rep in range(steps1):
                        probas=matrix_big@probas
                    rate1=probas[goal_state]
                #linear interpolation to find out the approximate step number for reaching rdwalker_base_rate
                solving_steps_rd[task_index]=steps1-1+(rdwalker_base_rate-prev_rate1)/(rate1-prev_rate1)

                #2.smart random walker
                prev_rate2, rate2, steps2=0,0,0 #for smart random walker
                while rate2<rdwalker_base_rate:
                    steps2+=1
                    prev_rate2=rate2
                    #initialize state occupancy probabilities
                    probas_nowall=torch.zeros(grid_dim**2, device=device)
                    probas_nowall[student_init]=1
                    #For each student step, apply the transition matrix to the initial probability distribution to get the new probability distribution
                    for rep in range(steps2):
                        probas_nowall=matrix_big_nowall@probas_nowall
                    rate2=probas_nowall[goal_state]
                #linear interpolation to find out the approximate step number for reaching rdwalker_base_rate
                solving_steps_smartrd[task_index]=steps2-1+(rdwalker_base_rate-prev_rate2)/(rate2-prev_rate2)

        #collect average solving rates in the single worlds
        solving_ratesworlds_rd[world_index]=np.mean(solving_rates_rd_world)
        solving_ratesworlds_smartrd[world_index]=np.mean(solving_rates_smartrd_world)
        if step_rate_bool:
            print(f"in world {world_index} with walls {wall_states} and {len(world_task_indices)} tasks the solving rates are rd walker:{round(100*np.mean(solving_rates_rd_world),2)}%, smart rd walker:{round(100*np.mean(solving_rates_smartrd_world),2)}%")



    #iterate over languages that should be analysed
    if not os.path.exists(file_loc+"student/"+saving_folder):
        os.mkdir(file_loc+"student/"+saving_folder)

    for language_nr in range(language_number):
        sv_folder=f"{saving_folder}/{saving_folder}_language{language_nr}"

        #save the results
        if save_rates:
            if not os.path.exists(file_loc+"student/"+sv_folder):
                os.mkdir(file_loc+"student/"+sv_folder)
            if step_rate_bool:
                np.savetxt(file_loc+"student/"+sv_folder+f"/solving_rates_rdwalker_stepfactor{stepfactor}.txt", solving_rates_rd)
                np.savetxt(file_loc+"student/"+sv_folder+f"/solving_rates_rdwalkersmart_stepfactor{stepfactor}.txt", solving_rates_smartrd)
                np.savetxt(file_loc+"student/"+sv_folder+f"/solving_rates_worlds_rdwalker_stepfactor{stepfactor}.txt", solving_ratesworlds_rd)
                np.savetxt(file_loc+"student/"+sv_folder+f"/solving_rates_worlds_rdwalkersmart_stepfactor{stepfactor}.txt", solving_ratesworlds_smartrd)
            if not step_rate_bool:
                np.savetxt(file_loc+"student/"+sv_folder+f"/solving_steps_rdwalker_rate{rdwalker_base_rate}.txt", solving_steps_rd)
                np.savetxt(file_loc+"student/"+sv_folder+f"/solving_steps_rdwalkersmart_rate{rdwalker_base_rate}.txt", solving_steps_smartrd)
    return solving_steps_rd, solving_steps_smartrd