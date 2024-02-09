#@title 12. FUNCTIONS - Q matrix generation from gridworlds

import math as mt
import random
from collections import deque
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from dqn_optimizers import *
from graph_operations import *
from policies import *


def q_matrix_generator(label_dict: 'dict[int,tuple[int, int, int]]', wall_state_dict: 'dict[int,list[int]]', perfect_qdict: 'dict[int, torch.tensor]', matrix_accuracy: float, max_attempts: int,
                       batch_size: Optional[int]=None, num_eps: int=200, loss_norm=nn.MSELoss(), max_steps: int=50, memory_size_short=L, alpha: float=16, learning_rate=lr_teacher)->'dict[int, torch.tensor]':
    '''
    For each maze task stored in the label dict, a (near-) optimal Q-matrix is created by the DQN and stored in the dictionary
    ---
    INPUT
    label_dict -dictionary with indices representing tasks as keys and corresponding tuple (wall_index, initial_state, goal_state)
                as value (wall index is key for wall_state_dict)
    wall_state_dict: dictionary with "wall indices" as keys and lists representing wall states of a particular gridworld as values
    perfect_qdict: the perfect q-matrices created by q_matrix_generator_deterministic (keyed by task integer), in the calculations we make sure that the teacher's q-matrix
                             is not too far off the perfect q-matrix
    batch_size: batching of transitions in network learning ("None" means there is no batching and we do full gradient descent with all transitions)
    num_eps: Number of episodes the (teacher) agent gets
    loss_norm: The norm that is used for Q-learning training on the transitions in the memory
    max_steps: Maximum steps the agent gets per episode
    memory_size_short: Size of the short-term memory
    alpha: constant in the "optimism in the face of uncertainty"-policy
    learning_rate: learning rate for the optimizer of the DQN
    matrix_accuracy: Guideline for how close (in terms of 1-norm) the teacher's q-matrix has to be to the perfect q-matrix to be accepted
    ---
    OUTPUT
    qdict: dictionary with gridworld task indices as keys and corresponding q-matrices as values
    '''

    qdict: 'dict[int, torch.tensor]'={}
    for task, [wall_index, init_state, goal_state] in label_dict.items():
        wall_states=wall_state_dict[wall_index]
        print(f"task is {task}")
        G=graph_from_walls(wall_states)
        #when the graph is not connected, then the wall states cut off some regular states -> we don't analyze this gridworld
        if not nx.is_connected(G):
            continue
        '''
        initialization
        '''

        #initialize environment
        env = SquareGridworld(init_state,goal_state,wall_states, lava)
        outcomes=env.get_outcomes()
        #initialize softmax and student probabilities
        softy=nn.Softmax(dim=0)

        #dictionary for faster switching between int and tuple representations of states
        state_int_to_tuple_dict={s:state_int_to_tuple(s) for s in range(grid_dim**2)}
        state_int_to_tuple_dict[None]=None

        transition_index_dict, batches_list=transition_memories(init_state, goal_state, wall_states)

        '''
        execution
        '''
        #We deem a q-matrix "good" if it is not too far off the "perfect" Q-matrix for the task (which was generated deterministically)
        good_qmatrix=False
        counter=0
        while counter<max_attempts and (not good_qmatrix):

            ep_steps=np.zeros(num_eps)
            teacher = DQN(0).to(device)
            message=torch.tensor([]).to(device) #teacher gets no messages
            optimizer = torch.optim.Adam(teacher.parameters(), lr=learning_rate, weight_decay=0)

            #initialize short-term and long-term memories
            memory_short = deque([],maxlen=memory_size_short)
            memory_long = deque([],maxlen=n_actions*grid_dim**2)
            #record actions already taken
            sa_counts={(s,a):0 for s in range(grid_dim**2) for a in range(n_actions)} #state action counter
            goalfound=False
            for ep in range(num_eps):
                #each episode starts at a random state to guarantee that the perfect Q-matrix is found!
                state_int=random.choice([s for s in range(grid_dim**2) if not (s in wall_states) and s!=goal_state])
                for t in range(max_steps):
                    #Choose the next action as combination of instructions and own experience plus exploration of the unknown
                    action=select_action_optimism(teacher, state_int, message ,alpha, sa_counts)
                    #observe outcomes from the environment (next state and immediate reward)
                    next_state_int, reward = outcomes[state_int,action]

                    #transform them to the input shape required by the teacher network
                    reward = torch.tensor([[reward]], device=device)
                    action = torch.tensor([[action]], device=device)

                    #episode is finished once we reached the goal
                    if next_state_int==None:
                        #add the goal to the memory if we encounter it for the first time -> if yes then include all four goal transitions into learning
                        if not goalfound:
                            goalfound=True
                        ep_steps[ep]=t
                        break

                    else:
                        #add experience to the memory
                        memory_short.append(transition_index_dict[state_int,action.item()])
                        if sa_counts[state_int,action.item()]==0:
                            memory_long.append(transition_index_dict[state_int,action.item()])

                        #episode is also finished if we took the maximum number of steps
                        if t==max_steps-1:
                            ep_steps[ep]+=max_steps
                        # Move to the next state
                        sa_counts[(state_int,action.item())]+=1
                        state_int = next_state_int
                    # Perform one step of the optimization (on the teacher network)
                    memory=torch.tensor(list(memory_short)+list(memory_long)).to(device)
                    '''
                    lp2 = LineProfiler()
                    lp2_wrapper = lp2(optimize_dqn)
                    lp2_wrapper(student, optimizer, memory, batches_list, goalfound, loss_norm)
                    lp2.print_stats()
                    '''
                    current_loss=optimize_dqn(teacher, optimizer, memory, batches_list, goalfound, loss_norm)

            #create student Q-matrix and print out
            q_matrix=q_matrix_from_network(teacher, message ,wall_states)
            difference_to_perfect=torch.linalg.vector_norm(q_matrix-perfect_qdict[task], ord=1)
            print(f"difference to perfect Q-matrix: {difference_to_perfect}")
            if difference_to_perfect < matrix_accuracy:
                good_qmatrix=True #if False, we repeat the calculation, the final Q-matrix was not "good enough"
                qdict[task]=q_matrix
                q_array=np.round_(np.flip(q_matrix.detach().numpy(),1).copy(),3) #flip dimension so it has the correct maze form when printing
                print(f"final q-matrix of student is {q_array}")
            elif counter==max_attempts-1:
                qdict[task]=perfect_qdict[task]
                counter+=1
            else:
                counter+=1
        #plt.plot(range(num_eps),ep_steps)
        plt.show()
    return qdict



def q_matrix_from_network(network: nn.Module, message: torch.tensor, wall_states:'list[int]')->torch.tensor:
    '''
    generate the entire q-matrix of a task from the (trained) network
    ---
    INPUT
    network - a teacher network that gets a state as input and outputs the four corresponding q-values
    message - the message the network may have received
    wall_states - list of the wall states of the gridworld under consideration
    ---
    OUTPUT
    q_matrix - the q-matrix
    '''

    q_matrix=torch.zeros(size=(4,grid_dim,grid_dim))
    for s in range(grid_dim**2):
        indx,indy=mt.floor(s/grid_dim),s%grid_dim
        if (not (s in wall_states)) or lava: #lava states are accessible, so we have trained teacher q-values for them
            q_matrix[:,indx,indy]=network(torch.cat((state_int_to_tuple(s)[0],message),0))
        else:
            q_matrix[:,indx,indy]=0*network(state_int_to_tuple(s)) #add zeros as placeholder values for the inaccessible wall states

    return q_matrix



def q_matrix_generator_deterministic(label_dict: 'dict[int,tuple[int, int, int]]', wall_state_dict: 'dict[int,list[int]]')->'dict[int, torch.tensor]':
    '''
    function to create (perfect) Q-matrices for maze tasks in a deterministic way - they can then be compared to the
    calculated Q-matrices of the teacher DQN (or student)
    ---
    INPUT
    label_dict: dictionary with indices representing tasks as keys and corresponding tuple (wall_index, initial_state, goal_state)
                as value (wall index is key for wall_state_dict)
    wall_state_dict: dictionary with "wall indices" as keys and lists representing wall states of a particular gridworld as values
    ---
    OUTPUT
    qdict - dictionary with indices of gridworld tasks as keys and corresponding "perfect" deterministic q-matrices as values

    '''
    qdict={} #final outputs

    for task, [wall_index, init_state, goal_state] in label_dict.items():
        wall_states=wall_state_dict[wall_index]
        G=graph_from_walls(wall_states)
        if not nx.is_connected(G):
            continue

        #initialize environment
        env = SquareGridworld(init_state,goal_state,wall_states, lava)
        outcomes=env.get_outcomes()

        #create q-matrix from the network results
        q_matrix=torch.zeros(size=(4,grid_dim,grid_dim))
        v_value_dict={} #value function for all the states
        v_value_dict[goal_state]=goal_reward

        #first find the value function v(s)
        for s in range(grid_dim**2):
            if (lava or not (s in wall_states)) and not s==goal_state:
                reverse_goal_path=nx.dijkstra_path(G, s, goal_state, weight='weight')
                reverse_goal_path.reverse() #reverse the path to move backwards step by step from the goal
                v_value=goal_reward
                for k in range(len(reverse_goal_path)-1): #iterate through the edges
                    edge_reward=-G.edges[(reverse_goal_path[k], reverse_goal_path[k+1])]['weight']
                    v_value=gamma_bellman*v_value+edge_reward #discount every step and add the (negative) reward
                v_value_dict[s]=v_value

        #then generate q(s,a) from v(s)
        for s in range(grid_dim**2):
            indx,indy=mt.floor(s/grid_dim),s%grid_dim
            if s in wall_states:
                q_matrix[:,indx,indy]=torch.tensor([0,0,0,0], device=device)
            elif s==goal_state:
                q_matrix[:,indx,indy]=torch.tensor([goal_reward, goal_reward, goal_reward, goal_reward], device=device)
            else:
                for a in range(4):
                    next_state, reward=outcomes[s,a]
                    q_matrix[a,indx,indy]=gamma_bellman*v_value_dict[next_state]+reward

        qdict[task]=q_matrix
    print(f"Completed calculation of deterministic Q-matrices")
    return qdict
