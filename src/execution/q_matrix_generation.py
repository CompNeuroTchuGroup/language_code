# @title 17. EXECUTION - Q-matrix generation by teacher agents using Reinforcement Learning

import networkx as nx

from fixed_parameters import *
from changeable_parameters import *

from helpers import *
from qmatrix_gridworlds import *


def create_label_dictionary(wall_state_dict):
    # create label dictionary (give index to all tasks in the mazes, including goal state and wall states)
    task_index = 0
    label_dict = {}
    for wall_index, wall_states in wall_state_dict.items():
        G = graph_from_walls(wall_states)
        # when the graph is not connected, then the wall states cut off some regular states -> we don't analyze this gridworld
        if not nx.is_connected(G):
            continue
        for goal_state in range(grid_dim ** 2):
            if goal_state == student_init:  # start location can not be a goal -> would be trivially solved
                continue
            elif goal_state in wall_states:  # wall states can not be a goal -> they are not accessible for the agents
                continue
            else:
                label_dict[task_index] = [wall_index, student_init, goal_state]
                task_index += 1

    return label_dict


def generate_qmat(wall_state_dict, qmat_gen):
    if qmat_gen:
        label_dict = create_label_dictionary(wall_state_dict)
        # first generate the perfect deterministic Q-matrices
        perfect_qdict = q_matrix_generator_deterministic(label_dict, wall_state_dict)
        '''
        lp = LineProfiler()
        lp_wrapper = lp(q_matrix_generator)
        lp_wrapper(wall_state_dict, perfect_q_matrices_dict)
        lp.print_stats()
        '''
        # Then use the DQN and the perfect matrices as a check
        q_matrix_dict = q_matrix_generator(label_dict, wall_state_dict, perfect_qdict, accuracy, max_attempts)
        # write calculated q matrices into file
        write_dict_into_pkl(q_matrix_dict,
                            file_loc + "teacher/q matrix dictionaries/" + f"q_matrices{qmat_save_code}.pkl")
        write_dict_into_pkl(label_dict,
                            file_loc + "teacher/label dictionaries/" + f"q_matrices_labels{qmat_save_code}.pkl")
        write_dict_into_pkl(wall_state_dict,
                            file_loc + "teacher/wall state dictionaries/" + f"wall_states{qmat_save_code}.pkl")
    if not qmat_gen:
        # read Q-matrices from a pre-generated file
        wall_state_dict = read_dict_from_pkl(
            file_loc + "teacher/wall state dictionaries/" + f"wall_states{qmat_read_code}.pkl")
        q_matrix_dict = read_dict_from_pkl(
            file_loc + "teacher/q matrix dictionaries/" + f"q_matrices{qmat_read_code}.pkl")
        label_dict = read_dict_from_pkl(
            file_loc + "teacher/label dictionaries/" + f"q_matrices_labels{qmat_read_code}.pkl")
        # transfer Q-matrices to correct device
        for key in q_matrix_dict.keys():
            q_matrix_dict[key] = q_matrix_dict[key].to(device)

    return q_matrix_dict, wall_state_dict, label_dict


def insert_shortest_path_length(label_dict):
    # insert shortest path length from initial state to goal state into label dictionary (needed for language training with student feedback)
    newlabel_dict = {}
    for i, label in label_dict.items():
        walls, init, goal = label
        wall_states = wall_state_dict[walls]
        G = graph_from_walls(wall_states)
        o = node_dist(G, init, goal)
        newlabel_dict[i] = [walls, init, goal, o]
    return newlabel_dict




qmat_save_code: str = "dummy"  # file for saving newly generated Q-matrices (if qmat_gen=True)
q_matrix_dict, wall_state_dict, label_dict = generate_qmat(wall_state_dict, qmat_gen)
label_dict = insert_shortest_path_length(label_dict)
