# @title 8. FUNCTIONS - Policies


import math as mt
import random

import numpy as np
import torch
import torch.nn as nn

from dqn import DQN
from gridworld import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action_epsgreedy(agent: DQN, state: torch.tensor, message: torch.tensor, epsilon: float) -> int:
    '''
    epsilon-greedy algorithm for action selection
    ---
    INPUT
    agent - the deep-Q-network representing the agent exploring the world
    state - the state the agent is currently in (in coordinate representation)
    message - the message the agent has received
    epsilon - epsilon from the epsilon greedy algorithm
    ---
    OUTPUT
    The action to be taken (0:right, 1:up, 2:left, 3:down)
    '''
    if random.random() > epsilon:
        with torch.no_grad():
            input = torch.cat((state[0], message), 0)
            return agent(input).argmax().item()  # do forward pass and return optimal action found
    else:
        return random.choice(range(4))  # return random action


def select_action_optimism(agent: DQN, state_int: int, message: torch.tensor, alpha: float,
                           sa_counts: 'dict[tuple[int, int],int]'):
    '''
    optimism-in-face-of-uncertainty algorithm for action selection
    using the unchanged Q-values and no probabilities
    ---
    INPUT
    agent - the deep-Q-network representing the agent exploring the world
    state_int - the state the agent is currently in (in integer representation)
    message - the message the agent has received
    alpha - constant for optimism in face of uncertainty algorithm
    sa_counts - counts how often each state-action combination has already been seen by the agent
    ---
    OUTPUT
    The action to be taken (0:right, 1:up, 2:left, 3:down)
    '''
    state = state_int_to_tuple(state_int)
    current_sa_counts = torch.tensor([sa_counts[(state_int, a)] for a in range(4)]).to(
        device)  # pick out counts of current state
    with torch.no_grad():
        input = torch.cat((state[0], message), 0)
        qvals = agent(input)  # do forward pass
        qvals = qvals + alpha / torch.sqrt(current_sa_counts)  # add current uncertainties
        return qvals.argmax().item()  # return optimal action given uncertainties


def select_action_eps_trust(agent: DQN, state_int: int, probas_message_matrix: torch.tensor, softy: nn.Softmax,
                            iota: float,
                            trust: float, current_ep: int, total_eps: int) -> 'tuple[torch.tensor, torch.tensor, int]':
    '''
    global trust parameter for teacher and epsilon greedy exploration
    ---
    INPUT
    agent - the deep-Q-network representing the agent exploring the world
    state_int - the state the agent is currently in (in integer representation)
    probas_message_matrix - matrix of action probabilities according to the message
    softy - a softmax function
    iota - parameter to adjust student Q-values, so that softmax gives reasonable probabilities
    trust - tells how much the student trusts the teacher (between 0 and 1)
    current_ep - current episode number
    total_eps - total episode number (both parameters relevant for decaying epsilon in epsilon-greedy)
    ---
    OUTPUT
    probas_message - action probabilities in current state according to the message
    probas_student - action probabilities in current state according to the student's own learning
    action - next action that will be executed by the student
    '''
    state = state_int_to_tuple(state_int)
    with torch.no_grad():
        # 1.message
        probas_message = probas_message_matrix[:, mt.floor(state_int / grid_dim),
                         state_int % grid_dim]  # filter out current state
        # 2.student
        qvals_student = agent(state[0])
        qvals_student -= min(qvals_student) * torch.ones(n_actions).to(device)
        probas_student = softy(
            iota * qvals_student)  # multiply by constant iota for suitable probabilities from Q-values
        # 3.combination
        probas = trust * probas_message + (1 - trust) * probas_student
        # 4.randomness
        r = random.random()
        if r < 0.4 * (1 - current_ep / total_eps) + 0.4 * (
                current_ep / total_eps):  # decaying eps-greedy according to episode number
            action = random.choice([0, 1, 2, 3])
        else:
            action = torch.argmax(probas).item()
    return probas_message, probas_student, action


def select_action_eps_trust_uninfo(agent: DQN, state_int: int, softy: nn.Softmax, iota: float, current_ep: int,
                                   total_eps: int) -> 'tuple[torch.tensor,int]':
    '''
    mirror image of above function, just without a message (i.e. without a teacher)
    ---
    INPUT
    agent - the deep-Q-network representing the agent exploring the world
    state_int - the state the agent is currently in (in integer representation)
    softy - a softmax function
    iota - parameter to adjust student Q-values, so that softmax gives reasonable probabilities
    current_ep - current episode
    total_eps - total episode number (both parameters relevant for decaying epsilon in epsilon-greedy)
    ---
    OUTPUT
    probas_student - action probabilities in current state according to the student's own learning
    action - next action that will be executed by the student
    '''
    state = state_int_to_tuple(state_int)
    with torch.no_grad():
        # 1.student
        qvals_student = agent(state[0])
        qvals_student -= min(qvals_student) * torch.ones(n_actions).to(device)
        probas = softy(iota * qvals_student)  # multiply by constant iota for suitable probabilities from Q-values
        # 2.randomness
        r = random.random()
        if r < 0.4 * (1 - current_ep / total_eps) + 0.4 * (
                current_ep / total_eps):  # decaying epsilon greedy according to episode number
            action = random.choice([0, 1, 2, 3])
        else:
            action = torch.argmax(probas).item()
    return probas, action


def select_action_opt_trust(agent: DQN, state_int: int, probas_message_matrix: torch.tensor, softy: nn.Softmax,
                            iota: float,
                            trust: float, alpha: float,
                            sa_counts: 'dict[tuple[int, int],int]') -> 'tuple[torch.tensor, torch.tensor, int]':
    '''
    global trust parameter for teacher and optimism in face of uncertainty exploration
    ---
    INPUT
    agent - the deep-Q-network representing the agent exploring the world
    state_int - the state the agent is currently in (in integer representation)
    probas_message_matrix - matrix of action probabilities according to the message
    softy - a softmax function
    iota - parameter to adjust student Q-values so that softmax gives reasonable probabilities
    trust - tells how much the student trusts the teacher (between 0 and 1)
    alpha - constant for optimism in face of uncertainty algorithm
    sa_counts - counts how often each state-action combination has already been seen by the agent
    ---
    OUTPUT
    probas_message - action probabilities in current state according to the message
    probas_student - action probabilities in current state according to the student's own learning
    action - next action that will be executed by the student
    '''
    state = state_int_to_tuple(state_int)
    current_sa_counts = torch.tensor([sa_counts[(state_int, a)] for a in range(4)]).to(
        device)  # pick out counts of current state
    with torch.no_grad():
        # 1.message
        probas_message = probas_message_matrix[:, mt.floor(state_int / grid_dim),
                         state_int % grid_dim]  # filter out current state
        # 2.student
        qvals_student = agent(state[0])
        qvals_student -= min(qvals_student) * torch.ones(n_actions).to(device)
        probas_student = softy(
            iota * qvals_student)  # multiply by constant iota for suitable probabilities from Q-values
        # 3.combination including optimism in the face of uncertainty
        probas = trust * probas_message + (1 - trust) * (probas_student + alpha / (current_sa_counts + 1))
        # argmax
        action = torch.argmax(probas).item()
    return probas_message, probas_student, action


def select_action_opt_trust_uninfo(agent: DQN, state_int: int, softy: nn.Softmax, iota: float, alpha: float,
                                   sa_counts: 'dict[tuple[int, int],int]') -> int:
    '''
    mirror image of above function, just without a message (i.e. without a teacher)
    ---
    INPUT
    agent - the deep-Q-network representing the agent exploring the world
    state_int - the state the agent is currently in (in integer representation)
    softy - a softmax function
    iota - parameter to adjust student Q-values so that softmax gives reasonable probabilities
    alpha - constant for optimism in face of uncertainty algorithm
    sa_counts - counts how often each state-action combination has already been seen by the agent
    ---
    OUTPUT
    probas - action probabilities in state s according to the student's own learning
    action - next action that will be executed by the student
    '''
    state = state_int_to_tuple(state_int)
    current_sa_counts = torch.tensor([sa_counts[(state_int, a)] for a in range(4)]).to(
        device)  # pick out counts of current state
    with torch.no_grad():
        # 1.student
        qvals_student = agent(state[0])
        qvals_student -= min(qvals_student) * torch.ones(n_actions).to(device)
        probas = softy(iota * qvals_student)  # multiply by constant iota for suitable probabilities from Q-values
        # 2.optimism in face of uncertainty
        probas += alpha / (current_sa_counts + 1)
        # argmax
        action = torch.argmax(probas).item()
    return probas, action
