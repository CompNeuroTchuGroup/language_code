import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fixed_parameters import *


#this is set for the printing of Q-matrices via console
torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)

#Uncomment if GPU is to be used - right now use CPU, as we have very small networks and for them, CPU is actually faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#suppress scientific notation in printouts
np.set_printoptions(suppress=True)

#@title 7. FUNCTIONS - DQN class

class DQN(nn.Module):
    '''
    Deep Q Network class
    '''

    def __init__(self, K: int, zero_init: bool = False):
        '''
        INPUT
        K: length of input message (zero for the teacher)
        zero_init: initialize all weights to zero?
        '''
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(2+K, 10, device=device) #input size is 2(gridworld coordinates)+length of message
        self.lin2=nn.Linear(10,20, device=device)
        self.lin3 = nn.Linear(20,20, device=device)
        self.lin4 = nn.Linear(20,n_actions, device=device)

        #Initialize all weights (and biases) to zero
        if zero_init:
            torch.nn.init.zeros_(self.lin1.weight)
            torch.nn.init.zeros_(self.lin1.bias)
            torch.nn.init.zeros_(self.lin2.weight)
            torch.nn.init.zeros_(self.lin2.bias)
            torch.nn.init.zeros_(self.lin3.weight)
            torch.nn.init.zeros_(self.lin3.bias)


    def forward(self, x:torch.tensor)->torch.tensor:
        '''
        forward pass of the network
        ---
        INPUT
        x - network input, i.e. combination of state and potentially message
        ---
        OUTPUT
        y - the four Q-values, i.e. torch.tensor([Q(s,0),Q(s,1),Q(s,2),Q(s,3)])
        '''
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return x


class BiasLayer(nn.Module):
    '''
    Bias Layer (add bias to individual network nodes/filter positions)
    '''
    def __init__(self, shape: tuple):
        '''
        Initialise parameters of bias layer
        ---
        INPUT
        shape: Requisite shape of bias layer
        '''
        super(BiasLayer, self).__init__()
        init_bias = torch.zeros(shape, device=device)
        self.bias = nn.Parameter(init_bias, requires_grad=True)

    def forward(self, x: torch.tensor)->torch.tensor:
        '''
        Forward pass
        ---
        INPUT
        x: Input features
        ---
        OUTPUT
        y: Output of bias layer
        '''
        y=x+self.bias
        return y
