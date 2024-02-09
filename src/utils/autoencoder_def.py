#@title 13. FUNCTIONS - Autoencoder definition + corresponding matrix data


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from dqn import BiasLayer
from dqn_optimizers import *

'''
Data Class definition (matrices) for the autoencoder
'''

#This is a class for our input data, which are the q-matrices (needs to be structured like this..)
class MatrixDataset(Dataset):
    #The __init__ function is run once when instantiating the Dataset object.
    def __init__(self, matrix_dict:'dict[int, torch.tensor]', label_dict: 'dict[int,tuple[int, int, int]]'):
        self.labels=label_dict #dictionary that stores matrix labels (grid world number/init state/goal state) keyed by index
        self.matrices = matrix_dict #dictionary that stores matrices keyed by index

    #The __len__ function returns the number of samples in our dataset
    def __len__(self)->int:
        return len(self.labels)

    #The __getitem__ function loads and returns a sample from the dataset at the given index idx.
    def __getitem__(self, idx: int)->'tuple[torch.tensor, tuple[int,int,int]]':
        matrix = self.matrices[idx]
        labels = self.labels[idx]
        return matrix, labels


'''
convolutional autoencoder class to encode q-matrices into messages
'''


def conv2d_output_dims(x: 'tuple[int,int,int]', layer: nn.Conv2d)->'tuple[int,int,int]':
    """
    Unnecessarily complicated but complete way to
    calculate the output depth, height
    and width size for a Conv2D layer
    ---
    INPUT
    Args:
    x: Input size (depth, height, width)
    layer: The Conv2D layer
    ---
    OUTPUT:
    Tuple of out-depth/out-height and out-width
    Output shape as given in [Ref]
    Ref:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    assert isinstance(layer, nn.Conv2d)
    p = layer.padding if isinstance(layer.padding, tuple) else (layer.padding,)
    k = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size,)
    d = layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation,)
    s = layer.stride if isinstance(layer.stride, tuple) else (layer.stride,)
    in_depth, in_height, in_width = x
    out_depth = layer.out_channels
    out_height = 1 + (in_height + 2 * p[0] - (k[0] - 1) * d[0] - 1) // s[0]
    out_width = 1 + (in_width + 2 * p[-1] - (k[-1] - 1) * d[-1] - 1) // s[-1]
    return (out_depth, out_height, out_width)


class ConvAutoEncoder(nn.Module):
    '''
    A Convolutional AutoEncoder
    '''
    def __init__(self, x_dim: 'tuple[int,int,int]', K: int, nonlinear_ae: bool, nonlinear_std: bool, n_filters: int=10, filter_size: int=2):
        '''
        Initialize parameters of ConvAutoEncoder
        ---
        INPUT
        x_dim: Input dimensions (channels, height, widths)
        K: message length/hidden dimension
        nonlinear_ae: are the activations in the autoencoder nonlinear?
        nonlinear_std: are the activations in the student nonlinear?
        n_filters: Number of filters (number of output channels)
        filter_size: Kernel size
        '''
        super().__init__()
        channels, height, widths = x_dim

        # Encoder input bias layer
        self.enc_bias = BiasLayer(x_dim)
        # First encoder conv2d layer
        #32 different filters -> grid_dim x grid_dim x n_actions to grid_dim+1 x grid_dim+1 x 32
        self.enc_conv_1 = nn.Conv2d(channels, n_filters, filter_size, padding=filter_size-1, device=device)
        #32 different filters -> grid_dim+1 x grid_dim+1 x 32 to grid_dim+2 x grid_dim+2 x 32
        # Output shape of the first encoder conv2d layer given x_dim input
        conv_1_shape = conv2d_output_dims(x_dim, self.enc_conv_1)
        # Second encoder conv2d layer
        self.enc_conv_2 = nn.Conv2d(n_filters, n_filters, filter_size, padding=filter_size-1, device=device) #and here once again 32 different filters?!
        # Output shape of the second encoder conv2d layer given conv_1_shape input
        conv_2_shape = conv2d_output_dims(conv_1_shape, self.enc_conv_2)
        # The bottleneck is a dense layer, therefore we need a flattenning layer
        self.enc_flatten = nn.Flatten()
        # Conv output shape is (depth, height, width), so the flatten size is:
        flat_after_conv = conv_2_shape[0] * conv_2_shape[1] * conv_2_shape[2]
        # Encoder Linear layer
        self.enc_lin = nn.Linear(flat_after_conv, K, device=device)

        # Decoder Linear layer
        self.dec_lin = nn.Linear(K, flat_after_conv, device=device)
        # Unflatten data to (depth, height, width) shape
        self.dec_unflatten = nn.Unflatten(dim=-1, unflattened_size=conv_2_shape)
        # First "deconvolution" layer
        self.dec_deconv_1 = nn.ConvTranspose2d(n_filters, n_filters, filter_size, padding=filter_size-1, device=device)
        # Second "deconvolution" layer
        self.dec_deconv_2 = nn.ConvTranspose2d(n_filters, channels, filter_size, padding=filter_size-1, device=device)
        # Decoder output bias layer
        self.dec_bias = BiasLayer(x_dim)

        #Student layers (only linear and bias layers)
        self.std_lin1=nn.Linear(2+K, 10, device=device) #input size is 2(gridworld coordinates)+length of message -> adjustable later
        self.std_lin1_bias=BiasLayer(10)
        self.std_lin2=nn.Linear(10,20, device=device) #first intermediate layer
        self.std_lin2_bias=BiasLayer(20)
        self.std_lin3=nn.Linear(20,20, device=device) #second intermediate layer
        self.std_lin3_bias=BiasLayer(20)
        self.std_lin4=nn.Linear(20,4, device=device) #output layer
        self.std_lin4_bias=BiasLayer(4)

        #booleans marking the nonlinearities
        self.nonlinear_ae=nonlinear_ae
        self.nonlinear_std=nonlinear_std





    def encode(self, q:torch.tensor)->torch.tensor:
        '''
        first half of autoencoder: encode q-matrix to create the message
        ---
        INPUT
        q: The Q-matrix
        ---
        OUTPUT
        m: The message, i.e. the encoded Q-matrix
        '''
        m = self.enc_bias(q)

        #nonlinear
        if self.nonlinear_ae:
            m = F.relu(self.enc_conv_1(m))
            m = F.relu(self.enc_conv_2(m))
        #linear
        else:
            m=self.enc_conv_1(m)
            m=self.enc_conv_2(m)

        m = self.enc_flatten(m)
        m = self.enc_lin(m)
        return m


    def decode_ae(self, m:torch.tensor)->torch.tensor:
        '''
        second half of autoencoder: reconstruct the original q-matrix from the message
        ---
        INPUT
        m: The message
        ---
        OUTPUT
        q: The decoded Q-matrix
        '''

        #nonlinear
        if self.nonlinear_ae:
            q = F.relu(self.dec_lin(m))
            q = self.dec_unflatten(q)
            q = F.relu(self.dec_deconv_1(q))
        #linear
        else:
            q=self.dec_lin(m)
            q = self.dec_unflatten(q)
            q=self.dec_deconv_1(q)

        q = self.dec_deconv_2(q)
        q = self.dec_bias(q)
        return q




    def student(self, m_all: torch.tensor, state_tensors: torch.tensor)->torch.tensor:
        '''
        student decoding - the student network creates its q-matrix from the message(s)
        Could probably improve this function by replacing the for-loop..
        ---
        INPUT
        m_all: A number of messages combined in one tensor
        ---
        OUTPUT
        Q_all: torch.tensor. The student Q-matrices corresponding to the input messages
        '''
        Q_all=torch.zeros(size=[len(m_all),n_actions,grid_dim,grid_dim]).to(device)
        m_all=m_all.repeat(grid_dim,1)
        for j in range(grid_dim):
            ms=torch.cat((m_all,state_tensors[j]), dim=1)

            #nonlinear
            if self.nonlinear_std:
                x=F.relu(self.std_lin1(ms))
                x=self.std_lin1_bias(x)
                x=F.relu(self.std_lin2(x))
                x=self.std_lin2_bias(x)
                x=F.relu(self.std_lin3(x))

            #linear
            else:
                x=self.std_lin1(ms)
                x=self.std_lin1_bias(x)
                x=self.std_lin2(x)
                x=self.std_lin2_bias(x)
                x=self.std_lin3(x)

            x=self.std_lin3_bias(x)
            x=self.std_lin4(x)
            x=self.std_lin4_bias(x)
            for i in range(grid_dim):
                Q_all[:,:,i,j]=x[i*int(len(m_all)/grid_dim):(i+1)*int(len(m_all)/grid_dim)]
        return Q_all


    def forward(self, q: torch.tensor)->'tuple[torch.tensor, torch.tensor]':
        '''
        do a forward pass of the autoencoder, i.e. encoding and decoding, but without the student
        ---
        INPUT
        q: A number of messages combined in one tensor
        OUTPUT
        m: The student Q-matrices corresponding to the input messages
        q_rec: The reconstructed Q-matrix (by the second half of the autoencoder)
        '''
        m=self.encode(q)
        q_rec=self.decode_ae(m)
        return m, q_rec

    #do a full pass of the autoencoder, i.e. encoding/decoding (WITH student)
    def forward_student(self, q: torch.tensor, state_tensors: torch.tensor)->'tuple[torch.tensor, torch.tensor, torch.tensor]':
        '''
        do a forward pass of the autoencoder, i.e. encoding and decoding, and also the student
        ---
        INPUT
        q: A number of messages combined in one tensor
        OUTPUT
        m: The student Q-matrices corresponding to the input messages
        q_rec: The reconstructed Q-matrix (by the second half of the autoencoder)
        q_std: The Q-matrix created by the student
        '''
        m=self.encode(q)
        q_rec=self.decode_ae(m)
        '''
        lp = LineProfiler()
        lp_wrapper = lp(self.student)
        lp_wrapper(m)
        lp.print_stats()
        '''
        q_std=self.student(m, state_tensors)
        return m, q_rec, q_std


