#@title 4. FUNCTIONS - Helper functions



from fixed_parameters import *
from changeable_parameters import *

import math as mt
import pickle as pkl
import random
from typing import Union, Optional

import numpy as np
import torch
from scipy.optimize import curve_fit

#Uncomment if GPU is to be used - right now use CPU, as we have very small networks and for them, CPU is actually faster
from autoencoder_def import ConvAutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#suppress scientific notation in printouts
np.set_printoptions(suppress=True)
# This is a sample Python script.

def set_seed(seed: Optional[int]=None, seed_torch: bool=True):
    {'''
        Function that controls randomness by setting a random seed
        ---
        INPUT:
        seed: the random state
        seed_torch: If `True` sets the random seed for pytorch tensors, so pytorch module
                    must be imported
        ---
        OUTPUT:
        nothing, write into file
      '''}
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')


def write_dict_into_pkl(dictionary: dict,pkl_file: str):
    {'''
        Write a dictionary into a .txt file
        ---
        INPUT:
        dictionary: the dictionary
        txt_file: location of the .txt file we are writing into
        ---
        OUTPUT:
        nothing, write into file
    '''}
    with open(pkl_file,"wb") as f:
        pkl.dump(dictionary, f)
    f.close()


def read_dict_from_pkl(pkl_file: str)->dict:
    {'''
        Read a dictionary from a .txt file
        ---
        INPUT:
        txt_file: location of the .txt file we are reading from
        ---
        OUTPUT:
        the dictionary we get from the file
    '''}
    pickle_off = open(pkl_file, 'rb')
    dictionary = pkl.load(pickle_off)
    pickle_off.close()
    return dictionary


def moving_average(a: Union[list, np.ndarray], n: int) ->np.ndarray:
    {'''
        computes the moving average of an array
        ---
        INPUT
        a - the array
        n - width of the moving window in steps
        ---
        OUTPUT
        the moving average array
    '''}
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def linfunc(x: Union[float,int],m: Union[float,int],b: Union[float,int])->Union[float,int]:
    '''
    A simple linear function
    ---
    INPUT
    x - input x-value
    m - linear function slope parameter
    b - linear function offset parameter
    ---
    OUTPUT
    y=m*x+b
    '''
    return m*x+b

def tanhfunc(x: Union[float,int],a: Union[float,int], c: Union[float,int], b: Union[float,int])->Union[float,int]:
    '''
    Hyperbolic tangent function with scaling parameters
    ---
    INPUT
    x - input x-value
    a - parameter
    b - scaling parameter (y-axis)
    ---
    OUTPUT
    y=tanh(a*x)
    '''
    return b*np.tanh(a*(x-1)**c)

def logfunc(x: Union[float,int],a: Union[float,int], c: Union[float,int], b: Union[float,int])->Union[float,int]:
    '''
    Logistic function with scaling parameters
    ---
    INPUT
    x - input x-value
    a - parameter
    b - scaling parameter (y-axis)
    ---
    OUTPUT
    y=(2*b)/(1+e^(-a*x))-b
    '''
    return 2*b/(1+np.exp(-a*(x-1)**c))-b

def hypfunc(x: Union[float,int],a: Union[float,int], c: Union[float,int], b: Union[float,int])->Union[float,int]:
    '''
    Hyperbola with scaling parameter
    ---
    INPUT
    x - input x-value
    a - parameter
    c - parameter
    b - scaling parameter (y-axis)
    ---
    OUTPUT
    y=1/(a*x)+b
    '''
    return -1/(a*(x-1)**c)+b

def arctanfunc(x: Union[float,int],a: Union[float,int], c: Union[float,int], b: Union[float,int])->Union[float,int]:
    '''
    Hyperbola with scaling parameter
    ---
    INPUT
    x - input x-value
    a - parameter
    c - parameter
    b - scaling parameter (y-axis)
    ---
    OUTPUT
    y=1/(a*x)+b
    '''
    return (2/mt.pi)*b*np.arctan(a*(x-1)**c)

def constfunc(x, a):
    return 0*x+a

def flattening_spotter(data: Union[list, np.ndarray], fit_size: int, certainty: float) ->int:
    '''
    This function finds the point when a series of data points (data) has converged/levelled/flattened
    It assumes that by the end of the data series (minus fit_size data points), convergence has definitely happened!!
    ---
    INPUT
    data - list/array of numbers. the data series
    fit_size - the number of data points taken into account for each linear fit
    certainty - if the slope plus this number of standard deviations is below zero, we stop and declare that the data is no longer level
    ---
    OUTPUT
    learned_index - the first index where the data series has become level
    '''

    learned_index: int=0
    for l in range(len(data)-fit_size,-1,-1):
        popt, pcov=curve_fit(linfunc,range(l,l+fit_size,1),data[l:l+fit_size]) #fit linear function
        if popt[0]<1e-3 and popt[0]+certainty*mt.sqrt(pcov[0,0])<1e-3: #if the slope is negative with given certainty, we stop
            learned_index=l+1 #+1 step, don't include the data that is not level anymore
            break
    return learned_index


def to_interval(x: float,left: float,right: float)->float:
    '''
    Fits data point into an interval (left outliers to left boundary, right outliers to right boundary)
    ---
    INPUT
    x - the data point
    left - left interval boundary
    right - right interval boundary
    ---
    OUTPUT
    y - the position of x in the interval
    '''

    return min(right,max(x,left))


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def get_message_list(language_code, q_matrix_dict):
    #create the message list corresponding to a certain language
    #load stored autoencoder network parameters
    autoencoder = ConvAutoEncoder(data_shape, K, nonlinear_ae_plots, nonlinear_std_plots).to(device)
    autoencoder.load_state_dict(torch.load(file_loc+"autoencoder/autoencoder network parameters/"+f"params_autoenc{language_code}.pt"))
    autoencoder.eval()
    #create a message dictionary, with indices corresponding to the task indices
    message_dict={}
    for task_index,q_matrix in q_matrix_dict.items():
        q_matrix=torch.unsqueeze(q_matrix,0) #need this because the autoencoder always expects batches of inputs!
        message=autoencoder.encode(q_matrix)[0]
        message_dict[task_index]=torch.flatten(message).detach().numpy()

    return list(message_dict.values()), autoencoder