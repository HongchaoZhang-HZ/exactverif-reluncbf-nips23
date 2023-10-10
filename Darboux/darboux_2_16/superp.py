import torch
import numpy as np
from functools import reduce
from operator import mul

############################################
# set default data type to double; for GPU
# training use float
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)

VERBOSE = 1 # set to 1 to display epoch and batch losses in the training process

FINE_TUNE = 0 # set to 1 for fine-tuning a pre-trained model

############################################
# set the network architecture
############################################
D_H = 16 # the number of neurons of each hidden layer
N_H = 2 # then number of hidden layers


############################################
# for activation function definition
############################################
BENT_DEG = 0.0001


############################################
# set loss function definition
############################################
TOL_INIT = 0.0
TOL_SAFE = 0.0
TOL_BOUNDARY = 0.05
TOL_LIE = 0.005
TOL_NORM_LIE = 0.0
WEIGHT_LIE = 1
WEIGHT_NORM_LIE = 0

DECAY_LIE = 1
DECAY_INIT = 1
DECAY_UNSAFE = 1


############################################
# for optimization method tunning: LBFGS
############################################
LBFGS_NUM_ITER = 1
LBFGS_TOL_GRAD = 1e-05
LBFGS_TOL_CHANGE = 1e-09
LBFGS_NUM_HISTORY = 100
LBFGS_LINE_SEARCH_FUN = None


TOL_OPTIMIZER_RESET = -1
SHRINK_RATE_FACTOR = 10
FRACTION_INSTABLE_BATCH = 10000000000000000000
NUM_BATCH_ITR = 3


############################################
# set the training super parameters
############################################
EPOCHS = 100


############################################
# my own scheduling policy: 
# rate = alpha / (1 + beta * epoch^gamma)
############################################
ALPHA = 0.1 # initial learning rate
BETA = 0 # if beta equals 0 then constant rate = alpha
GAMMA = 0 # when beta is nonzero, larger gamma gives faster drop of rate


############################################
# training termination flags
############################################
LOSS_OPT_FLAG = 1e-16
TOL_MAX_GRAD = 6


############################################
# for training set generation
############################################
TOL_DATA_GEN = 1e-16

DATA_EXP_I = np.array([5, 5]) # for sampling from initial; length = prob.DIM
DATA_LEN_I = np.power(2, DATA_EXP_I) # the number of samples for each dimension of domain
BLOCK_EXP_I = np.array([3, 3]) # 0 <= BATCH_EXP <= DATA_EXP
BLOCK_LEN_I = np.power(2, BLOCK_EXP_I) # number of batches for each dimension
    # for this example, it is important to set the size of initial and unsafe not too large
    # compared with the size of each batch of domain-lie
DATA_EXP_U = np.array([7, 7]) # for sampling from initial; length = prob.DIM
DATA_LEN_U = np.power(2, DATA_EXP_U) # the number of samples for each dimension of domain
BLOCK_EXP_U = np.array([5, 5]) # 0 <= BATCH_EXP <= DATA_EXP
BLOCK_LEN_U = np.power(2, BLOCK_EXP_U) # number of batches for each dimension

DATA_EXP_D = np.array([8, 8]) # for sampling from initial; length = prob.DIM
DATA_LEN_D = np.power(2, DATA_EXP_D) # the number of samples for each dimension of domain
BLOCK_EXP_D = np.array([6, 6]) # 0 <= BATCH_EXP <= DATA_EXP
BLOCK_LEN_D = np.power(2, BLOCK_EXP_D) # number of batches for each dimension


############################################
# number of mini_batches
############################################
BATCHES_I = reduce(mul, list(BLOCK_LEN_I))
BATCHES_U = reduce(mul, list(BLOCK_LEN_U))
BATCHES_D = reduce(mul, list(BLOCK_LEN_D))

BATCHES = max(BATCHES_I, BATCHES_U, BATCHES_D)

############################################
# for plotting
############################################
PLOT_EXP_B = np.array([6, 6]) # sampling from domain for plotting the boundary of barrier using contour plot
PLOT_LEN_B = np.power(2, PLOT_EXP_B) # the number of samples for each dimension of domain, usually larger than superp.DATA_LEN_D

PLOT_EXP_V = np.array([6, 6]) # sampling from domain for plotting the vector field
PLOT_LEN_V = np.power(2, PLOT_EXP_V) # the number of samples for each dimension of domain, usually equal to superp.DATA_LEN_D

PLOT_EXP_P = np.array([6, 6]) # sampling from domain for plotting the scattering sampling points, should be equal to superp.DATA_LEN_D
PLOT_LEN_P = np.power(2, PLOT_EXP_P) # the number of samples for each dimension of domain

PLOT_VEC_SCALE = None

