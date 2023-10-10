import torch
import superp

############################################
# set default data type to double
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)


############################################
# set the system dimension
############################################
DIM = 2


############################################
# set the super-rectangle range
############################################
# set the initial in super-rectangle
INIT = [[0, 1], \
            [1, 2], \
        ]

INIT_SHAPE = 1 # 1 for rectangle; 2 for cycle


SUB_INIT = []
SUB_INIT_SHAPE = []


# the the unsafe in super-rectangle
UNSAFE = [[-2, 0], \
            [-1.5, 1.5], \
        ]

UNSAFE_SHAPE = 3 # 4 for parabola


SUB_UNSAFE = []
SUB_UNSAFE_SHAPE = []


# the the domain in super-rectangle
DOMAIN = [[-2, 2], \
            [-2, 2], \
        ]

DOMAIN_SHAPE = 1

############################################
# set the range constraints
############################################
def cons_init(x): # accept a two-dimensional tensor and return a tensor of bool with the same number of columns
    # return x[:, 0] == x[:, 0] # equivalent to True
    return x[:, 0] == x[:, 0]

def cons_unsafe(x):
    return x[:, 0] + x[:, 1] * x[:, 1] <= 0.0 + superp.TOL_DATA_GEN # a parabola

def cons_domain(x):
    return x[:, 0] == x[:, 0] # equivalent to True


############################################
# set the vector field
############################################
# this function accepts a tensor input and returns the vector field of the same size
def vector_field(x):
    # the vector of functions
    def f(i, x):
        if i == 1:
            return x[:, 1] + 2 * x[:, 0] * x[:, 1]# x[:, 1] stands for x2
        elif i == 2:
            return -x[:, 0] + 2 * x[:, 0] * x[:, 0] - x[:, 1] * x[:, 1] # x[:, 0] stands for x1
        else:
            print("Vector function error!")
            exit()

    vf = torch.stack([f(i + 1, x) for i in range(DIM)], dim=1)

    return vf
