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
DIM = 3


############################################
# set the super-rectangle range
############################################
# set the initial in super-rectangle
INIT = [[-0.1, 0.1], \
            [-2.0, -1.8], \
                [-0.52, 0.52] # -Pi / 6 to Pi / 6
        ]

INIT_SHAPE = 1 # 1 for rectangle/cube; 2 for cycle/sphere



# the the unsafe in super-rectangle
UNSAFE = [[-0.2, 0.2], \
            [-0.2, 0.2], \
                [-2, 2]
        ]

UNSAFE_SHAPE = 3 # 1 for rectangle/cube; 2 for cycle/sphere

SUB_UNSAFE = []
SUB_UNSAFE_SHAPE = []

# SUB_UNSAFE = [ # [[-1.2, -0.8], [0.3, 0.7], [-2, 2]], \
#                 [[-0.2, 0.2], [-1.2, -0.8], [-2, 2] ]
# ]
# SUB_UNSAFE_SHAPE = [3, 3, 3] # 3 for cylinder


# the the domain in super-rectangle
DOMAIN = [[-2, 2], \
            [-2, 2], \
                [-1.57, 1.57] # -Pi / 2 to Pi / 2
        ]

DOMAIN_SHAPE = 1

############################################
# set the range constraints
############################################
def cons_init(x): # accept a two-dimensional tensor and return a tensor of bool with the same number of columns
    return x[:, 0] == x[:, 0] # equivalent to True

def cons_unsafe(x):
    unsafe = (x[:, 0] - 0) * (x[:, 0] - 0) + (x[:, 1] + 0) * (x[:, 1] + 0) <= 0.04 + superp.TOL_DATA_GEN # a cylinder
    return unsafe

def cons_domain(x):
    return x[:, 0] == x[:, 0] # equivalent to True


############################################
# set the vector field
############################################
# this function accepts a tensor input and returns the vector field of the same size
def vector_field(x):
    c1 = 1
    c2 = 3
    c3 = 0.5
    r = 0.0
    v = 1                
    def f(i, x):
        if i == 1:
            return v * torch.sin(x[:, 2])
        elif i == 2:
            return v * torch.cos(x[:, 2])
        elif i == 3:
            def dot_prod(x, y):
                return torch.sum(x * y, dim=1)

            x_vec = x[:, 0:2]
            obs = torch.tensor([0.0, 0.0])
            obs_vec = obs.repeat(len(x), 1)
            x_obs = obs_vec - x_vec
            x_obs_norm_sqr = dot_prod(x_obs, x_obs)

            goal = torch.tensor([1.0, 0.0])
            goal_vec = goal.repeat(len(x), 1)

            dir_x = torch.stack([torch.sin(x[:, 2]), torch.cos(x[:, 2])], dim=1)

            deri_phi = -c1 * dot_prod(dir_x, goal_vec) - c2 * dot_prod(dir_x, x_obs) / (c3 + x_obs_norm_sqr) 

            return deri_phi
        else:
            print("Vector function error!")
            exit()

    vf = torch.stack([f(i + 1, x) for i in range(DIM)], dim=1)
    return vf
