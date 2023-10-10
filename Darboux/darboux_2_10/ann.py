import torch
import torch.nn as nn
import numpy as np
import superp
import prob
import acti

############################################
# set default data type to double
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)


############################################
# generate nn architecture
############################################

def gen_nn():
    # input layer and output layer

     #act_fun = nn.Sigmoid()
    # act_fun = acti.my_act()
    act_fun = nn.ReLU()
    layer_input = [nn.Linear(prob.DIM, superp.D_H, bias=True)]
    layer_output = [act_fun, nn.Linear(superp.D_H, 1, bias=True)]

    # hidden layer
    module_hidden = [[act_fun, nn.Linear(superp.D_H, superp.D_H, bias=True)] for _ in range(superp.N_H - 1)]
    layer_hidden = list(np.array(module_hidden).flatten())

    # nn model
    layers = layer_input + layer_hidden + layer_output
    model = nn.Sequential(*layers)

    return model
