import torch
import torch.nn as nn
import superp

############################################
# set default data type to double
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)


############################################
# self-defined activation function
############################################
class my_act(nn.Module):
    def __init__(self):
        super(my_act, self).__init__()
        
    def forward(self, x):
        x = 0.5 * x + torch.sqrt(0.25 * x * x + superp.BENT_DEG) #bent relu, approximate relu as close as possible for post-verfication
        #x = x + 0.5 * (torch.sqrt(x * x + 1) - 1) #bent identity
        #x = 0.51 * x + torch.sqrt(0.2401 * x * x + superp.BENT_DEG) #bent leakly relu
        #x = 0.5 * x + torch.sqrt(0.16 * x * x + superp.BENT_DEG) #bent leakly relu
        return x