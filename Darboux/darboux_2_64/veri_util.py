import torch
import torch.nn as nn
import ann
import numpy as np

# Model Children
# def get_children(model: torch.nn.Module):
#     # get children form model!
#     children = list(model.children())
#     flatt_children = []
#     if children == []:
#         # if model has no children; model is last child! :O
#         return model
#     else:
#        # look for children from children... to the last child!
#        for child in children:
#            try:
#                flatt_children.extend(get_children(child))
#            except TypeError:
#                flatt_children.append(get_children(child))
#        return flatt_children

# Model layer
def get_layers(model: torch.nn.Module):
    Layers = []
    children = list(model.eval())
    count_children = np.size(children)
    for i in range(count_children):
        layer = torch.nn.Sequential(*list(model.eval())[:i])
        Layers.append(layer)
    return Layers

def output_forward_activation(input,layer_w,layer_a):
    out_w = layer_w(input)
    out_a = layer_a(input)
    # Find activated neurons in each layer
    activated = torch.eq(out_w, out_a)
    # Reshape the activated vector to [len(activated), 1] for latter multiply
    activated = torch.reshape(activated, [len(activated), 1])
    return out_w, out_a, activated

# ToDo: (do it later) def activated_set(input,Layers)

def activated_weight_bias(model,activated_set):
    W_list = []
    r_list = []
    para_list = list(model.state_dict())
    i = 0
    while i < (len(para_list)):
        weight = model.state_dict()[para_list[i]]
        i += 1
        bias = model.state_dict()[para_list[i]]
        i += 1
        W_list.append(weight)
        r_list.append(bias)
    # compute the activated weight of the layer
    W_l = torch.mul(activated_set, W_list[0])
    W_overl = torch.matmul(W_list[1],W_l) # compute \overline{W}(S)
    # compute the activated bias of the layer
    r_l = torch.mul(activated_set, torch.reshape(r_list[0],[len(r_list[0]),1]))
    r_overl = torch.matmul(W_list[1], r_l) + r_list[1]  # compute \overline{r}(S)
    # compute region/boundary weight
    W_a = W_l
    r_a = r_l
    B_act = [W_a,r_a] # W_a x <= r_a
    W_i = W_list[0]-W_l

    r_i = -torch.reshape(r_list[0],[len(r_list[0]),1]) + r_l
    B_inact = [W_i,r_i] # W_a x <= r_a
    return W_overl, r_overl, B_act, B_inact

def activated_weight_bias_ml(model,activated_set,num_neuron):
    W_list = []
    r_list = []
    para_list = list(model.state_dict())
    i = 0
    while i < (len(para_list)):
        weight = model.state_dict()[para_list[i]]
        i += 1
        bias = model.state_dict()[para_list[i]]
        i += 1
        W_list.append(weight)
        r_list.append(bias)
    # compute the activated weight of the layer
    for l in range(2):
        # compute region/boundary weight


        if l == 0:
            W_l = torch.mul(activated_set[num_neuron*l:num_neuron*(l+1)], W_list[l])
            r_l = torch.mul(activated_set[num_neuron*l:num_neuron*(l+1)], torch.reshape(r_list[l], [len(r_list[l]), 1]))
            W_a = W_l
            r_a = r_l
            W_i = W_list[l] - W_l
            r_i = -torch.reshape(r_list[l], [len(r_list[l]), 1]) + r_l
        else:
            W_pre = W_list[l] @ W_l
            r_pre = W_list[l] @ r_l + r_list[l].reshape([len(r_list[l]), 1])
            W_l = activated_set[num_neuron*l:num_neuron*(l+1)]*W_pre
            r_l = activated_set[num_neuron*l:num_neuron*(l+1)]*r_pre
            W_a = torch.vstack([W_a, W_l])
            r_a = torch.vstack([r_a, r_l])
            W_i = torch.vstack([W_i, W_pre - W_l])
            r_i = torch.vstack([r_i, -torch.reshape(r_pre, [len(r_pre), 1]) + r_l])
        B_act = [W_a, r_a]  # W_a x <= r_a
        B_inact = [W_i, r_i]  # W_a x <= r_a
    # W_overl = torch.matmul(W_list[-1], torch.matmul(W_list[-2], W_l))  # compute \overline{W}(S)
    # r_overl = torch.matmul(W_list[-1], torch.matmul(W_list[-2], r_l) + r_list[-2].reshape([num_neuron,1])) + r_list[-1]  # compute \overline{r}(S)
    W_overl = torch.matmul(W_list[-1], W_l)  # compute \overline{W}(S)
    r_overl = torch.matmul(W_list[-1], r_l) + r_list[-1]  # compute \overline{r}(S)
    return W_overl, r_overl, B_act, B_inact

def find_one_zero_point_autograd(data,model_input):
    model=model_input

    # randomly pick initial points
    index = np.random.randint(0, len(data))
    xi = data[index]

    # back propagation training
    learning_rate = 1e-1
    x_restart = data[index]
    x_i = torch.tensor(xi, requires_grad=True)
    y_i = model(x_i)
    loss = abs(y_i)
    epoch = 0.0
    while loss > 0.000001:
        epoch = epoch + 1
        loss.backward()

        beta = 0.01
        gamma = 1.0
        rate = learning_rate / (1 + beta * epoch ** gamma)
        with torch.no_grad():
            x_i = x_i - rate * x_i.grad
            x_i.grad = None
        x_i.requires_grad = True
        y_i = model(x_i)
        loss = abs(y_i)
        if epoch > 10000:
            with torch.no_grad():
                x_i = torch.tensor(x_restart, requires_grad=True)
                print(x_i, 'Please restart the function and run again')
                epoch = 0

    return x_i