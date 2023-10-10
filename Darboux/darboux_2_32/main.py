import numpy as np
import torch

from veri_util import *
import time
import torch.nn.functional as F
from visualization import *
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import minimize, optimize
from scipy.optimize import LinearConstraint, NonlinearConstraint
import warnings
warnings.filterwarnings("ignore")
num_neuron = 32
def dbdxf(x,W_overl):
    # x1_dot = x2 + 2*x1*x2
    # x2_dot = -x1 + 2*x1**2 - x2**2
    x1 = x[0]
    x2 = x[1]
    x1_dot = x2 + 2 * x1 * x2
    x2_dot = -x1 + 2 * x1 ** 2 - x2 ** 2
    dbdxf = W_overl[0]*x1_dot + W_overl[1]*x2_dot
    return dbdxf

def verification(nnmodel, actual_set):
    problematic_set = []
    for act_array in actual_set:
        W_overl, r_overl, B_act, B_inact = activated_weight_bias_ml(nnmodel, torch.Tensor(act_array), num_neuron)
        # compute boundary condition of polyhedron
        W_overl = W_overl.numpy()
        r_overl = r_overl.numpy()
        W_Bound = np.array(-B_act[0] + B_inact[0])
        r_Bound = np.array(-B_act[1] - B_inact[1])

        x0 = np.array([[0], [0]])

        # con = lambda x: W_overl[0]*(x[1] + 2 * x[0] * x[1]) + W_overl[1]*(-x[0] + 2 * x[0] ** 2 - x[1] ** 2)
        # nlc = NonlinearConstraint(con, -np.inf*np.ones(len(W_Bound)), -r_Bound)
        lcon = LinearConstraint(W_Bound,-np.inf*np.ones(len(W_Bound)),-r_Bound.reshape(len(r_Bound)))
        eqcon = LinearConstraint(W_overl[0], -r_overl[0], -r_overl[0])
        res = minimize(dbdxf, x0, args=-W_overl[0], constraints=[lcon,eqcon],tol=1e-6)
        print(res.fun)
        if res.fun < 0:
            problematic_set.append(act_array.copy())

    # print(len(problematic_set))
    if len(problematic_set) == 0:
        return True
    else:
        return False

def gridify(state_space, shape, cell_length):
    nx = torch.linspace(state_space[0][0] + cell_length / 2, state_space[0][1] - cell_length / 2, shape[0])
    ny = torch.linspace(state_space[1][0] + cell_length / 2, state_space[1][1] - cell_length / 2, shape[1])
    vx, vy = torch.meshgrid(nx, ny)
    data = np.dstack([vx.reshape([shape[0], shape[1], 1]), vy.reshape([shape[0], shape[1], 1])])
    data = torch.Tensor(data.reshape(shape[0] * shape[1], 2))
    return data

def sect_search(nnmodel, data, cell_length):
    model = BoundedModule(nnmodel, data)
    ptb = PerturbationLpNorm(norm=np.inf, eps=cell_length / 2)
    my_input = BoundedTensor(data, ptb)
    lb, ub = model.compute_bounds(x=(my_input,), method="backward")
    return lb, ub

# def weighted_bound(nnmodel, prev_upper, prev_lower):
#     prev_mu = (prev_upper + prev_lower) / 2
#     prev_r = (prev_upper - prev_lower) / 2
#     mu = F.linear(prev_mu, nnmodel.state_dict()['0.weight'], nnmodel.state_dict()['0.bias'])
#     r = F.linear(prev_r, torch.abs(nnmodel.state_dict()['0.weight']))
#     upper = mu + r
#     lower = mu - r
#     return upper, lower

def weighted_bound(weight, bias, prev_upper, prev_lower):
    prev_mu = (prev_upper + prev_lower) / 2
    prev_r = (prev_upper - prev_lower) / 2
    mu = F.linear(prev_mu, weight, bias)
    r = F.linear(prev_r, torch.abs(weight))
    upper = mu + r
    lower = mu - r
    return upper, lower

def list_flip(upper, lower):
    ind = (torch.sign(lower) * torch.sign(upper) - 1) / 2
    idx = torch.nonzero(ind)

    return ind, idx

def list_activated(nnmodel,state_space,shape,cell_length):
    # Grid the space
    data = gridify(state_space, shape, cell_length)

    # Call auto_LiRPA and return cells across.
    lb, ub = sect_search(nnmodel, data, cell_length)

    # Print number of cells
    sect_ind = (torch.sign(lb) * torch.sign(ub) - 1) / 2
    sect_idx = torch.nonzero(sect_ind.reshape([shape[0] * shape[1]]))
    sections = data[sect_idx]
    num_sec = torch.sum(sect_ind)
    print(f'There are', abs(num_sec), 'to be checked')

    # 3. For grids that use IBP (use, find '?')
    num_act_layer = 2
    sections = []
    activated_sets = []
    for cell in data[sect_idx]:
        bounds = cell.reshape([cell.shape[1], cell.shape[0]]) + cell_length / 2 * np.array([[-1, 1], [-1, 1]])
        prev_lower = bounds[:, 0]
        prev_upper = bounds[:, 1]
        # upper, lower = weighted_bound(nnmodel, prev_upper, prev_lower)
        # act_ind, act_set = list_flip(upper, lower)
        weight = nnmodel.state_dict()['0.weight']
        bias = nnmodel.state_dict()['0.bias']

        for num_layer in range(num_act_layer):
            upper, lower = weighted_bound(weight, bias, prev_upper, prev_lower)
            act_ind, act_set = list_flip(upper, lower)
            prev_lower = lower
            prev_upper = upper
            weight = nnmodel.state_dict()['{}.weight'.format(2 * (num_layer + 1))]
            bias = nnmodel.state_dict()['{}.bias'.format(2 * (num_layer + 1))]

            if num_layer == 0:
                act_set_layers = act_set
                num = torch.sum(torch.abs(act_ind))
            else:
                act_set_layers = np.vstack([act_set_layers, (act_set + num_neuron * num_layer)])
                if num_layer != num_act_layer:
                    num += torch.sum(torch.abs(act_ind))

        if num >= 2:
            small_cell_length = (bounds.numpy()[0][1] - bounds.numpy()[0][0]) / shape[0]
            small_act_set_layers, small_sections = list_activated(nnmodel, bounds.numpy(), shape, small_cell_length)
        else:
            small_act_set_layers = act_set_layers
            small_sections = cell
        if len(small_act_set_layers) > 0:
            for element in small_act_set_layers:
                activated_sets.append(element)
            for element in small_sections:
                sections.append(element)
        else:
            if small_act_set_layers != []:
                activated_sets.append(small_act_set_layers)
                sections.append(small_sections)

    return activated_sets, sections

def find_intersects(actuatl_set_list,possible_intersections):
    intersections = []
    act_intersections_list = []
    actuatl_set = np.asarray(actuatl_set_list)
    for sets in possible_intersections:
        cnt = 0
        set_asarray = np.asarray(sets.copy())
        set_temp = []
        act_set_temp = []
        for item in set_asarray:
            if item in actuatl_set:
                cnt += 1
                act_str = np.array2string(item.reshape([len(item)]))
                set_temp.append(act_str)
                act_set_temp.append(item)
        additional_set = set(set_temp)
        if cnt > 1 and additional_set not in intersections:
            intersections.append(set(set_temp))
            act_intersections_list.append(act_set_temp)
    return intersections, act_intersections_list

def main():
    # Load Model
    print('Load Model')
    nnmodel = ann.gen_nn()
    # nnmodel.load_state_dict(torch.load('pre-trained.pt'), strict=True)
    # nnmodel.load_state_dict(torch.load('darboux_1_20_lr01.pt'), strict=True)
    nnmodel.load_state_dict(torch.load('darboux_2_{}.pt'.format(num_neuron)), strict=True)
    # this line is for brutal force search
    # lst = list(itertools.product([0, 1], repeat=20))

    t_start = time.time()
    state_space = [[-2,2],[-2,2]]
    shape = [100,100]
    cell_length = (state_space[0][1] - state_space[0][0]) / shape[0]

    activated_sets, sections = list_activated(nnmodel, state_space, shape, cell_length)
    layers = get_layers(nnmodel)
    l1z = layers[1]
    l1a = layers[2]
    l2z = layers[3]
    l2a = layers[4]

    U_actset_list = []
    pts = []
    act_sets_list = []
    U_actset_list = []
    possible_intersections = []
    for item in range(len(sections)):
        [out_w, out_a, activated] = output_forward_activation(sections[item].reshape([2]), l1z, l1a)
        act_array = activated.int().numpy()
        [out_w, out_a, activated] = output_forward_activation(sections[item].reshape([2]), l2z, l2a)
        act_array = np.vstack([act_array, activated])
        act_str = np.array2string(act_array.reshape([len(act_array)]))
        U_actset_list.append(act_str)
        act_sets_list.append(act_array)
        if len(activated_sets[item]) > 0:
            # print(activated_sets[item])
            lst = list(itertools.product([0, 1], repeat=len(activated_sets[item])))
            intersect_items = []
            for possible in lst:
                act_array[activated_sets[item]] = np.array(possible).reshape([len(possible), 1, 1])
                act_str = np.array2string(act_array.reshape([len(act_array)]))
                U_actset_list.append(act_str)
                act_sets_list.append(act_array.copy())
                intersect_items.append(act_array.copy())
            possible_intersections.append(intersect_items)

    act_sets_array = np.asarray(act_sets_list)
    res_list = []
    unique_set = np.unique(act_sets_array, axis=0)
    actual_set_list = []
    for act_array in unique_set:
        act_str = np.array2string(act_array.reshape([len(act_array)]))
        W_overl, r_overl, B_act, B_inact = activated_weight_bias_ml(nnmodel, torch.Tensor(act_array), num_neuron)
        # compute boundary condition of polyhedron
        W_Bound = torch.Tensor(-B_act[0] + B_inact[0])
        r_Bound = torch.Tensor(-B_act[1] - B_inact[1])
        res_zero = linprog(c=[1, 1],
                           A_ub=W_Bound, b_ub=-r_Bound,
                           A_eq=W_overl, b_eq=-r_overl, bounds=([-2, 2], [-2, 2]),
                           method='highs')
        print(res_zero.success)
        res_list.append(res_zero.success)
        if res_zero.success:
            actual_set_list.append(act_array.copy())

    intersections, act_intersections_list = find_intersects(actual_set_list,possible_intersections)
    print(len(intersections))
    U_actset = set(U_actset_list)
    print('U_actset', len(set(U_actset)))
    print('Unique set', len(np.unique(act_sets_array, axis=0)))
    print('Activation Patterns', sum(res_list))
    t_end = time.time()
    time_spent = t_end - t_start
    print('activated set compute complete with size', len(activated_sets))
    print('in', time_spent, 'seconds')

    # Verification
    veri_res_set = verification(nnmodel, actual_set_list)
    veri_res_intersect = []
    for act_intersections in act_intersections_list:
        veri_res_intersect_item = verification(nnmodel, act_intersections)
        veri_res_intersect.append(veri_res_intersect_item)
    if veri_res_set and all(veri_res_intersect):
        print('Successfully Verified')
    else:
        print('Failed Verification')
    v_end = time.time()
    veri_time = v_end - t_start
    print('in', veri_time)

    for i in pts:
        plt.scatter(i[0][0], i[0][1])
    plt.show()
    # print(activated_sets)

if __name__ == "__main__":
    main()