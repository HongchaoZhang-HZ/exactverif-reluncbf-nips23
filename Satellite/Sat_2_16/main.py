import numpy as np
import torch

from veri_util import *
import time
import torch.nn.functional as F
from visualization import *
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import minimize, optimize, linprog
from scipy.optimize import LinearConstraint, NonlinearConstraint
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")
num_neuron = 16
def gridify(state_space, shape, cell_length):
    nx = torch.linspace(state_space[0][0] + cell_length / 2, state_space[0][1] - cell_length / 2, shape[0])
    ny = torch.linspace(state_space[1][0] + cell_length / 2, state_space[1][1] - cell_length / 2, shape[1])
    nz = torch.linspace(state_space[2][0] + cell_length / 2, state_space[2][1] - cell_length / 2, shape[2])
    ndx = torch.linspace(state_space[3][0] + cell_length / 2, state_space[3][1] - cell_length / 2, shape[3])
    ndy = torch.linspace(state_space[4][0] + cell_length / 2, state_space[4][1] - cell_length / 2, shape[4])
    ndz = torch.linspace(state_space[5][0] + cell_length / 2, state_space[5][1] - cell_length / 2, shape[5])

    vx, vy, vz, vdx, vdy, vdz = torch.meshgrid(nx, ny, nz, ndx, ndy, ndz)
    data = torch.stack((vx,vy,vz,vdx,vdy,vdz),dim=-1).reshape(shape[0] * shape[1] * shape[2]
                                                              * shape[3] * shape[4] * shape[5], 6)
    return data

def sect_search(nnmodel, data, cell_length):
    model = BoundedModule(nnmodel, data)
    ptb = PerturbationLpNorm(norm=np.inf, eps=cell_length / 2)
    my_input = BoundedTensor(data, ptb)
    lb, ub = model.compute_bounds(x=(my_input,), method="backward")
    return lb, ub


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

def find_intersects(actuatl_set_list,possible_intersections):
    intersections = []
    act_intersections_list = []
    actuatl_set = np.asarray(actuatl_set_list)
    for sets in possible_intersections:
        cnt = 0
        set_asarray = np.asarray(sets.copy())
        set_temp = []
        act_set_temp = []
        for item in sets:
            if item in actuatl_set:
                cnt += 1
                act_str = np.array2string(item.reshape([len(item)]))
                set_temp.append(act_str)
                act_set_temp.append(item)
        additional_set = set(set_temp)
        if cnt > 1 and additional_set not in intersections:
            intersections.append(set(set_temp))
            act_intersections_list.append(act_set_temp.copy())
    return intersections, act_intersections_list

def list_activated(nnmodel,state_space,shape,cell_length):
    # Grid the space
    data = gridify(state_space, shape, cell_length)

    # Call auto_LiRPA and return cells across.
    lb, ub = sect_search(nnmodel, data, cell_length)

    # Print number of cells
    sect_ind = (torch.sign(lb) * torch.sign(ub) - 1) / 2
    sect_idx = torch.nonzero(sect_ind.reshape([shape[0] * shape[1] * shape[2]* shape[3]* shape[4]* shape[5]]))
    sections = data[sect_idx]
    num_sec = torch.sum(sect_ind)
    print(f'There are', abs(num_sec), 'to be checked')

    # 3. For grids that use IBP (use, find '?')
    num_act_layer = 2
    sections = []
    activated_sets = []
    for cell in data[sect_idx]:
        bounds = cell.reshape([cell.shape[1], cell.shape[0]]) + cell_length / 2 * np.array(
            [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]])
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

        if num >= 1e10:
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


def dbdxf(x,W_overl,n=0.5):
    Fx = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],
                  [3*n**2,0,0,0,2*n,0],[0,0,0,-2*n,0,0],[0,0,0,-n**2,0,0]])
    dbdxf = W_overl @ Fx @ x
    return dbdxf

def verification(nnmodel, actual_set):
    G = np.array([[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    problematic_set = []
    for act_array in actual_set:
        W_overl, r_overl, B_act, B_inact = activated_weight_bias_ml(nnmodel, torch.Tensor(act_array),num_neuron)
        # compute boundary condition of polyhedron
        W_overl = W_overl.numpy()
        r_overl = r_overl.numpy()
        if np.linalg.norm(W_overl[0] @ G) < 1e-6:
            W_Bound = np.array(-B_act[0] + B_inact[0])
            r_Bound = np.array(-B_act[1] - B_inact[1])

            x0 = np.array([[0], [0], [0]])

            # con = lambda x: W_overl[0]*(x[1] + 2 * x[0] * x[1]) + W_overl[1]*(-x[0] + 2 * x[0] ** 2 - x[1] ** 2)
            # nlc = NonlinearConstraint(con, -np.inf*np.ones(len(W_Bound)), -r_Bound)
            lcon = LinearConstraint(W_Bound, -np.inf * np.ones(len(W_Bound)), -r_Bound.reshape(len(r_Bound)))
            eqcon = LinearConstraint(W_overl[0], -r_overl[0], -r_overl[0])
            res = minimize(dbdxf, x0, args=-W_overl[0], constraints=[lcon, eqcon], tol=1e-6)

            if res.fun < 0:
                problematic_set.append(act_array.copy())

    # print(len(problematic_set)/len(actual_set))
    if len(problematic_set) == 0:
        return True
    else:
        return False

def h_x(x):
    hx = (x[0]**2+x[1]**2) - 0.5
    return hx

def correctness_verification(nnmodel, actual_set):
    G = np.array([[0], [0], [1]])
    problematic_set = []
    for act_array in actual_set:
        W_overl, r_overl, B_act, B_inact = activated_weight_bias(nnmodel, torch.Tensor(act_array))
        # compute boundary condition of polyhedron
        W_overl = W_overl.numpy()
        r_overl = r_overl.numpy()
        if np.linalg.norm(W_overl[0] @ G) < 1e-6:
            W_Bound = np.array(-B_act[0] + B_inact[0])
            r_Bound = np.array(-B_act[1] - B_inact[1])

            x0 = np.array([[0], [0], [0]])

            # con = lambda x: W_overl[0]*(x[1] + 2 * x[0] * x[1]) + W_overl[1]*(-x[0] + 2 * x[0] ** 2 - x[1] ** 2)
            # nlc = NonlinearConstraint(con, -np.inf*np.ones(len(W_Bound)), -r_Bound)
            lcon = LinearConstraint(W_Bound, -np.inf*np.ones(len(W_Bound)), -r_Bound.reshape(len(r_Bound)))
            eqcon = LinearConstraint(W_overl[0], -r_overl[0], -r_overl[0])
            res = minimize(h_x, x0, args=-W_overl[0], constraints=[lcon, eqcon], tol=1e-6)

            if res.fun < 0:
                problematic_set.append(act_array.copy())
    # print(len(problematic_set)/len(actual_set))
    if len(problematic_set) == 0:
        return True
    else:
        return False

def inter_verification(nnmodel, actual_set):
    G = np.array([[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    problematic_set = []

    for idx in range(len(actual_set)):
        act_array = actual_set[idx]
        W_overl, r_overl, B_act, B_inact = activated_weight_bias_ml(nnmodel, torch.Tensor(act_array),num_neuron)
        # compute boundary condition of polyhedron
        W_overl = W_overl.numpy()
        r_overl = r_overl.numpy()
        W_Bound = np.array(-B_act[0] + B_inact[0])
        r_Bound = np.array(-B_act[1] - B_inact[1])
        if idx == 0:
            W_overl_inter = W_overl
            r_overl_inter = r_overl
            W_Bound_inter = W_Bound
            r_Bound_inter = r_Bound
        else:
            W_overl_inter = np.vstack([W_overl_inter, W_overl])
            r_overl_inter = np.vstack([r_overl_inter, r_overl])
            W_Bound_inter = np.vstack([W_Bound_inter, W_Bound])
            r_Bound_inter = np.vstack([r_Bound_inter, r_Bound])

        x0 = np.array([[0], [0], [0], [0], [0], [0]])

        # con = lambda x: W_overl[0]*(x[1] + 2 * x[0] * x[1]) + W_overl[1]*(-x[0] + 2 * x[0] ** 2 - x[1] ** 2)
        # nlc = NonlinearConstraint(con, -np.inf*np.ones(len(W_Bound)), -r_Bound)
        lcon = LinearConstraint(W_Bound, -np.inf * np.ones(len(W_Bound)), -r_Bound.reshape(len(r_Bound)))
        eqcon = LinearConstraint(W_overl[0], -r_overl[0], -r_overl[0])
        res = minimize(dbdxf, x0, args=-W_overl[0], constraints=[lcon, eqcon], tol=1e-6)

        if res.fun < 0:
            problematic_set.append(act_array.copy())
    # print(len(problematic_set)/len(actual_set))
    if len(problematic_set) == 0:
        return True
    else:
        size = [len(W_overl),len(r_overl[0]),len(W_Bound),len(r_Bound)]
        res_value = suf_nec_inter_verification(W_overl_inter, r_overl_inter, W_Bound_inter, r_Bound_inter, size)
        if res_value >= -1e-6:
            return True
        else:
            return False

def y_dbdxf(xy,W_overl_inter):
    # sum_range = int(len(W_overl_inter) / 1)
    y = xy[-(1+3):]
    x = xy[:-(1+3)]
    W_overl = W_overl_inter
    xarray = dbdxf(x, W_overl)
    # obj_sum = np.vstack([xarray, np.zeros([4,1])])

    return y[0] * xarray

def suf_nec_inter_verification(W_overl_inter, r_overl_inter, W_Bound_inter, r_Bound_inter, size):
    results = []
    size_W_overl = size[0]
    size_r_overl = size[1]
    size_W_Bound = size[2]
    size_r_Bound = size[3]
    sum_range = int(len(W_overl_inter) / size_W_overl)
    for i in range(sum_range):
        W_overl = W_overl_inter[i]
        r_overl = r_overl_inter[i]
        W_Bound = W_Bound_inter[i]
        r_Bound = r_Bound_inter[i]

        x0 = np.array([[0], [0], [0], [0], [0], [0]])
        # lcon = LinearConstraint(W_Bound, -np.inf * np.ones(len(W_Bound)), -r_Bound.reshape(len(r_Bound)))
        # eqcon = LinearConstraint(W_overl[0], -r_overl[0], -r_overl[0])
        # res = minimize(dbdxf, x0, args=-W_overl[0], constraints=[lcon, eqcon], tol=1e-6)
        # results.append(res.fun)

        # results_array = np.asarray(results)
        initial_x0 = x0
        # for num in range(sum_range-1):
        #     initial_x0 = np.vstack([initial_x0,x0])
        y0 = np.zeros(1+3)
        initial_state = np.vstack([initial_x0,y0.reshape([y0.shape[0],1])])
        xy0 = np.vstack([x0,y0.reshape([y0.shape[0],1])])
        lcon = LinearConstraint(np.hstack([np.vstack([W_Bound, np.zeros([1 + 3, 6])]),
                                           np.vstack([np.zeros([1, 1 + 3]), np.eye(1 + 3)])]),
                                np.hstack([-np.inf * np.ones(len(r_Bound)), np.zeros(1 + 3)]),
                                np.hstack([-r_Bound.reshape(len(r_Bound)), np.inf * np.ones(1 + 3)]))
        eqcon = LinearConstraint(np.hstack([np.vstack([W_overl, np.zeros([1 + 3, 6])]),
                                           np.vstack([np.ones([1, 1+3]), np.eye(1+3)])]),
                                np.hstack([-r_overl, np.zeros(1+3)]),
                                np.hstack([-r_overl, np.zeros(1+3)]))
        # eqcon = LinearConstraint(W_overl[0], -r_overl[0], -r_overl[0])

        res = minimize(y_dbdxf, xy0.reshape(len(xy0)), args=-W_overl, constraints=[lcon, eqcon], tol=1e-6)
        results.append(res.fun)


    return max(results)

def main(cbounds, cscale):
    # Load Model
    print('Load Model')
    nnmodel = ann.gen_nn()
    # nnmodel.load_state_dict(torch.load('darboux_1_20_lr01.pt'), strict=True)
    # nnmodel.load_state_dict(torch.load('satellitev1_2_32.pt'), strict=False)
    tempt_model = torch.load('satellitev1_2_{}.pt'.format(num_neuron))
    new_state_dict = OrderedDict()
    for key, value in tempt_model.items():
        new_key = key.replace('V_nn.', '')
        new_key = new_key.replace('input_linear', '0')
        new_key = new_key.replace('layer_0_linear', '2')
        new_key = new_key.replace('layer_1_linear', '4')
        new_key = new_key.replace('output_linear', '5')
        new_state_dict[new_key] = value
    nnmodel.load_state_dict(new_state_dict, strict=True)
    # this line is for brutal force search
    # lst = list(itertools.product([0, 1], repeat=20))
    scale = cscale
    t_start = time.time()
    # crange = [-1.5, -1.3]
    state_space = cbounds
    # state_space = [crange,crange,crange,crange,crange,crange]
    # state_space = [[-1.5, 1.5],[-1.5, 1.5],[-1.5, 1.5],[-1.5, 1.5],[-1.5, 1.5],[-1.5, 1.5]]
    shape = [scale,scale,scale,scale,scale,scale]
    cell_length = (state_space[0][1] - state_space[0][0]) / shape[0]

    activated_sets, sections = list_activated(nnmodel, state_space, shape, cell_length)
    layers = get_layers(nnmodel)
    l1z = layers[1]
    l1a = layers[2]
    l2z = layers[3]
    l2a = layers[4]

    act_sets_list = []
    U_actset_list = []
    possible_intersections = []
    for item in range(len(sections)):
        [out_w, out_a, activated] = output_forward_activation(sections[item].reshape([6]), l1z, l1a)
        act_array = activated.int().numpy()
        [out_w, out_a, activated] = output_forward_activation(sections[item].reshape([6]), l2z, l2a)
        act_array = np.vstack([act_array, activated])
        # act_str = np.array2string(act_array.reshape([len(act_array)]))
        # U_actset_list.append(act_str)
        act_sets_list.append(act_array.copy())
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
    res_num = 0
    unique_act_sets = np.unique(act_sets_array, axis=0)
    actual_set_list = []
    for act_array in unique_act_sets:
        # [out_w, out_a, activated] = output_forward_activation(sections[item].reshape([3]), l1z, l1a)

        W_overl, r_overl, B_act, B_inact = activated_weight_bias_ml(nnmodel, torch.Tensor(act_array), num_neuron)
        # compute boundary condition of polyhedron
        W_Bound = torch.Tensor(-B_act[0] + B_inact[0])
        r_Bound = torch.Tensor(-B_act[1] - B_inact[1])
        # print(W_overl.shape)
        # print(r_overl.shape)
        res_zero = linprog(c=[1, 1, 1, 1, 1, 1],
                           A_ub=W_Bound, b_ub=-r_Bound,
                           A_eq=W_overl, b_eq=-r_overl,
                           bounds=state_space,
                           method='highs')
        # print(item/len(sections))
        res_num += int(res_zero.success)
        if res_zero.success:
            actual_set_list.append(act_array.copy())

    intersections, act_intersections_list = find_intersects(actual_set_list, possible_intersections)
    print(len(intersections))
    U_actset = set(U_actset_list)
    print('U_actset', len(set(U_actset)))
    print('Unique set', len(np.unique(act_sets_array, axis=0)))
    print('Activation Patterns', res_num)
    t_end = time.time()
    time_spent = t_end - t_start
    print('activated set compute complete with size', len(activated_sets))
    print('in', time_spent, 'seconds')

    # Verification
    veri_res_set = verification(nnmodel, actual_set_list)
    veri_res_intersect = []
    G = np.array([[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    for act_intersections in act_intersections_list:
        if len(act_intersections_list) <= 1:
            break
        dbdxg_list = []
        for idx in range(len(act_intersections)):
            act_array = act_intersections[idx]
            W_overl, r_overl, B_act, B_inact = activated_weight_bias_ml(nnmodel, torch.Tensor(act_array),num_neuron)
            dbdxg = W_overl[0] @ G
            # dbdxg_list.append(dbdxg.numpy())
            dbdxg_list.append(np.sign(dbdxg.numpy()))
        det_dbdxg = []
        for cmp_item in range(len(dbdxg_list)):
            det_item = all(dbdxg_list[cmp_item] == dbdxg_list[0])
            det_dbdxg.append(det_item)
        # if all(np.sign(np.asarray(dbdxg_list))):
        #     veri_res_intersect_item = True
        if all(det_dbdxg):
            veri_res_intersect_item = True
        else:
            veri_res_intersect_item = inter_verification(nnmodel, act_intersections)
        veri_res_intersect.append(veri_res_intersect_item)
    if veri_res_set and all(veri_res_intersect):
        print('Successfully Verified')
    else:
        print('Failed Verification')
        print(veri_res_intersect)
    v_end = time.time()
    veri_time = v_end - t_start
    print('in', veri_time)


if __name__ == "__main__":
    scale = 1
    crange = [-1.5, 1.5]
    state_space = [crange, crange, crange, crange, crange, crange]
    main(state_space, 10)
    # state_space = [[-1.5, 1.5],[-1.5, 1.5],[-1.5, 1.5],[-1.5, 1.5],[-1.5, 1.5],[-1.5, 1.5]]
    # shape = [scale, scale, scale, scale, scale, scale]
    # cell_length = (state_space[0][1] - state_space[0][0]) / shape[0]
    # data = gridify(state_space, shape, cell_length)
    # num = 0
    # total = scale**6
    # for cell in data:
    #     bounds = cell.reshape([6, 1]) + cell_length / 2 * np.array(
    #         [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]])
    #     main(bounds, 5)
    #     num+=1
    #     print(num/total)