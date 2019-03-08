# load the necessary packages and libs

from pyquil.quil import Program
import pyquil.api as api
from pyquil.gates import *
import numpy as np
import matplotlib.pyplot as plt
from random import *
import itertools
import networkx as nx
import pyswarms as ps

qvm = api.QVMConnection()  # to use the virtual machine
PRECISION = 8
CLIP = 1e-8


def bars_and_stripes(rows, cols):
    data = []

    for h in itertools.product([0, 1], repeat=cols):
        pic = np.repeat([h], rows, 0)
        data.append(pic.ravel().tolist())

    for h in itertools.product([0, 1], repeat=rows):
        pic = np.repeat([h], cols, 1)
        data.append(pic.ravel().tolist())

    data = np.unique(np.asarray(data), axis=0)

    return data

global n_qubits, hist_sample

n, m = 2, 2

bas = bars_and_stripes(n, m)

n_points, n_qubits = bas.shape

fig, ax = plt.subplots(1, bas.shape[0], figsize=(9, 1))
for i in range(bas.shape[0]):
    ax[i].matshow(bas[i].reshape(n, m), vmin=-1, vmax=1)
    ax[i].set_xticks([])
    ax[i].set_yticks([])

#TOPOLOGIES

'''sample distribution'''
hist_sample = [0 for _ in range(2 ** n_qubits)]
for s in bas:
    b = ''.join(str(int(e)) for e in s)
    idx = int(b, 2)
    hist_sample[idx] += 1. / float(n_points)

edges_all_connected = []
edges_star = []
edges_line = []

''' line topology'''
for i in range(n_qubits - 1):
    connections = [i, i + 1]
    edges_line.append(connections)

'''star topology'''
for i in range(1, n_qubits):
    connections = [0, i]
    edges_star.append(connections)

'''all connected topology'''
for i in range(n_qubits - 1):
    for j in range(i + 1, n_qubits):
        connections = [i, j]
        edges_all_connected.append(connections)

# Or build any other you would like to explore. For example, ring topology?


'''lets define a little function for the connections'''


def top(x):
    if x == 0:
        edges = edges_line
    if x == 1:
        edges = edges_star
    if x == 2:
        edges = edges_all_connected
    return edges


'''lets plot the different configuratios'''

options = {
    'node_color': 'orange',
    'node_size': 300,
    'width': 1}

fig = plt.figure(figsize=(9, 4))

graph_line = nx.Graph(top(0))
ax1 = fig.add_subplot(131)
ax1.set_title('line')
nx.draw_shell(graph_line, with_labels=True, **options)

graph_star = nx.Graph(top(1))
ax2 = fig.add_subplot(132)
ax2.set_title('star')
nx.draw_shell(graph_star, with_labels=True, **options)

graph_all_connected = nx.Graph(top(2))
ax3 = fig.add_subplot(133)
ax3.set_title('all connected')
nx.draw_shell(graph_all_connected, with_labels=True, **options)

#CIRCUIT PARAMETERS

global n_top, single_g, entangling_g, n_layers, dimension
global min_bounds, max_bounds, n_rotations

'''define the topology'''
n_top = 2        # options: 0 - line, 1 - star,  2 - all connected

'''define single and entangling gates'''
init_single_g =  ['RY','RZ']
single_g      =  ['RY','RZ','RY']
entangling_g  =  ['YY']       # options YY, ZZ, CPHASE -- the entangling gate can be modify in the circuit

'''define the number of layers'''
n_layers = 4

'''angles'''
first_layer   = len(init_single_g)
n_rotations   = len(single_g)
even_n_layers = int(np.floor(n_layers/2))
odd_n_layers  = n_layers - even_n_layers
dimension     = (n_rotations*(odd_n_layers-1) + first_layer)*n_qubits + len(top(n_top))*even_n_layers #len(top(n_top)) is the number of edges in the graphs

#CIRCUIT

def circuit(angles):
    C = Program()
    for i in range(n_qubits):
        C.inst(I(
            i))  # here we are considering |00000000> as the input state. Options: Changing I to H applies Hadamards to all qubits and create a full superposition

    i = 0
    for qb in range(n_qubits):
        for op in init_single_g:
            x = angles[i] * np.pi  # rotating angles are written in units of pi
            gate = (op + "(" + str(x) + ")", qb)
            C.inst(gate)
            i += 1  # now it will call the next element in angles

    for ly in range(2, n_layers + 1):
        if ly % 2 == 1:  # single gates are only in odd layers
            for qb in range(n_qubits):
                for op in single_g:
                    x = angles[i] * np.pi  # rotating angles are written in units of pi
                    gate = (op + "(" + str(x) + ")", qb)
                    C.inst(gate)

                    i += 1  # now it will call the next element in angles

        else:
            for qb_qb in top(n_top):
                if entangling_g[0] == 'YY':

                    x = angles[i] * np.pi / 2.0  # entangling angles are written in units of 2*pi
                    idx1 = qb_qb[0]
                    idx2 = qb_qb[1]

                    C.inst(CNOT(idx1, idx2))
                    C.inst(RY(2.0 * x, idx2))
                    C.inst(CNOT(idx1, idx2))

                    i += 1

                elif entangling_g[0] == 'ZZ':  # ZZ(theta,1,2) = CNOT(1,2) RZ(2 theta,2) CNOT(1,2), - pi < 2 theta < pi

                    x = angles[i] * np.pi / 2.0  # theta = x*pi/2, then -1 < x < 1
                    idx1 = qb_qb[0]
                    idx2 = qb_qb[1]

                    C.inst(CNOT(idx1, idx2))
                    C.inst(RZ(2.0 * x, idx2))
                    C.inst(CNOT(idx1, idx2))

                    i += 1

                    ## we can define more entangling gates!
                else:

                    x = angles[i] * np.pi / 2.0
                    idx1 = qb_qb[0]
                    idx2 = qb_qb[1]
                    gate = (entangling_g[0] + "(" + str(x) + ")", idx1, idx2)
                    C.inst(gate)

                    i += 1

    qvm = api.QVMConnection()
    wf = qvm.wavefunction(C)  # get the output circuit wavefunction
    probs_dist = wf.get_outcome_probs()  # get the probability distribution --- be careful it is a dict
    probs = list(range(2 ** n_qubits))

    '''here we extract the distribution values from the dict'''
    for i in range(2 ** n_qubits):
        aa = str(np.binary_repr(i, n_qubits))[::-1]
        probs[i] = probs_dist[aa]

    assert (round(sum(probs), PRECISION) == 1.)  # just to make sure

    return (probs, C)

#KLEIBER-LEIBLER FUNCTION

def KL(angles):
    probs, C = circuit(angles)

    l = 0.0
    for idx in range(2 ** n_qubits):
        l += -hist_sample[idx] * np.log(np.clip(probs[idx], CLIP, 1.)) \
             + hist_sample[idx] * np.log(np.clip(hist_sample[idx], CLIP, 1.))

    return l

def f(x):
    """Higher-level method to do forward_prop in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [KL(x[i]) for i in range(n_particles)]
    return np.array(j)

#PARTICLE SWARM OPTIMIZER

# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Call instance of PSO
dimensions = 4 ** n_layers
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, print_step=1, iters=10, verbose=3)

result, P = circuit(pos)

#Print comparison of circuit distribution versus target distribution
print(list(zip(hist_sample, result)))

#Compiler performance measurements
from pyquil.api import CompilerConnection, get_devices

devices = get_devices(as_dict=True)
agave = devices['8Q-Agave']
compiler = CompilerConnection(agave)

job_id = compiler.compile_async(P)
job = compiler.wait_for_job(job_id)

print('compiled quil', job.compiled_quil())
print('gate volume', job.gate_volume())
print('gate depth', job.gate_depth())
print('topological swaps', job.topological_swaps())
print('program fidelity', job.program_fidelity())
print('multiqubit gate depth', job.multiqubit_gate_depth())

#qBAS22 Measurements
cost, P = circuit(pos)
meas = P.measure(0, 0).measure(1, 1).measure(2, 2).measure(3, 3)

sampling = qvm.run(meas, [0, 1, 2, 3], 500)
target = [[0,0,0,0],[1,1,0,0],[0,0,1,1],[1,0,1,0],[0,1,0,1],[1,1,1,1]] #valid BAS22 targets
nbas=[]
for x in range(500):
    test_target = [(sampling[x] == el) for el in target]  # => [[True, False, True, False],[False, False, True, False], ...]
    if any(test_target):
        nbas.append(test_target.index(True)) # => [2,3,1,4,3,2,3,1]

bas22measures = len(nbas)
totalmeasures = 500
precision = bas22measures/totalmeasures

numberofdiffbas22 = len(list(set(nbas)))
numberofbas22 = len(target)
recall = numberofdiffbas22/numberofbas22

f1 = (2*precision*recall)/(precision+recall)

print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("qBAS F1: {}".format(f1))


#print("Sampling: {}".format(qvm.run(meas, [0, 1, 2, 3], 17)))




