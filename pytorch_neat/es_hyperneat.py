import neat 
import copy
import numpy as np
import itertools
from math import factorial
from pytorch_neat.recurrent_net import RecurrentNet
from pytorch_neat.cppn import get_nd_coord_inputs
import torch
#encodes a substrate of input and output coords with a cppn, adding 
#hidden coords along the 

class ESNetwork:

    def __init__(self, substrate, cppn, params):
        self.substrate = substrate
        self.cppn = cppn
        self.initial_depth = params["initial_depth"]
        self.max_depth = params["max_depth"]
        self.variance_threshold = params["variance_threshold"]
        self.band_threshold = params["band_threshold"]
        self.iteration_level = params["iteration_level"]
        self.division_threshold = params["division_threshold"]
        self.max_weight = params["max_weight"]
        self.connections = set()
        self.activations = 2 ** params["max_depth"] + 1  # Number of layers in the network.
        activation_functions = neat.activations.ActivationFunctionSet()
        self.activation = activation_functions.get(params["activation"])
        self.width = len(substrate.output_coordinates)
        self.root_x = self.width/2
        self.root_y = (len(substrate.input_coordinates)/self.width)/2


    # creates phenotype with n dimensions
    def create_phenotype_network_nd(self, filename=None):
        rnn_params = self.es_hyperneat_nd_tensors()
        
        return RecurrentNet(
            n_inputs = rnn_params["n_inputs"],
            n_outputs = rnn_params["n_outputs"],
            n_hidden = rnn_params["n_hidden"],
            output_to_hidden = rnn_params["output_to_hidden"],
            output_to_output = rnn_params["output_to_output"],
            hidden_to_hidden = rnn_params["hidden_to_hidden"],
            input_to_hidden = rnn_params["input_to_hidden"],
            input_to_output = rnn_params["input_to_output"],
            hidden_to_output = rnn_params["hidden_to_output"],
            hidden_responses = rnn_params["hidden_responses"],
            output_responses = rnn_params["output_responses"],
            hidden_biases = rnn_params["hidden_biases"],
            output_biases = rnn_params["output_biases"]
        )

    def division_initialization_nd_tensors(self, coords, outgoing):
        root = BatchednDimensionTree([0.0 for x in range(len(coords[0]))], 1.0, 1)
        q = [root]
        while q:
            p = q.pop(0)
            # here we will subdivide to 2^coordlength as described above
            # this allows us to search from +- midpoints on each axis of the input coord
            p.divide_childrens()
            out_coords = []
            weights = query_torch_cppn_tensors(coords, p.child_coords, outgoing, self.cppn, self.max_weight)
            #print(weights)
            low_var_count = 0
            for x in range(len(coords)):
                if(torch.var(weights[: ,x]) < self.division_threshold):
                    low_var_count += 1 
            for idx,c in enumerate(p.cs):
                c.w = weights[idx]
            if (p.lvl < self.initial_depth) or (p.lvl < self.max_depth and low_var_count != len(coords)):
                    q.extend(p.cs)
        return root

    def prune_all_the_tensors_aha(self, coords, p, outgoing):
        coord_len = len(coords[0])
        num_coords = len(coords)
        for c in p.cs:
            # where the magic shall bappen
            if(torch.var(c.w) >= self.variance_threshold):
                self.prune_all_the_tensors_aha(coords, c, outgoing)
            else:
                tree_coords = []
                tree_coords_2 = []
                child_array = []
                sign = 1
                #gotta be a better way to accomplish this permutation
                for i in range(coord_len):
                    query_coord = []
                    query_coord2 = []
                    dimen = c.coord[i] - p.width
                    dimen2 = c.coord[i] + p.width
                    for x in range(coord_len):
                        if x != i:
                            query_coord.append(c.coord[x])
                            query_coord2.append(c.coord[x])
                        else:
                            query_coord.append(dimen2)
                            query_coord2.append(dimen)
                    tree_coords.append(query_coord)
                    tree_coords.append(query_coord2)
                con = None
                weights = abs(c.w - query_torch_cppn_tensors(coords, tree_coords, outgoing, self.cppn, self.max_weight))
                for x in range(num_coords):
                    # group each dimensional permutation for plus/minus offsets 
                    #print(weights[:,x])
                    grouped = torch.reshape(weights[: ,x], [weights.shape[0] // 2, 2])
                    mins = torch.min(grouped, dim=1)
                    #print("mins: ")
                    #print(mins[0])
                    if( torch.max(mins[0]) > self.band_threshold):
                        if outgoing:
                            con = nd_Connection(coords[x], c.coord, c.w[x])
                        else:
                            con = nd_Connection(c.coord, coords[x], c.w[x])
                    if con is not None:
                        #print(con.weight)
                        if not con.weight == 0.0:
                            self.connections.add(con)
        return
            
    def es_hyperneat_nd_tensors(self):
        inputs = self.substrate.input_coordinates
        #print(inputs)
        outputs = self.substrate.output_coordinates
        hidden_full = []
        hidden_nodes, unexplored_hidden_nodes, hidden_ids = [], [], []
        connections1, connections2, connections3 = set(), set(), set()
        root = self.division_initialization_nd_tensors(inputs, True)
        self.prune_all_the_tensors_aha(inputs, root, True)
        connections1 = connections1.union(self.connections)
        #print(connections1)
        for c in connections1:
            hidden_nodes.append(tuple(c.coord2))
        hidden_full.extend([c for c in hidden_nodes])
        self.connections = set()
        unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)
        if(len(unexplored_hidden_nodes) != 0):
            root = self.division_initialization_nd_tensors(unexplored_hidden_nodes, True)
            self.prune_all_the_tensors_aha(unexplored_hidden_nodes, root, True)
            connections2 = connections2.union(self.connections)
            for c in connections2:
                hidden_nodes.append(tuple(c.coord2))
            unexplored_hidden_nodes = set(unexplored_hidden_nodes)
            unexplored_hidden_nodes = set(hidden_nodes) - unexplored_hidden_nodes
            self.connections = set()
        hidden_full.extend([c for c in unexplored_hidden_nodes])
        root = self.division_initialization_nd_tensors(outputs, False)
        self.prune_all_the_tensors_aha(outputs, root, False)
        #print(connections1, connections2, connections3)
        connections3 = connections3.union(self.connections)
        temp = []
        for c in connections3:
            if(c.coord1 in hidden_full):
                temp.append(c)
        connections3 = set(temp)
        self.connections = set()
        rnn_params = self.structure_for_rnn(hidden_full, connections1, connections2, connections3)
        return rnn_params

    def structure_for_rnn(self, hidden_node_coords, conns_1, conns_2, conns_3):
        param_dict = {
            "n_inputs": len(self.substrate.input_coordinates),
            "n_outputs": len(self.substrate.output_coordinates),
            "n_hidden": len(hidden_node_coords),
            "hidden_responses": [1.0],
            "hidden_biases": [0.0],
            "output_responses": [1.0],
            "output_biases":  [0.0],
            "output_to_hidden": ([], []),
            "input_to_output": ([],[]),
            "output_to_output": ([],[])
        }
        temp_nodes = []
        temp_weights = []
        for c in conns_1:
            #print(c.coord1, c.coord2)
            temp_nodes.append((
                hidden_node_coords.index(c.coord2),
                self.substrate.input_coordinates.index(c.coord1)
            ))
            temp_weights.append(c.weight)
        param_dict["input_to_hidden"] = tuple([temp_nodes, temp_weights])
        #print(temp_nodes, temp_weights)
        temp_nodes, temp_weights = [], []
        #print(param_dict["input_to_hidden"])
        for c in conns_2:
            temp_nodes.append((
                hidden_node_coords.index(c.coord2),
                hidden_node_coords.index(c.coord1)
            ))
            temp_weights.append(c.weight)
        param_dict["hidden_to_hidden"] = tuple([temp_nodes, temp_weights])
        temp_nodes, temp_weights = [], []
        for c in conns_3:
            temp_nodes.append((
                self.substrate.output_coordinates.index(c.coord2),
                hidden_node_coords.index(c.coord1)
            ))
            temp_weights.append(c.weight)
        param_dict["hidden_to_output"] = tuple([temp_nodes, temp_weights])
        return param_dict

# a tree that subdivides n dimensional euclidean spaces
class BatchednDimensionTree:
    
    def __init__(self, in_coord, width, level):
        self.w = 0.0
        self.coord = in_coord
        self.width = width
        self.lvl = level
        self.num_children = 2**len(self.coord)
        self.child_coords = []
        self.cs = []
        self.signs = self.set_signs() 
        self.child_weights = 0.0
    def set_signs(self):
        return list(itertools.product([1,-1], repeat=len(self.coord)))
    
    def divide_childrens(self):
        for x in range(self.num_children):
            new_coord = []
            for y in range(len(self.coord)):
                new_coord.append(self.coord[y] + (self.width/(2*self.signs[x][y])))
            self.child_coords.append(new_coord)
            newby = BatchednDimensionTree(new_coord, self.width/2, self.lvl+1)
            self.cs.append(newby)
    
# new tree's corresponding connection structure
class nd_Connection:
    def __init__(self, coord1, coord2, weight):
        if(type(coord1) == list):
            coord1 = tuple(coord1)
        if(type(coord2) == list):
            coord2 = tuple(coord2)
        self.coord1 = coord1
        self.coords = coord1 + coord2
        self.weight = weight
        self.coord2 = coord2
    def __eq__(self, other):
        return self.coords == other.coords
    def __hash__(self):
        return hash(self.coords + (self.weight,))

def query_torch_cppn_tensors(coords_in, coords_out, outgoing, cppn, max_weight=5.0):
    inputs = get_nd_coord_inputs(coords_in, coords_out)
    activs = cppn(inputs)
    return activs