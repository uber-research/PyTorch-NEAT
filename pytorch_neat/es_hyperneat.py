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
        self.root_tree = nDimensionTree((0.0, 0.0, 0.0), 1.0, 1)


    # creates phenotype with n dimensions
    def create_phenotype_network_nd(self, filename=None):
        self.es_hyperneat_nd_tensors()
        input_coordinates = self.substrate.input_coordinates
        output_coordinates = self.substrate.output_coordinates

        input_nodes = range(len(input_coordinates))
        output_nodes = range(len(input_nodes), len(input_nodes)+len(output_coordinates))
        hidden_idx = len(input_coordinates)+len(output_coordinates)

        coordinates, indices, draw_connections, node_evals = [], [], [], []
        nodes = {}

        coordinates.extend(input_coordinates)
        coordinates.extend(output_coordinates)
        indices.extend(input_nodes)
        indices.extend(output_nodes)
       
        # Map input and output coordinates to their IDs. 
        coords_to_id = dict(zip(coordinates, indices))
        
        # Where the magic happens.
        hidden_nodes, connections = self.es_hyperneat_nd_tensors()
        
        for cs in hidden_nodes:
            coords_to_id[cs] = hidden_idx
            hidden_idx += 1
        for cs, idx in coords_to_id.items():
            for c in connections:
                if c.coord2 == cs:
                    draw_connections.append(c)
                    if idx in nodes:
                        initial = nodes[idx]
                        initial.append((coords_to_id[c.coord1], c.weight))
                        nodes[idx] = initial
                    else:
                        nodes[idx] = [(coords_to_id[c.coord1], c.weight)]
                        
        for idx, links in nodes.items():
            node_evals.append((idx, self.activation, sum, 0.0, 1.0, links))
                    
        # Visualize the network?
        if filename is not None:
            #draw_es_nd(coords_to_id, draw_connections, filename)
            print("not today yeahhaaa")
                    
        return RecurrentNet.create_from_es(input_nodes, output_nodes, node_evals)
        
    # Create a RecurrentNetwork using the ES-HyperNEAT approach.
  
    # Recursively collect all weights for a given QuadPoint.
    @staticmethod
    def get_weights(p):
        temp = []

        def loop(pp):
            if pp is not None and all(child is not None for child in pp.cs):
                if len(pp.cs) > 0:
                    for i in range(0, pp.num_children):
                        loop(pp.cs[i])
            else:
                if pp is not None:
                    temp.append(pp.w)
        loop(p)
        return temp

    # Find the variance of a given QuadPoint.
    def variance(self, p):
        if not p:
            return 0.0
        return np.var(self.get_weights(p))
    '''
    TO BE USED WITH TENSOR IMPLEMENTATION
    def tensor_variance(self, p, ix):
        if not p:
            return 0.0
        return torch.var(p.coords[ix])
    '''

    @staticmethod
    def get_weights_tensor(p):
        weights = []
        return

    def initialize_at_depth(self, depth=3):
        root_coord = []
        for s in range(depth):
            root_coord.append(0.0)
        
        root = nDimensionTree(root_coord, 1.0, 1)
        return root

    def division_initialization_nd(self, coord, outgoing):
        root = self.root_tree
        q = [root]
        while q:
            p = q.pop(0)
            # here we will subdivide to 2^coordlength as described above
            # this allows us to search from +- midpoints on each axis of the input coord
            p.divide_childrens()
            for c in p.cs:
                c.w = query_torch_cppn(coord, c.coord, outgoing, self.cppn, self.max_weight)
            
            if (p.lvl < self.initial_depth) or (p.lvl < self.max_depth and self.variance(p) > self.division_threshold):
                for child in p.cs:
                    q.append(child)

        return root

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
            for idx,c in enumerate(p.cs):
                c.w = weights[idx]
            if (p.lvl < self.initial_depth) or (p.lvl < self.max_depth):
                low_var_count = 0
                for x in range(len(coords)):
                    if(torch.var(weights[: ,x]) < self.division_threshold):
                        weights[: ,x] = weights[: ,x] * 0.0
                        low_var_count += 1
                if low_var_count != len(coords): 
                    for idx,c in enumerate(p.cs):
                        c.w = weights[idx]
                    q.extend(p.cs)
        return root

    def prune_all_the_tensors_aha(self, coords, p, outgoing):
        coord_len = len(coords[0])
        num_coords = len(coords)
        for c in p.cs:
            # where the magic shall bappen
            tree_coords = []
            tree_coords_2 = []
            child_array = []
            sign = 1
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
                grouped = torch.reshape(weights[: ,x], [weights.shape[0] // 2, 2])
                mins = torch.min(grouped, dim=1)
                if( torch.max(mins[0]) > self.band_threshold):
                    if outgoing:
                        con = nd_Connection(coords[x], c.coord, c.w[x])
                    else:
                        con = nd_Connection(c.coord, coords[x], c.w[x])
                if con is not None:
                    if not c.w[x] == 0.0:
                        self.connections.add(con)
        #print(self.connections)
        return

    # n-dimensional pruning and extradition
    def prune_all_the_dimensions(self, coord, p, outgoing):
        coord_len = len(coord)
        for c in p.cs:
            child_array = []
            if self.variance(c) > self.variance_threshold:
                self.prune_all_the_dimensions(coord, c, outgoing)
            else:
                #c_len = len(child_array)
                sign = 1
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
                    child_array.append(abs(c.w - query_torch_cppn(coord, query_coord, outgoing, self.cppn, self.max_weight)))
                    child_array.append(abs(c.w - query_torch_cppn(coord, query_coord2, outgoing, self.cppn, self.max_weight)))
                con = None
                max_val = 0.0
                cntrl = len(child_array)-1
                for new_ix in range(cntrl):
                    if(min(child_array[new_ix], child_array[new_ix+1]) > max_val):
                        max_val = min(child_array[new_ix], child_array[new_ix+1])
                if max_val > self.band_threshold:
                    if outgoing:
                        con = nd_Connection(coord, c.coord, c.w)
                    else:
                        con = nd_Connection(c.coord, coord, c.w)
                if con is not None:
                    if not c.w == 0.0:
                        print("adding conn")
                        self.connections.add(con)

    # Explores the hidden nodes and their connections.
    def es_hyperneat_nd(self):
        inputs = self.substrate.input_coordinates
        outputs = self.substrate.output_coordinates
        hidden_nodes, unexplored_hidden_nodes = set(), set()
        connections1, connections2, connections3 = set(), set(), set()
        
        for i in inputs:
            root = self.division_initialization_nd(i, True)
            self.prune_all_the_dimensions(i, root, True)
            connections1 = connections1.union(self.connections)
            for c in connections1:
                hidden_nodes.add(tuple(c.coord2))
            self.connections = set()

        unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)

        for i in range(self.iteration_level):
            for index_coord in unexplored_hidden_nodes:
                root = self.division_initialization_nd(index_coord, True)
                self.prune_all_the_dimensions(index_coord, root, True)
                connections2 = connections2.union(self.connections)
                for c in connections2:
                    hidden_nodes.add(tuple(c.coord2))
                self.connections = set()
        
        unexplored_hidden_nodes -= hidden_nodes
        
        for c_index in range(len(outputs)):
            root = self.division_initialization_nd(outputs[c_index], False)
            self.prune_all_the_dimensions(outputs[c_index], root, False)
            connections3 = connections3.union(self.connections)
            self.connections = set()
        connections = connections1.union(connections2.union(connections3))
        return self.clean_n_dimensional(connections)
            
    def es_hyperneat_nd_tensors(self):
        inputs = self.substrate.input_coordinates
        outputs = self.substrate.output_coordinates
        hidden_nodes, unexplored_hidden_nodes = [], []
        connections1, connections2, connections3 = set(), set(), set()
        root = self.division_initialization_nd_tensors(inputs, True)
        self.prune_all_the_tensors_aha(inputs, root, True)
        connections1 = connections1.union(self.connections)
        for c in connections1:
            hidden_nodes.append(tuple(c.coord2))
        self.connections = set()
        unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)
        if(len(unexplored_hidden_nodes) != 0):
            root = self.division_initialization_nd_tensors(unexplored_hidden_nodes, True)
            self.prune_all_the_tensors_aha(unexplored_hidden_nodes, root, True)
            connections2 = connections2.union(self.connections)
            for c in connections2:
                hidden_nodes.append(tuple(c.coord2))
            unexplored_hidden_nodes = set(unexplored_hidden_nodes)
            unexplored_hidden_nodes -= set(hidden_nodes)
        root = self.division_initialization_nd_tensors(outputs, False)
        self.prune_all_the_tensors_aha(outputs, root, False)
        connections1 = connections2.union(self.connections)
        connections = connections1.union(connections2.union(connections3))
        return self.clean_n_dimensional(connections)

    def es_hyperneat(self):
        inputs = self.substrate.input_coordinates
        outputs = self.substrate.output_coordinates
        hidden_nodes, unexplored_hidden_nodes = set(), set()
        connections1, connections2, connections3 = set(), set(), set()        

        for x, y in inputs:  # Explore from inputs.
            root = self.division_initialization((x, y), True)
            self.pruning_extraction((x, y), root, True)
            connections1 = connections1.union(self.connections)
            for c in connections1:
                hidden_nodes.add((c.x2, c.y2))
            self.connections = set()

        unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)
        
        for i in range(self.iteration_level):  # Explore from hidden.
            for x, y in unexplored_hidden_nodes:
                root = self.division_initialization((x, y), True)
                self.pruning_extraction((x, y), root, True)
                connections2 = connections2.union(self.connections)
                for c in connections2:
                    hidden_nodes.add((c.x2, c.y2))
                self.connections = set()
        
        unexplored_hidden_nodes -= hidden_nodes

        for x, y in outputs:  # Explore to outputs.
            root = self.division_initialization((x, y), False)      
            self.pruning_extraction((x, y), root, False)
            connections3 = connections3.union(self.connections)
            self.connections = set()

        connections = connections1.union(connections2.union(connections3))

        return self.clean_n_dimensional(connections)

    # clean n dimensional net
    def clean_n_dimensional(self, connections):
        connect_to_inputs = set(tuple(i) for i in self.substrate.input_coordinates)
        connect_to_outputs = set(tuple(i) for i in self.substrate.output_coordinates)
        true_connections = set()
        initial_input_connections = copy.deepcopy(connections)
        initial_output_connections = copy.deepcopy(connections)
        
        add_happened = True
        while add_happened:
            add_happened = False
            temp_input_connections = copy.deepcopy(initial_input_connections)
            for c in temp_input_connections:
                if c.coord1 in connect_to_inputs:
                    connect_to_inputs.add(c.coord2)
                    initial_input_connections -= {c}
                    add_happened = True
        add_happened = True
        while add_happened:
            add_happened = False
            temp_output_connections = copy.deepcopy(initial_output_connections)
            for c in temp_output_connections:
                if c.coord2 in connect_to_outputs:
                    connect_to_outputs.add(c.coord1)
                    initial_output_connections -= {c}
                    add_happened = True
        true_nodes = connect_to_inputs.intersection(connect_to_outputs)
        for c in connections:
            if (c.coord1 in true_nodes) and (c.coord2 in true_nodes):
                true_connections.add(c)
        true_nodes -= (set(self.substrate.input_coordinates).union(set(self.substrate.output_coordinates)))
        return true_nodes, true_connections
        
        
    # Clean a net for dangling connections by intersecting paths from input nodes with paths to output.


# Class representing an area in the quadtree defined by a center coordinate and the distance to the edges of the area. 
class QuadPoint:

    def __init__(self, x, y, width, lvl):
        self.x = x
        self.y = y
        self.w = 0.0
        self.width = width
        self.cs = [None] * 4
        self.lvl = lvl

#
class nDimensionTree:
    
    def __init__(self, in_coord, width, level):
        self.w = 0.0
        self.coord = in_coord
        self.width = width
        self.lvl = level
        self.num_children = 2**len(self.coord)
        self.cs = []
        self.child_coords = []
        self.signs = self.set_signs()
        #print(self.signs)
    def set_signs(self):
        return list(itertools.product([1,-1], repeat=len(self.coord)))
    
    def divide_childrens(self):
        for x in range(self.num_children):
            new_coord = []
            for y in range(len(self.coord)):
                new_coord.append(self.coord[y] + (self.width/(2*self.signs[x][y])))
            newby = nDimensionTree(new_coord, self.width/2, self.lvl+1)
            self.child_coords.append(new_coord)
            self.cs.append(newby)

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
            newby = nDimensionTree(new_coord, self.width/2, self.lvl+1)
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
# Class representing a connection from one point to another with a certain weight.
class Connection:
    
    def __init__(self, x1, y1, x2, y2, weight):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.weight = weight

    # Below is needed for use in set.
    def __eq__(self,other):
        return self.x1, self.y1, self.x2, self.y2 == other.x1, other.y1, other.x2, other.y2

    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2, self.weight))


# From a given point, query the cppn for weights to all other points. This can be visualized as a connectivity pattern.
def find_pattern(cppn, coord, res=60, max_weight=5.0):
    im = np.zeros((res, res))

    for x2 in range(res):
        for y2 in range(res):

            x2_scaled = -1.0 + (x2/float(res))*2.0 
            y2_scaled = -1.0 + (y2/float(res))*2.0
            
            i = [coord[0], coord[1], x2_scaled, y2_scaled, 1.0]
            n = cppn.activate(i)[0]

            im[x2][y2] = n * max_weight

    return im

def query_torch_cppn(coord1, coord2, outgoing, cppn, max_weight=5.0):
    result = 0.0
    num_dimen = len(coord1)
    master = {}
    for x in range(num_dimen):
        if(outgoing):
            master["leaf_one_"+str(x)] = np.array(coord1[x])
            master["leaf_two_"+str(x)] = np.array(coord2[x])
        else:
            master["leaf_one_"+str(x)] = np.array(coord2[x])
            master["leaf_two_"+str(x)] = np.array(coord1[x])
    #master = np.array(master)
    activs = cppn(master)
    print(activs)
    w = float(activs[0])
    print(w)
    
    if abs(w) > 0.2:  # If abs(weight) is below threshold, treat weight as 0.0.
        return w * max_weight
    else:
        return 0.0

def query_torch_cppn_tensors(coords_in, coords_out, outgoing, cppn, max_weight=5.0):
    inputs = get_nd_coord_inputs(coords_in, coords_out)
    '''
    for x in range(num_dimen):
        if(outgoing):
            master["leaf_one_"+str(x)] = np.array(coord1[x])
            master["leaf_two_"+str(x)] = np.array(coord2[x])
        else:
            master["leaf_one_"+str(x)] = np.array(coord2[x])
            master["leaf_two_"+str(x)] = np.array(coord1[x])
    '''
    #master = np.array(master)
    activs = cppn(inputs)
    #print("weights", activs)
    return activs