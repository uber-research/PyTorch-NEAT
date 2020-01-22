import torch
from pytorch_neat.cppn import get_nd_coord_inputs, get_coord_inputs, create_cppn

def test_coords():
    input_coords = [[-1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]
    output_coords = [[-2.0, 0.0, 2.0], [0.0, 0.0, -2.0], [2.0, 0.0, 1.0]]
    
    input_coords_2d = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, -1.0]]
    output_coords_2d = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    
    inputs = torch.tensor(
        input_coords, dtype=torch.float32
    )
    outputs = torch.tensor(
        output_coords, dtype=torch.float32
    )
    inputs_2 = torch.tensor(
        input_coords_2d, dtype=torch.float32
    )
    outputs_2 = torch.tensor(
        output_coords_2d, dtype=torch.float32
    )
    print(get_coord_inputs(inputs_2, outputs_2))
    print(get_nd_coord_inputs(inputs, outputs))
test_coords()