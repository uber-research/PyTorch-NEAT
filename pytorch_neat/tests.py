import torch
from .cppn import get_coord_inputs

class TestGetInputs(object):
    def __init__(self):
        input_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, -1.0]]
        output_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
        inputs = torch.tensor(
            input_coords, dtype=torch.float32, device=device
        )
        outputs = torch.tensor(
            output_coords, dtype=torch.float32, device=device
        )

        print(get_coord_inputs(inputs, outputs))

t = TestGetInputs()