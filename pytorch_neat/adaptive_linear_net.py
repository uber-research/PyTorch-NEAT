# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import torch

from .activations import identity_activation, tanh_activation
from .cppn import clamp_weights_, create_cppn, get_coord_inputs


class AdaptiveLinearNet:
    def __init__(
        self,
        delta_w_node,
        input_coords,
        output_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        activation=tanh_activation,
        cppn_activation=identity_activation,
        batch_size=1,
        device="cuda:0",
    ):

        self.delta_w_node = delta_w_node

        self.n_inputs = len(input_coords)
        self.input_coords = torch.tensor(
            input_coords, dtype=torch.float32, device=device
        )

        self.n_outputs = len(output_coords)
        self.output_coords = torch.tensor(
            output_coords, dtype=torch.float32, device=device
        )

        self.weight_threshold = weight_threshold
        self.weight_max = weight_max

        self.activation = activation
        self.cppn_activation = cppn_activation

        self.batch_size = batch_size
        self.device = device
        self.reset()

    def get_init_weights(self, in_coords, out_coords, w_node):
        (x_out, y_out), (x_in, y_in) = get_coord_inputs(in_coords, out_coords)

        n_in = len(in_coords)
        n_out = len(out_coords)

        zeros = torch.zeros((n_out, n_in), dtype=torch.float32, device=self.device)

        weights = self.cppn_activation(
            w_node(
                x_out=x_out,
                y_out=y_out,
                x_in=x_in,
                y_in=y_in,
                pre=zeros,
                post=zeros,
                w=zeros,
            )
        )
        clamp_weights_(weights, self.weight_threshold, self.weight_max)

        return weights

    def reset(self):
        with torch.no_grad():
            self.input_to_output = (
                self.get_init_weights(
                    self.input_coords, self.output_coords, self.delta_w_node
                )
                .unsqueeze(0)
                .expand(self.batch_size, self.n_outputs, self.n_inputs)
            )

            self.w_expressed = self.input_to_output != 0

            self.batched_coords = get_coord_inputs(
                self.input_coords, self.output_coords, batch_size=self.batch_size
            )

    def activate(self, inputs):
        """
        inputs: (batch_size, n_inputs)

        returns: (batch_size, n_outputs)
        """
        with torch.no_grad():
            inputs = torch.tensor(
                inputs, dtype=torch.float32, device=self.device
            ).unsqueeze(2)

            outputs = self.activation(self.input_to_output.matmul(inputs))

            input_activs = inputs.transpose(1, 2).expand(
                self.batch_size, self.n_outputs, self.n_inputs
            )
            output_activs = outputs.expand(
                self.batch_size, self.n_outputs, self.n_inputs
            )

            (x_out, y_out), (x_in, y_in) = self.batched_coords

            delta_w = self.cppn_activation(
                self.delta_w_node(
                    x_out=x_out,
                    y_out=y_out,
                    x_in=x_in,
                    y_in=y_in,
                    pre=input_activs,
                    post=output_activs,
                    w=self.input_to_output,
                )
            )

            self.delta_w = delta_w

            self.input_to_output[self.w_expressed] += delta_w[self.w_expressed]
            clamp_weights_(
                self.input_to_output, weight_threshold=0.0, weight_max=self.weight_max
            )

        return outputs.squeeze(2)

    @staticmethod
    def create(
        genome,
        config,
        input_coords,
        output_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        output_activation=None,
        activation=tanh_activation,
        cppn_activation=identity_activation,
        batch_size=1,
        device="cuda:0",
    ):

        nodes = create_cppn(
            genome,
            config,
            ["x_in", "y_in", "x_out", "y_out", "pre", "post", "w"],
            ["delta_w"],
            output_activation=output_activation,
        )

        delta_w_node = nodes[0]

        return AdaptiveLinearNet(
            delta_w_node,
            input_coords,
            output_coords,
            weight_threshold=weight_threshold,
            weight_max=weight_max,
            activation=activation,
            cppn_activation=cppn_activation,
            batch_size=batch_size,
            device=device,
        )
