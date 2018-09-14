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

import numpy as np
import torch

from pytorch_neat.activations import identity_activation as identity
from pytorch_neat.adaptive_linear_net import AdaptiveLinearNet
from pytorch_neat.aggregations import sum_aggregation as sum_ag
from pytorch_neat.cppn import Leaf, Node


def slow_tanh(x):
    return torch.tanh(0.5 * x)


def np_tanh(x):
    return torch.tanh(0.5 * torch.tensor(x)).numpy()


def test_pre():
    leaves = {
        name: Leaf(name=name)
        for name in ["x_in", "y_in", "x_out", "y_out", "pre", "post", "w"]
    }

    delta_w_node = Node(
        [leaves["x_in"], leaves["x_out"], leaves["pre"]],
        [1.0, 2.0, 3.0],
        1.0,
        0.0,
        identity,
        sum_ag,
        name="delta_w",
        leaves=leaves,
    )

    input_coords = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0]]
    output_coords = [[-1.0, 0.0], [1.0, 0.0]]

    net = AdaptiveLinearNet(
        delta_w_node,
        input_coords,
        output_coords,
        activation=slow_tanh,
        cppn_activation=slow_tanh,
        device="cpu",
    )

    w = np_tanh(
        np.array(
            [
                [-1.0 + 2 * -1.0, 1.0 + 2 * -1.0, 2 * -1.0],
                [-1.0 + 2 * 1.0, 1.0 + 2 * 1.0, 2 * 1.0],
            ],
            dtype=np.float32,
        )
    )
    w[np.abs(w) < 0.2] = 0
    w[w < 0] += 0.2
    w[w > 0] -= 0.2
    w[w > 3.0] = 3.0
    w[w < -3.0] = -3.0
    w_expressed = w != 0
    assert np.allclose(net.input_to_output.numpy(), w)

    for _ in range(3):
        inputs = np.array([[-1.0, 2.0, 3.0]], dtype=np.float32)
        outputs = net.activate(inputs)
        activs = np.tanh(0.5 * w.dot(inputs[0]))
        assert np.allclose(outputs, activs)

        delta_w = np_tanh(
            np.array(
                [
                    [-1.0 + 2 * -1.0, 1.0 + 2 * -1.0, 2 * -1.0],
                    [-1.0 + 2 * 1.0, 1.0 + 2 * 1.0, 2 * 1.0],
                ],
                dtype=np.float32,
            )
            + 3 * inputs
        )
        # delta_w[np.abs(delta_w) < 0.2] = 0
        # delta_w[delta_w < 0] += 0.2
        # delta_w[delta_w > 0] -= 0.2
        w[w_expressed] += delta_w[w_expressed]
        w[w > 3.0] = 3.0
        w[w < -3.0] = -3.0
        assert np.allclose(net.input_to_output.numpy(), w)


def test_w():
    leaves = {
        name: Leaf(name=name)
        for name in ["x_in", "y_in", "x_out", "y_out", "pre", "post", "w"]
    }

    delta_w_node = Node(
        [leaves["x_in"], leaves["x_out"], leaves["w"]],
        [1.0, 2.0, 3.0],
        1.0,
        0.0,
        identity,
        sum_ag,
        name="delta_w",
        leaves=leaves,
    )

    input_coords = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0]]
    output_coords = [[-1.0, 0.0], [1.0, 0.0]]

    net = AdaptiveLinearNet(
        delta_w_node,
        input_coords,
        output_coords,
        activation=slow_tanh,
        cppn_activation=slow_tanh,
        device="cpu",
    )

    w = np_tanh(
        np.array(
            [
                [-1.0 + 2 * -1.0, 1.0 + 2 * -1.0, 2 * -1.0],
                [-1.0 + 2 * 1.0, 1.0 + 2 * 1.0, 2 * 1.0],
            ],
            dtype=np.float32,
        )
    )
    w[np.abs(w) < 0.2] = 0
    w[w < 0] += 0.2
    w[w > 0] -= 0.2
    w[w > 3.0] = 3.0
    w[w < -3.0] = -3.0
    w_expressed = w != 0
    assert np.allclose(net.input_to_output.numpy(), w)

    for _ in range(3):
        inputs = np.array([[-1.0, 2.0, 3.0]])
        outputs = net.activate(inputs)[0]
        activs = np.tanh(0.5 * w.dot(inputs[0]))
        assert np.allclose(outputs, activs)

        delta_w = np_tanh(
            np.array(
                [
                    [-1.0 + 2 * -1.0, 1.0 + 2 * -1.0, 2 * -1.0],
                    [-1.0 + 2 * 1.0, 1.0 + 2 * 1.0, 2 * 1.0],
                ],
                dtype=np.float32,
            )
            + 3 * w
        )
        # delta_w[np.abs(delta_w) < 0.2] = 0
        # delta_w[delta_w < 0] += 0.2
        # delta_w[delta_w > 0] -= 0.2
        w[w_expressed] += delta_w[w_expressed]
        w[w > 3.0] = 3.0
        w[w < -3.0] = -3.0
        assert np.allclose(net.input_to_output.numpy(), w)


def test_post():
    leaves = {
        name: Leaf(name=name)
        for name in ["x_in", "y_in", "x_out", "y_out", "pre", "post", "w"]
    }

    delta_w_node = Node(
        [leaves["x_in"], leaves["x_out"], leaves["post"]],
        [1.0, 2.0, 3.0],
        1.0,
        0.0,
        identity,
        sum_ag,
        name="delta_w",
        leaves=leaves,
    )

    input_coords = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0]]
    output_coords = [[-1.0, 0.0], [1.0, 0.0]]

    net = AdaptiveLinearNet(
        delta_w_node,
        input_coords,
        output_coords,
        activation=slow_tanh,
        cppn_activation=slow_tanh,
        device="cpu",
    )

    w = np_tanh(
        np.array(
            [
                [-1.0 + 2 * -1.0, 1.0 + 2 * -1.0, 2 * -1.0],
                [-1.0 + 2 * 1.0, 1.0 + 2 * 1.0, 2 * 1.0],
            ],
            dtype=np.float32,
        )
    )
    w[np.abs(w) < 0.2] = 0
    w[w < 0] += 0.2
    w[w > 0] -= 0.2
    w[w > 3.0] = 3.0
    w[w < -3.0] = -3.0
    w_expressed = w != 0
    assert np.allclose(net.input_to_output.numpy(), w)

    for _ in range(3):
        inputs = np.array([[-1.0, 2.0, 3.0]])
        outputs = net.activate(inputs)[0]
        activs = np.tanh(0.5 * w.dot(inputs[0]))
        assert np.allclose(outputs, activs)

        delta_w = np_tanh(
            np.array(
                [
                    [-1.0 + 2 * -1.0, 1.0 + 2 * -1.0, 2 * -1.0],
                    [-1.0 + 2 * 1.0, 1.0 + 2 * 1.0, 2 * 1.0],
                ],
                dtype=np.float32,
            )
            + 3 * np.expand_dims(activs, 1)
        )
        # delta_w[np.abs(delta_w) < 0.2] = 0
        # delta_w[delta_w < 0] += 0.2
        # delta_w[delta_w > 0] -= 0.2
        w[w_expressed] += delta_w[w_expressed]
        w[w > 3.0] = 3.0
        w[w < -3.0] = -3.0
        assert np.allclose(net.input_to_output.numpy(), w)
