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
from pytorch_neat.aggregations import sum_aggregation as sum_ag
from pytorch_neat.cppn import Leaf, Node


def assert_almost_equal(x, y, tol):
    assert abs(x - y) < tol, "{!r} !~= {!r}".format(x, y)


def test_cppn_simple():
    shape = (2, 2)
    x = Leaf(name="x")
    y = Node([x], [1.0], 1.0, 0.0, identity, sum_ag, name="y")
    z = Node([x], [1.0], 1.0, 0.0, identity, sum_ag, name="z")
    x_activs = torch.full(shape, 3)
    x.set_activs(x_activs)
    assert np.allclose(x_activs, y.get_activs(shape).numpy())
    assert np.allclose(x_activs, z.get_activs(shape).numpy())


def test_cppn_unconnected():
    shape = (2, 2)
    x = Leaf(name="x")
    y = Node([], [1.0], 1.0, 0.5, identity, sum_ag, name="y")
    x_activs = torch.full(shape, 3)
    x.set_activs(x_activs)
    assert np.allclose(y.get_activs(shape).numpy(), np.full(shape, 0.5))


def test_cppn_call():
    leaves = {"x": Leaf(name="x"), "y": Leaf(name="y")}
    a = Node([leaves["x"]], [1.0], 1.0, 0.0, identity, sum_ag, name="a", leaves=leaves)
    b = Node(
        [leaves["x"], leaves["y"]],
        [1.0, 1.0],
        1.0,
        0.0,
        identity,
        sum_ag,
        name="b",
        leaves=leaves,
    )
    c = Node([a], [1.0], 1.0, 0.0, identity, sum_ag, leaves=leaves)

    shape = (2, 2)
    a_activs = a(x=torch.full(shape, 0.5), y=torch.full(shape, 2.0)).numpy()
    assert np.allclose(a_activs, np.full(shape, 0.5))
    b_activs = b(x=torch.full(shape, 1.5), y=torch.full(shape, 2.0))
    assert np.allclose(b_activs, np.full(shape, 3.5))
    c_activs = c(x=torch.full(shape, 5.5), y=torch.full(shape, 3.0))
    assert np.allclose(c_activs, np.full(shape, 5.5))


def test_cppn_deep_call():
    leaves = {"x": Leaf(name="x"), "y": Leaf(name="y")}
    a = Node([leaves["y"]], [1.0], 1.0, 0.0, identity, sum_ag, name="a", leaves=leaves)
    b = Node(
        [leaves["x"], a],
        [1.0, 1.0],
        1.0,
        0.0,
        identity,
        sum_ag,
        name="b",
        leaves=leaves,
    )
    c = Node([a], [1.0], 1.0, 0.0, identity, sum_ag, leaves=leaves)

    shape = (2, 2)
    b_activs = b(x=torch.full(shape, 1.5), y=torch.full(shape, 2.0))
    assert np.allclose(b_activs, np.full(shape, 3.5))
    c_activs = c(x=torch.full(shape, 5.5), y=torch.full(shape, 3.0))
    assert np.allclose(c_activs, np.full(shape, 3.0))
    b_activs = b(x=torch.full(shape, 1.5), y=torch.full(shape, 2.0))
    assert np.allclose(b_activs, np.full(shape, 3.5))
