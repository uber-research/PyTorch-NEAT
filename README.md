# PyTorch NEAT

## Background
NEAT (NeuroEvolution of Augmenting Topologies) is a popular neuroevolution algorithm, one of the few such algorithms that evolves the architectures of its networks in addition to the weights. For more information, see this research paper: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf.

HyperNEAT is an extension to NEAT that indirectly encodes the weights of the network (called the substrate) with a separate network (called a CPPN, for compositional pattern-producing network). For more information on HyperNEAT, see this website: http://eplex.cs.ucf.edu/hyperNEATpage/.

Adaptive HyperNEAT is an extension to HyperNEAT which indirectly encodes both the initial weights and an update rule for the weights such that some learning can occur during a network's "lifetime." For more information, see this research paper: http://eplex.cs.ucf.edu/papers/risi_sab10.pdf.

## About
PyTorch NEAT builds upon [NEAT-Python](https://github.com/CodeReclaimers/neat-python) by providing some functions which can turn a NEAT-Python genome into either a recurrent PyTorch network or a PyTorch CPPN for use in HyperNEAT or Adaptive HyperNEAT.
We also provide some environments in which to test NEAT and Adaptive HyperNEAT, and a more involved example using the CPPN infrastructure with Adaptive HyperNEAT on a T-maze.

## Examples
The following snippet turns a NEAT-Python genome into a recurrent PyTorch network:
```
from pytorch_neat.recurrent_net import RecurrentNet

net = RecurrentNet.create(genome, config, bs)
outputs = net.activate(some_array)
```

You can also turn a NEAT-Python genome into a CPPN:
```
from pytorch_neat.cppn import create_cppn

cppn_nodes = create_cppn(genome, config)
```

A CPPN is represented as a graph structure. For easy evaluation, a CPPN's input and output nodes may be named:
```
from pytorch_neat.cppn import create_cppn

[delta_w_node] = create_cppn(
    genome,
    config,
    ["x_in", "y_in", "x_out", "y_out", "pre", "post", "w"],
    ["delta_w"],
)

delta_w = delta_w_node(x_in=some_array, y_in=other_array, ...)
```

We also provide some infrastructure for running networks in Gym environments:
```
from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.recurrent_net import RecurrentNet

def make_net(genome, config, batch_size):
    return RecurrentNet.create(genome, config, batch_size)


def activate_net(net, states):
    outputs = net.activate(states).numpy()
    return outputs[:, 0] > 0.5

def make_env():
    return gym.make("CartPole-v0")

evaluator = MultiEnvEvaluator(
    make_net, activate_net, make_env=make_env, max_env_steps=max_env_steps, batch_size=batch_size,
)

fitness = evaluator.eval_genome(genome)
```
This allows multiple environments to run in parallel for efficiency.

A simple example using NEAT to solve the Cartpole can be run like this:
```
python3 -m examples.simple.main
```

And a simple example using Adaptive HyperNEAT to partially solve a T-maze can be run like this:
```
python3 -m examples.adaptive.main
```

## Author / Support

PyTorch NEAT is extended from Python NEAT by Alex Gajewsky.

Questions can be directed to joel.lehman@uber.com.
