import multiprocessing
import os

import click
import neat
import gym
# import torch
import numpy as np
import tensorflow as tf
from pytorch_neat.activations import tanh_activation
from pytorch_neat.adaptive_linear_net import AdaptiveLinearNet
from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.es_hyperneat import ESNetwork
from pytorch_neat.substrate import Substrate
from pytorch_neat.cppn import create_cppn



max_env_steps = 200


def make_env():
    return gym.make("MountainCar-v0")

def make_net(genome, config, bs):
    #start by setting up a substrate for this bad cartpole boi
    params = {"initial_depth": 1,
            "max_depth": 2,
            "variance_threshold": 0.8,
            "band_threshold": 0.05,
            "iteration_level": 3,
            "division_threshold": 0.3,
            "max_weight": 3.0,
            "activation": "tanh"}
    input_cords = [(0.1, 0.1, 0.1), (-0.1, -0.1, -0.1)]
    output_cords = [(0.0, 1.0, -1.0), (0.0, 0.0, -1.0), (0.0, -1.0, -1.0)]
    leaf_names = []
    for i in range(len(output_cords[0])):
        leaf_names.append(str(i) + "_in")
        leaf_names.append(str(i) + "_out")

    [cppn] = create_cppn(genome, config, leaf_names, ['cppn_out'])
    net_builder = ESNetwork(Substrate(input_cords, output_cords), cppn, params)
    net = net_builder.create_phenotype_network_nd('./genome_vis')
    return net

def activate_net(net, states):
    outputs = net.activate(states).numpy()
    #print(outputs)
    return outputs[0] > 0.5


@click.command()
@click.option("--n_generations", type=int, default=100)
def run(n_generations):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    evaluator = MultiEnvEvaluator(
        make_net, activate_net, make_env=make_env, max_env_steps=max_env_steps
    )

    def eval_genomes(genomes, config):
        for _, genome in genomes:
            genome.fitness = evaluator.eval_genome(genome, config)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    #logger = LogReporter("neat.log", evaluator.eval_genome)
    #pop.add_reporter(logger)

    pop.run(eval_genomes, n_generations)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
