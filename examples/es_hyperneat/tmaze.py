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

import multiprocessing
import os

import click
import neat

# import torch
import numpy as np
from pytorch_neat import t_maze
from pytorch_neat.activations import tanh_activation
from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.es_hyperneat import ESNetwork
from pytorch_neat.substrate import Substrate
from pytorch_neat.cppn import create_cppn

batch_size = 4
DEBUG = True


def make_net(genome, config, _batch_size):
    params = {"initial_depth": 1,
            "max_depth": 2,
            "variance_threshold": 0.8,
            "band_threshold": 0.05,
            "iteration_level": 3,
            "division_threshold": 0.3,
            "max_weight": 30.0,
            "activation": "tanh"}
    input_cords = [(-1.0, 1.0, 0.0), (-.5, 1.0, 0.0), (.5, 1.0, 0.0), (1.0, 1.0, 0.0)]
    output_cords = [(-1.0, -1.0, -1.0), (0.0, -1.0, -1.0), (1.0, -1.0, -1.0)]
    leaf_names = []
    for i in range(len(output_cords[0])):
        leaf_names.append(str(i) + "_in")
        leaf_names.append(str(i) + "_out")
    [cppn] = create_cppn(genome, config, leaf_names, ['cppn_out'])
    net_builder = ESNetwork(Substrate(input_cords, output_cords), cppn, params)
    net = net_builder.create_phenotype_network_nd('./genome_vis')
    return net

def activate_net(net, states, debug=False, step_num=0):
    '''
    net.substrate.input_coordinates = []
    adjust_input_coords = [[0.0, 0.0, 0.0] for x in range(4)]
    sign = 1
    for s in range(len(states[0])):
        adj = states[0][s]/4
        sign *= -1
        for x in range(3):
            adjust_input_coords[s][x] += sign * adj
            sign *= -1
    adjust_input_coords = [tuple(c) for c in adjust_input_coords]
    net.substrate.input_coordinates = adjust_input_coords
    pheno = net.create_phenotype_network_nd('./genome_vis')
    outputs = pheno.activate(states).numpy()
    '''
    outputs = net.activate(states).numpy()
    return np.argmax(outputs, axis=1)


@click.command()
@click.option("--n_generations", type=int, default=10000)
@click.option("--n_processes", type=int, default=1)
def run(n_generations, n_processes):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), "tmaze.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    envs = [t_maze.TMazeEnv(init_reward_side=i, n_trials=100) for i in [1, 0, 1, 0]]

    evaluator = MultiEnvEvaluator(
        make_net, activate_net, envs=envs, batch_size=batch_size, max_env_steps=1000
    )

    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)

        def eval_genomes(genomes, config):
            fitnesses = pool.starmap(
                evaluator.eval_genome, ((genome, config) for _, genome in genomes)
            )
            for (_, genome), fitness in zip(genomes, fitnesses):
                genome.fitness = fitness

    else:

        def eval_genomes(genomes, config):
            for i, (_, genome) in enumerate(genomes):
                try:
                    genome.fitness = evaluator.eval_genome(
                        genome, config, debug=DEBUG and i % 100 == 0
                    )
                except Exception as e:
                    print(genome)
                    raise e

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("log.json", evaluator.eval_genome)
    pop.add_reporter(logger)

    winner = pop.run(eval_genomes, n_generations)

    print(winner)
    final_performance = evaluator.eval_genome(winner, config)
    print("Final performance: {}".format(final_performance))
    generations = reporter.generation + 1
    return generations


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
