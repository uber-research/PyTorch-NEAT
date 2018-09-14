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

import json
import time
from pprint import pprint

import numpy as np
from neat.reporting import BaseReporter


class LogReporter(BaseReporter):
    def __init__(self, fnm, eval_best, eval_with_debug=False):
        self.log = open(fnm, "a")
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.eval_best = eval_best
        self.eval_with_debug = eval_with_debug
        self.log_dict = {}

    def start_generation(self, generation):
        self.log_dict["generation"] = generation
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        self.log_dict["pop_size"] = ng

        ns = len(species_set.species)
        self.log_dict["n_species"] = ns

        elapsed = time.time() - self.generation_start_time
        self.log_dict["time_elapsed"] = elapsed

        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = np.mean(self.generation_times)
        self.log_dict["time_elapsed_avg"] = average

        self.log_dict["n_extinctions"] = self.num_extinctions

        pprint(self.log_dict)
        self.log.write(json.dumps(self.log_dict) + "\n")

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = np.mean(fitnesses)
        fit_std = np.std(fitnesses)

        self.log_dict["fitness_avg"] = fit_mean
        self.log_dict["fitness_std"] = fit_std

        self.log_dict["fitness_best"] = best_genome.fitness

        print("=" * 50 + " Best Genome: " + "=" * 50)
        if self.eval_with_debug:
            print(best_genome)

        best_fitness_val = self.eval_best(
            best_genome, config, debug=self.eval_with_debug
        )
        self.log_dict["fitness_best_val"] = best_fitness_val

        n_neurons_best, n_conns_best = best_genome.size()
        self.log_dict["n_neurons_best"] = n_neurons_best
        self.log_dict["n_conns_best"] = n_conns_best

    def complete_extinction(self):
        self.num_extinctions += 1

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass
