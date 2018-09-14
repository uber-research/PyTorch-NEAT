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

import logging

import gym
import numpy as np

logger = logging.getLogger(__name__)


class MetaMazeEnv(gym.Env):
    def __init__(
        self,
        size=7,
        receptive_size=3,
        episode_len=250,
        wall_penalty=0.1,
        extra_inputs=True,
    ):
        self.size = size
        self.receptive_size = receptive_size
        self.center = size // 2
        self.episode_len = episode_len
        self.wall_penalty = wall_penalty
        self.extra_inputs = extra_inputs

        self.reward = 0.0
        self.step_num = self.episode_len
        self.reward_row_pos = self.center
        self.reward_col_pos = self.center
        self.row_pos = self.center
        self.col_pos = self.center

        self.make_maze()

    def make_maze(self):
        self.maze = np.ones((self.size, self.size))  # ones are walls
        self.maze[1 : self.size - 1, 1 : self.size - 1].fill(0)
        for row in range(1, self.size - 1):
            for col in range(1, self.size - 1):
                if row % 2 == 0 and col % 2 == 0:
                    self.maze[row, col] = 1
        self.maze[self.center, self.center] = 0

    def render(self, mode="human"):
        raise NotImplementedError()

    def state(self):
        if self.extra_inputs:
            state = np.zeros(self.receptive_size ** 2 + 3)
        else:
            state = np.zeros(self.receptive_size ** 2 + 1)
        state[: self.receptive_size ** 2] = self.maze[
            self.row_pos
            - self.receptive_size // 2 : self.row_pos
            + self.receptive_size // 2
            + 1,
            self.col_pos
            - self.receptive_size // 2 : self.col_pos
            + self.receptive_size // 2
            + 1,
        ].flatten()
        state[-1] = self.reward
        if self.extra_inputs:
            state[-2] = self.step_num
            state[-3] = 1  # bias
        return state

    def step(self, action):
        assert action in {0, 1, 2, 3}
        self.step_num += 1
        assert self.step_num <= self.episode_len
        self.reward = 0.0

        target_row = self.row_pos
        target_col = self.col_pos
        if action == 0:
            target_row -= 1
        elif action == 1:
            target_row += 1
        elif action == 2:
            target_col -= 1
        elif action == 3:
            target_col += 1

        if self.maze[target_row, target_col] == 1:
            self.reward = -self.wall_penalty
        else:
            self.row_pos = target_row
            self.col_pos = target_col

        if self.row_pos == self.reward_row_pos and self.col_pos == self.reward_col_pos:
            self.reward += 10.0
            self.row_pos = np.random.randint(1, self.size - 1)
            self.col_pos = np.random.randint(1, self.size - 1)
            while self.maze[self.row_pos, self.col_pos] == 1:
                self.row_pos = np.random.randint(1, self.size - 1)
                self.col_pos = np.random.randint(1, self.size - 1)

        return self.state(), self.reward, self.step_num == self.episode_len, {}

    def reset(self):
        self.step_num = 0
        self.reward = 0
        self.row_pos = self.center
        self.col_pos = self.center
        self.reward_row_pos = self.reward_col_pos = 0
        while self.maze[self.reward_row_pos, self.reward_col_pos] == 1:
            self.reward_row_pos = np.random.randint(1, self.size - 1)
            self.reward_col_pos = np.random.randint(1, self.size - 1)

        return self.state()

    def __repr__(self):
        return "MetaMazeEnv({}, step_num={}, pos={}, reward_pos={})".format(
            self.maze,
            self.step_num,
            (self.row_pos, self.col_pos),
            (self.reward_row_pos, self.reward_col_pos),
        )


class SimpleMazeEnv(MetaMazeEnv):
    def __init__(self, size=4, receptive_size=3, episode_len=250, wall_penalty=0.0):
        super().__init__(
            size=size,
            receptive_size=receptive_size,
            episode_len=episode_len,
            wall_penalty=wall_penalty,
        )

    def make_maze(self):
        self.maze = np.ones((self.size, self.size))  # ones are walls
        self.maze[1 : self.size - 1, 1 : self.size - 1].fill(0)

    def render(self, mode="human"):
        raise NotImplementedError()

    def __str__(self):
        return "SimpleMazeEnv({}, step_num={}, pos={}, reward_pos={})".format(
            self.maze,
            self.step_num,
            (self.row_pos, self.col_pos),
            (self.reward_row_pos, self.reward_col_pos),
        )
