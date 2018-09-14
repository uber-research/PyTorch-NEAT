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

import random

import gym
import numpy as np


class TurningTMazeEnv(gym.Env):
    def __init__(
        self,
        hall_len=3,
        n_trials=100,
        wall_penalty=0.4,
        init_reward_side=1,
        reward_flip_mean=50,
        reward_flip_range=15,
        high_reward=1.0,
        low_reward=0.2,
    ):
        self.hall_len = hall_len
        self.n_trials = n_trials
        self.trial_num = n_trials
        self.init_reward_side = init_reward_side
        self.reward_side = init_reward_side
        self.reward_flip_mean = reward_flip_mean
        self.reward_flip_range = reward_flip_range
        self.wall_penalty = wall_penalty
        self.high_reward = high_reward
        self.low_reward = low_reward

        self.color = 0.0
        self.row_pos = self.hall_len + 1
        self.col_pos = hall_len + 1
        self.direction = 0
        self.reward_flip = reward_flip_mean
        self.reset_trial_on_step = False
        self.trial_num = self.n_trials

        self.make_maze()

    def make_maze(self):
        self.maze = np.ones(
            (self.hall_len + 3, 2 * self.hall_len + 3)
        )  # ones are walls
        self.maze[1:-1, self.hall_len + 1].fill(0)
        self.maze[1, 1:-1].fill(0)

    def render(self, mode="human"):
        raise NotImplementedError()

    def state(self):
        state = np.zeros(4)
        assert self.direction in {0, 1, 2, 3}
        if self.direction == 0:  # up
            state[0] = self.maze[self.row_pos, self.col_pos - 1]
            state[1] = self.maze[self.row_pos - 1, self.col_pos]
            state[2] = self.maze[self.row_pos, self.col_pos + 1]
        elif self.direction == 1:  # right
            state[0] = self.maze[self.row_pos - 1, self.col_pos]
            state[1] = self.maze[self.row_pos, self.col_pos + 1]
            state[2] = self.maze[self.row_pos + 1, self.col_pos]
        elif self.direction == 2:  # down
            state[0] = self.maze[self.row_pos, self.col_pos + 1]
            state[1] = self.maze[self.row_pos + 1, self.col_pos]
            state[2] = self.maze[self.row_pos, self.col_pos - 1]
        elif self.direction == 3:  # left
            state[0] = self.maze[self.row_pos + 1, self.col_pos]
            state[1] = self.maze[self.row_pos, self.col_pos - 1]
            state[2] = self.maze[self.row_pos - 1, self.col_pos]
        state[3] = self.color
        return state

    def reset_trial(self):
        self.color = 0.0
        self.row_pos = self.hall_len + 1
        self.col_pos = self.hall_len + 1
        self.direction = 0
        if self.trial_num == self.reward_flip:
            self.reward_side = 1 - self.reward_side

    def step(self, action):
        assert action in {0, 1, 2}

        if self.reset_trial_on_step:
            self.trial_num += 1
            self.reset_trial()
            self.reset_trial_on_step = False
            return self.state(), 0.0, self.trial_num == self.n_trials, {}

        assert self.trial_num < self.n_trials

        reward = 0
        self.color = 0

        if action == 0:
            self.direction = (self.direction - 1) % 4
        elif action == 2:
            self.direction = (self.direction + 1) % 4
        elif action == 1:
            target_row = self.row_pos
            target_col = self.col_pos

            if self.direction == 0:  # up
                target_row -= 1
            elif self.direction == 1:  # right
                target_col += 1
            elif self.direction == 2:  # down
                target_row += 1
            elif self.direction == 3:  # left
                target_col -= 1

            if self.maze[target_row, target_col] == 1:
                reward -= self.wall_penalty
                self.reset_trial_on_step = True
            else:
                self.row_pos = target_row
                self.col_pos = target_col

        if self.row_pos == 1 and self.col_pos == 1:
            self.color = self.high_reward if self.reward_side == 0 else self.low_reward
            reward += self.color
            self.reset_trial_on_step = True
        elif self.row_pos == 1 and self.col_pos == 2 * self.hall_len + 1:
            self.color = self.high_reward if self.reward_side == 1 else self.low_reward
            reward += self.color
            self.reset_trial_on_step = True

        return self.state(), reward, False, {}

    def reset(self):
        self.trial_num = 0
        self.reset_trial_on_step = False
        self.reward_flip = self.reward_flip_mean + random.randint(
            -self.reward_flip_range, self.reward_flip_range
        )
        self.reward_side = self.init_reward_side
        self.reset_trial()
        return self.state()

    def __repr__(self):
        return "TurningTMazeEnv({}, step_num={}, pos={}, direction={}, reward_side={})".format(
            self.maze,
            self.trial_num,
            (self.row_pos, self.col_pos),
            self.direction,
            self.reward_side,
        )
