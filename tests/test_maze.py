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

import pytest

from pytorch_neat.maze import MetaMazeEnv


def test_default_initialization():
    env = MetaMazeEnv()
    assert env.size == 7
    assert env.maze.shape == (7, 7)
    assert env.receptive_size == 3
    assert env.episode_len == 250
    assert (
        env.maze
        == [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]
    ).all()
    assert env.maze[3, 3] == 0
    assert env.center == 3


def test_step_without_reset():
    env = MetaMazeEnv()
    with pytest.raises(AssertionError):
        env.step(3)


def test_render():
    env = MetaMazeEnv()
    with pytest.raises(NotImplementedError):
        env.render()


def test_step_with_reset():
    env = MetaMazeEnv()
    obs = env.reset()
    assert obs.shape == (12,)
    assert env.row_pos == env.col_pos == 3
    obs, reward, done, _ = env.step(3)
    assert obs.shape == (12,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_step_reward():
    env = MetaMazeEnv()
    obs = env.reset()
    assert (obs == [1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0.0]).all()
    env.reward_row_pos = env.reward_col_pos = 1
    assert env.row_pos == env.col_pos == 3

    obs, reward, done, _ = env.step(2)
    assert (obs == [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0.0]).all()
    assert not done
    assert reward == 0
    assert env.row_pos == 3
    assert env.col_pos == 2

    obs, reward, done, _ = env.step(1)
    assert (obs == [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 2, -0.1]).all()
    assert not done
    assert reward == -0.1
    assert env.row_pos == 3
    assert env.col_pos == 2

    obs, reward, done, _ = env.step(2)
    assert env.row_pos == 3
    assert env.col_pos == 1

    obs, reward, done, _ = env.step(0)
    assert env.row_pos == 2
    assert env.col_pos == 1

    obs, reward, done, _ = env.step(0)
    assert reward == 10
    assert obs[-1] == 10
    assert obs[-2] == 5
    assert obs[-3] == 1


def test_no_extra():
    env = MetaMazeEnv(extra_inputs=False)
    obs = env.reset()
    assert (obs == [1, 0, 1, 0, 0, 0, 1, 0, 1, 0.0]).all()
    env.reward_row_pos = env.reward_col_pos = 1
    assert env.row_pos == env.col_pos == 3

    obs, reward, done, _ = env.step(2)
    assert (obs == [0, 1, 0, 0, 0, 0, 0, 1, 0, 0.0]).all()
    assert not done
    assert reward == 0
    assert env.row_pos == 3
    assert env.col_pos == 2

    obs, reward, done, _ = env.step(1)
    assert (obs == [0, 1, 0, 0, 0, 0, 0, 1, 0, -0.1]).all()
    assert not done
    assert reward == -0.1
    assert env.row_pos == 3
    assert env.col_pos == 2

    obs, reward, done, _ = env.step(2)
    assert env.row_pos == 3
    assert env.col_pos == 1

    obs, reward, done, _ = env.step(0)
    assert env.row_pos == 2
    assert env.col_pos == 1

    obs, reward, done, _ = env.step(0)
    assert reward == 10
    assert obs[-1] == 10
