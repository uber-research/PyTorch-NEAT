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

from pytorch_neat.t_maze import TMazeEnv


def test_default_initialization():
    env = TMazeEnv()
    assert env.hall_len == 3
    assert env.n_trials == 100
    assert env.maze.shape == (6, 9)
    print(env.maze)
    assert (
        env.maze
        == [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ).all()


def test_step_without_reset():
    env = TMazeEnv()
    with pytest.raises(AssertionError):
        env.step(1)


def test_render():
    env = TMazeEnv()
    with pytest.raises(NotImplementedError):
        env.render()


def test_step_with_reset():
    env = TMazeEnv()
    obs = env.reset()
    assert obs.shape == (4,)
    assert env.row_pos == env.col_pos == 4
    assert (obs == [1, 0, 1, 0]).all()
    obs, reward, done, _ = env.step(0)
    assert (obs == [1, 0, 1, 0]).all()
    assert reward == -0.4
    assert not done


def test_full_trial():
    env = TMazeEnv()
    obs = env.reset()
    for _ in range(3):
        assert (obs == [1, 0, 1, 0]).all()
        obs, reward, done, _ = env.step(1)
        assert not done
        assert reward == 0
    for _ in range(3):
        assert (obs == [0, 1, 0, 0]).all()
        assert reward == 0
        obs, reward, done, _ = env.step(2)
        assert not done
    assert (obs == [0, 1, 1, 1]).all()
    assert reward == 1
    obs, reward, done, _ = env.step(2)
    assert reward == 0
    assert (obs == [1, 0, 1, 0]).all()
    assert env.row_pos == env.col_pos == 4
    assert not done


def test_init_reward_side():
    env = TMazeEnv(init_reward_side=0)
    obs = env.reset()
    for _ in range(3):
        assert (obs == [1, 0, 1, 0]).all()
        obs, reward, done, _ = env.step(1)
        assert not done
        assert reward == 0
    for _ in range(3):
        assert (obs == [0, 1, 0, 0]).all()
        assert reward == 0
        obs, reward, done, _ = env.step(0)
        assert not done
    assert (obs == [1, 1, 0, 1]).all()
    assert reward == 1
    obs, reward, done, _ = env.step(1)
    assert reward == 0
    assert (obs == [1, 0, 1, 0]).all()
    assert env.row_pos == env.col_pos == 4
    assert not done


def test_low_reward():
    env = TMazeEnv()
    obs = env.reset()
    for _ in range(3):
        assert (obs == [1, 0, 1, 0]).all()
        obs, reward, done, _ = env.step(1)
        assert not done
        assert reward == 0
    for _ in range(3):
        assert (obs == [0, 1, 0, 0]).all()
        assert reward == 0
        obs, reward, done, _ = env.step(0)
        assert not done
    assert (obs == [1, 1, 0, 0.2]).all()
    assert reward == 0.2
    obs, reward, done, _ = env.step(1)
    assert reward == 0
    assert (obs == [1, 0, 1, 0]).all()
    assert env.row_pos == env.col_pos == 4
    assert not done


def test_deployment():
    env = TMazeEnv(n_trials=3)
    for _ in range(3):
        obs = env.reset()
        for _ in range(3):
            for _ in range(3):
                assert (obs == [1, 0, 1, 0]).all()
                obs, reward, done, _ = env.step(1)
                assert not done
                assert reward == 0
            for _ in range(3):
                assert (obs == [0, 1, 0, 0]).all()
                assert reward == 0
                obs, reward, done, _ = env.step(2)
                assert not done
            assert (obs == [0, 1, 1, 1]).all()
            assert reward == 1
            obs, reward, done, _ = env.step(2)
            assert reward == 0
            assert (obs == [1, 0, 1, 0]).all()
            assert env.row_pos == env.col_pos == 4
        assert done


def test_reward_flip():
    env = TMazeEnv(n_trials=10, reward_flip_mean=5, reward_flip_range=3)
    for _ in range(10):
        obs = env.reset()
        for i in range(10):
            for _ in range(3):
                assert (obs == [1, 0, 1, 0]).all()
                obs, reward, done, _ = env.step(1)
                assert not done
                assert reward == 0
            for _ in range(3):
                assert (obs == [0, 1, 0, 0]).all()
                assert reward == 0
                obs, reward, done, _ = env.step(2)
                assert not done
            assert (obs[:-1] == [0, 1, 1]).all()
            assert reward == obs[-1]
            assert reward in {0.2, 1.0}
            if i < 2:
                assert reward == 1.0
            elif i > 8:
                assert reward == 0.2
            obs, reward, done, _ = env.step(2)
            assert reward == 0
            assert (obs == [1, 0, 1, 0]).all()
            assert env.row_pos == env.col_pos == 4
        assert done
