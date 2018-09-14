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

from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.strict_t_maze import StrictTMazeEnv


def test_default_initialization():
    env = StrictTMazeEnv()
    assert env.hall_len == 3
    assert env.n_trials == 100
    assert env.maze.shape == (6, 9)
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
    env = StrictTMazeEnv()
    with pytest.raises(AssertionError):
        env.step(1)


def test_render():
    env = StrictTMazeEnv()
    with pytest.raises(NotImplementedError):
        env.render()


def test_step_with_reset():
    env = StrictTMazeEnv()
    obs = env.reset()
    assert obs.shape == (4,)
    assert env.row_pos == env.col_pos == 4
    assert (obs == [1, 0, 1, 0]).all()

    obs, reward, done, _ = env.step(0)
    assert (obs == [1, 1, 0, 0]).all()
    assert reward == -0.4
    assert not done

    obs, reward, done, _ = env.step(0)
    assert (obs == [1, 0, 1, 0]).all()
    assert reward == 0.0
    assert not done


def test_full_trial():
    env = StrictTMazeEnv()
    obs = env.reset()
    for _ in range(3):
        assert (obs == [1, 0, 1, 0]).all()
        assert env.direction == 0
        obs, reward, done, _ = env.step(1)
        assert not done
        assert reward == 0
    assert (obs == [0, 1, 0, 0]).all()
    assert env.direction == 0
    assert reward == 0
    obs, reward, done, _ = env.step(2)
    assert env.direction == 1
    assert (obs == [1, 0, 0, 0]).all()
    assert reward == 0
    assert not done
    for _ in range(2):
        obs, reward, done, _ = env.step(1)
        assert env.direction == 1
        assert (obs == [1, 0, 1, 0]).all()
        assert reward == 0
        assert not done
    obs, reward, done, _ = env.step(1)
    assert (obs == [1, 1, 1, 1]).all()
    assert reward == 1
    assert env.direction == 1
    assert not done
    obs, reward, done, _ = env.step(2)
    assert reward == 0
    assert (obs == [1, 0, 1, 0]).all()
    assert env.direction == 0
    assert env.row_pos == env.col_pos == 4
    assert not done


def test_repeat_turn_penalty():
    env = StrictTMazeEnv()
    obs = env.reset()
    for _ in range(3):
        assert (obs == [1, 0, 1, 0]).all()
        assert env.direction == 0
        obs, reward, done, _ = env.step(1)
        assert not done
        assert reward == 0
    assert (obs == [0, 1, 0, 0]).all()
    assert env.direction == 0
    assert reward == 0
    obs, reward, done, _ = env.step(2)
    assert env.direction == 1
    assert (obs == [1, 0, 0, 0]).all()
    assert reward == 0
    assert not done
    obs, reward, done, _ = env.step(2)
    assert env.direction == 2
    assert (obs == [0, 0, 0, 0]).all()
    assert reward == -0.4
    assert not done
    obs, reward, done, _ = env.step(1)
    assert (obs == [1, 0, 1, 0]).all()
    assert env.row_pos == env.col_pos == 4
    assert env.direction == 0


def test_cross_turn_penalty():
    env = StrictTMazeEnv()
    obs = env.reset()
    for _ in range(3):
        assert (obs == [1, 0, 1, 0]).all()
        assert env.direction == 0
        obs, reward, done, _ = env.step(1)
        assert not done
        assert reward == 0
    assert (obs == [0, 1, 0, 0]).all()
    assert env.direction == 0
    assert reward == 0
    obs, reward, done, _ = env.step(2)
    assert env.direction == 1
    assert (obs == [1, 0, 0, 0]).all()
    assert reward == 0
    assert not done
    obs, reward, done, _ = env.step(1)
    assert (obs == [1, 0, 1, 0]).all()
    assert env.direction == 1
    assert reward == 0
    assert not done
    obs, reward, done, _ = env.step(2)
    assert env.direction == 2
    assert (obs == [0, 1, 0, 0]).all()
    assert reward == -0.4
    assert not done
    obs, reward, done, _ = env.step(1)
    assert (obs == [1, 0, 1, 0]).all()
    assert env.row_pos == env.col_pos == 4
    assert env.direction == 0


def test_init_reward_side():
    env = StrictTMazeEnv(init_reward_side=0)
    obs = env.reset()
    for _ in range(3):
        assert (obs == [1, 0, 1, 0]).all()
        assert env.direction == 0
        obs, reward, done, _ = env.step(1)
        assert not done
        assert reward == 0
    assert (obs == [0, 1, 0, 0]).all()
    assert env.direction == 0
    assert reward == 0
    obs, reward, done, _ = env.step(0)
    assert env.direction == 3
    assert (obs == [0, 0, 1, 0]).all()
    assert reward == 0
    assert not done
    for _ in range(2):
        obs, reward, done, _ = env.step(1)
        assert env.direction == 3
        assert (obs == [1, 0, 1, 0]).all()
        assert reward == 0
        assert not done
    obs, reward, done, _ = env.step(1)
    assert (obs == [1, 1, 1, 1]).all()
    assert reward == 1
    assert env.direction == 3
    assert not done
    obs, reward, done, _ = env.step(2)
    assert reward == 0
    assert (obs == [1, 0, 1, 0]).all()
    assert env.direction == 0
    assert env.row_pos == env.col_pos == 4
    assert not done


def test_low_reward():
    env = StrictTMazeEnv()
    obs = env.reset()
    for _ in range(3):
        assert (obs == [1, 0, 1, 0]).all()
        assert env.direction == 0
        obs, reward, done, _ = env.step(1)
        assert not done
        assert reward == 0
    assert (obs == [0, 1, 0, 0]).all()
    assert env.direction == 0
    assert reward == 0
    obs, reward, done, _ = env.step(0)
    assert env.direction == 3
    assert (obs == [0, 0, 1, 0]).all()
    assert reward == 0
    assert not done
    for _ in range(2):
        obs, reward, done, _ = env.step(1)
        assert env.direction == 3
        assert (obs == [1, 0, 1, 0]).all()
        assert reward == 0
        assert not done
    obs, reward, done, _ = env.step(1)
    assert (obs == [1, 1, 1, 0.2]).all()
    assert reward == 0.2
    assert env.direction == 3
    assert not done
    obs, reward, done, _ = env.step(2)
    assert reward == 0
    assert (obs == [1, 0, 1, 0]).all()
    assert env.direction == 0
    assert env.row_pos == env.col_pos == 4
    assert not done


def test_deployment():
    env = StrictTMazeEnv(n_trials=3)
    for _ in range(5):
        obs = env.reset()
        for _ in range(3):
            for _ in range(3):
                assert (obs == [1, 0, 1, 0]).all()
                assert env.direction == 0
                obs, reward, done, _ = env.step(1)
                assert not done
                assert reward == 0
            assert (obs == [0, 1, 0, 0]).all()
            assert env.direction == 0
            assert reward == 0
            obs, reward, done, _ = env.step(2)
            assert env.direction == 1
            assert (obs == [1, 0, 0, 0]).all()
            assert reward == 0
            assert not done
            for _ in range(2):
                obs, reward, done, _ = env.step(1)
                assert env.direction == 1
                assert (obs == [1, 0, 1, 0]).all()
                assert reward == 0
                assert not done
            obs, reward, done, _ = env.step(1)
            assert (obs == [1, 1, 1, 1]).all()
            assert reward == 1
            assert env.direction == 1
            assert not done
            obs, reward, done, _ = env.step(2)
            assert reward == 0
            assert (obs == [1, 0, 1, 0]).all()
            assert env.direction == 0
            assert env.row_pos == env.col_pos == 4
        assert done


def test_reward_flip():
    env = StrictTMazeEnv(n_trials=10, reward_flip_mean=5, reward_flip_range=3)
    for _ in range(5):
        obs = env.reset()
        for i in range(10):
            for _ in range(3):
                assert (obs == [1, 0, 1, 0]).all()
                assert env.direction == 0
                obs, reward, done, _ = env.step(1)
                assert not done
                assert reward == 0
            assert (obs == [0, 1, 0, 0]).all()
            assert env.direction == 0
            assert reward == 0
            obs, reward, done, _ = env.step(2)
            assert env.direction == 1
            assert (obs == [1, 0, 0, 0]).all()
            assert reward == 0
            assert not done
            for _ in range(2):
                obs, reward, done, _ = env.step(1)
                assert env.direction == 1
                assert (obs == [1, 0, 1, 0]).all()
                assert reward == 0
                assert not done
            obs, reward, done, _ = env.step(1)
            assert (obs[:-1] == [1, 1, 1]).all()
            assert reward == obs[-1]
            assert reward in {0.2, 1.0}
            if i < 2:
                assert reward == 1.0
            elif i > 8:
                assert reward == 0.2
            assert env.direction == 1
            assert not done
            obs, reward, done, _ = env.step(2)
            assert reward == 0
            assert (obs == [1, 0, 1, 0]).all()
            assert env.direction == 0
            assert env.row_pos == env.col_pos == 4
        assert done


class OptimalNet:
    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.sides = [0] * n_envs

    def act(self, states):
        actions = []
        for i, state in enumerate(states):
            if all(state == [1, 0, 1, 0]):
                actions.append(1)
            elif all(state == [0, 1, 0, 0]):
                actions.append(0 if self.sides[i] == 0 else 2)
            elif all(state == [1, 0, 0, 0]):
                actions.append(1)
            elif all(state == [0, 0, 1, 0]):
                actions.append(1)
            elif all(state[:-1] == [1, 1, 1]):
                actions.append(1)
                assert state[-1] in {0.2, 1.0}
                if state[-1] == 0.2:
                    self.sides[i] = 1 - self.sides[i]
            else:
                raise ValueError("Invalid state")
        return actions


def make_net(_genome, _config, n_envs):
    return OptimalNet(n_envs)


def activate_net(net, states):
    return net.act(states)


def test_optimal():
    envs = [StrictTMazeEnv(init_reward_side=i, n_trials=100) for i in [1, 0, 1, 0]]

    evaluator = MultiEnvEvaluator(
        make_net, activate_net, envs=envs, batch_size=4, max_env_steps=1600
    )

    fitness = evaluator.eval_genome(None, None)
    assert fitness == 98.8
