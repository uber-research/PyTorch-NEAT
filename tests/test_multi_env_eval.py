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

from pytorch_neat.multi_env_eval import MultiEnvEvaluator


class DummyEnv:
    def __init__(self, ep_len=2, reward_mag=1):
        self.ep_len = ep_len
        self.reward_mag = reward_mag
        self.reset()

    def step(self, action):
        self.step_num += 1
        if action == 0:
            reward = self.reward_mag
        else:
            reward = -self.reward_mag
        return self.step_num, reward, self.step_num == self.ep_len, {}

    def reset(self):
        self.step_num = 0
        return self.step_num


class EndlessEnv:
    def step(self, _action):
        assert self.step_num < 10
        self.step_num += 1
        return 0, 0, False, {}

    def reset(self):
        self.step_num = 0
        return 0


class DummyNet:
    def __init__(self, actions):
        self.actions = actions

    def activate(self, states):
        return [actions[state] for actions, state in zip(self.actions, states)]


env_num = 1


def make_env():
    global env_num
    env = DummyEnv(1 + env_num, env_num)
    env_num += 1
    return env


def make_endless_env():
    return EndlessEnv()


def make_net(_genome, _config, _batch_size):
    return DummyNet(
        [
            [0, 0],  # r=2*1
            [0, 1, 0],  # r=1*2
            [1, 0, 0, 0],  # r=2*3
            [1, 1, 1, 0, 1],  # r=-3*4
        ]
    )


def activate_net(net, states):
    return net.activate(states)


def test_multi():
    evaluator = MultiEnvEvaluator(
        make_net, activate_net, batch_size=4, make_env=make_env
    )
    returns = evaluator.eval_genome(None, None)
    assert returns == (2 + 2 + 6 - 12) / 4


def test_endless():
    evaluator = MultiEnvEvaluator(
        make_net,
        activate_net,
        batch_size=4,
        make_env=make_endless_env,
        max_env_steps=10,
    )
    evaluator.eval_genome(None, None)
