
from collections import namedtuple

import random
import time
import time

import numpy as np
import matplotlib.pyplot as plt
import gym

import mxnet as mx
from mxnet import autograd, nd, init, gluon
from mxnet.gluon import loss as gloss, nn
import gluonbook as gb

time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


class Actor(nn.Block):
    def __init__(self, n_action, **kwargs):
        super(Actor, self).__init__(**kwargs)
        self.n_action = n_action
        self.dense0 = nn.Dense(64, activation='relu')
        self.dense1 = nn.Dense(n_action)

    def forward(self, x):
        x = self.dense0(x)
        action_prob = nd.softmax(self.dense1(x), axis=1)
        return action_prob


class Critic(nn.Block):
    def __init__(self, **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.dense0 = nn.Dense(64, activation='relu')
        self.dense1 = nn.Dense(1)

    def forward(self, x):
        x = self.dense0(x)
        value = self.dense1(x)
        return value


class PPO:
    def __init__(self,
                 n_action,
                 clip_param,
                 max_grad_norm,
                 ppo_update_time,
                 buffer_capacity,
                 batch_size,
                 gamma,
                 actor_learning_rate,
                 critic_learning_rate,
                 ctx):
        self.n_action = n_action
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.ppo_update_time = ppo_update_time
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.ctx = ctx

        self.actor_network = Actor(self.n_action)
        self.actor_network.initialize(ctx=self.ctx)
        self.critic_network = Critic()
        self.critic_network.initialize(ctx=self.ctx)

        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = gluon.Trainer(self.actor_network.collect_params(),
                                             'adam', {'learning_rate': self.actor_learning_rate})
        self.critic_optimizer = gluon.Trainer(self.critic_network.collect_params(),
                                             'adam', {'learning_rate': self.critic_learning_rate})

    def choose_action(self, state):
        state = nd.array([state], ctx=self.ctx)
        all_action_prob = self.actor_network(state)
        action = nd.sample_multinomial(all_action_prob)
        action_prob = nd.pick(all_action_prob, action, axis=1).asnumpy()
        action = int(action.asnumpy())
        return action, action_prob

    def get_value(self, state):
        state = nd.array([state], ctx=self.ctx)
        value = self.critic_network(state)
        return value

    def save_parameters(self):
        self.actor_network.save_parameters('PPO discrete actor parameters')
        self.critic_network.save_parameters('PPO discrete critic parameters')

    def load_parameters(self):
        self.actor_network.load_parameters('PPO discrete actor parameters')
        self.critic_network.load_parameters('PPO discrete critic parameters')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self):
        state = nd.array([t.state for t in self.buffer], ctx=self.ctx)
        action = nd.array([t.action for t in self.buffer], ctx=self.ctx)
        reward = [t.reward for t in self.buffer]
        # next_state = nd.array([t.next_state for t in self.buffer], ctx=self.ctx)
        old_action_log_prob = nd.array([t.a_log_prob for t in self.buffer], ctx=self.ctx)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = nd.array(Gt, ctx=self.ctx)
        # sample 'ppo_update_time' times
        # sample 'batch_size' samples every time
        for i in range(self.ppo_update_time):
            assert len(self.buffer) >= self.batch_size
            sample_index = random.sample(range(len(self.buffer)), self.batch_size)
            for index in sample_index:

                # optimize the actor network
                with autograd.record():
                    Gt_index = Gt[index]
                    V = self.critic_network(state[index].reshape(1, -1)).detach()
                    advantage = (Gt_index - V)

                    all_action_prob = self.actor_network(state[index].reshape(1, -1))
                    action_prob = nd.pick(all_action_prob, action[index])

                    ratio = action_prob / old_action_log_prob[index]
                    surr1 = ratio * advantage
                    surr2 = nd.clip(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                    action_loss = -nd.mean(nd.minimum(surr1, surr2))      # attention
                self.actor_network.collect_params().zero_grad()
                action_loss.backward()
                actor_network_params = [p.data() for p in self.actor_network.collect_params().values()]
                gb.grad_clipping(actor_network_params, theta=self.clip_param, ctx=self.ctx)
                self.actor_optimizer.step(1)

                # optimize the critic network
                with autograd.record():
                    Gt_index = Gt[index]
                    V = self.critic_network(state[index].reshape(1, -1))
                    loss = gloss.L2Loss()
                    value_loss = nd.mean(loss(Gt_index, V))
                self.critic_network.collect_params().zero_grad()
                value_loss.backward()
                critic_network_params = [p.data() for p in self.critic_network.collect_params().values()]
                gb.grad_clipping(critic_network_params, theta=self.clip_param, ctx=self.ctx)
                self.critic_optimizer.step(1)

                self.training_step += 1
        # clear buffer
        del self.buffer[:]


env = gym.make('CartPole-v0').unwrapped
seed = 1
env.seed(1)
mx.random.seed(seed)
np.random.seed(seed)
random.seed(seed)
ctx = gb.try_gpu()

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
num_state = env.observation_space.shape[0]
num_acion = env.action_space.n


def main():
    render = False
    episode_reward_list = []
    agent = PPO(n_action=num_acion,
                clip_param=0.2,
                max_grad_norm=0.5,
                ppo_update_time=10,
                buffer_capacity=1000,
                batch_size=32,
                gamma=0.99,
                actor_learning_rate=0.001,
                critic_learning_rate=0.003,
                ctx=ctx)
    # agent.load()
    for episode in range(200):
        state = env.reset()
        while True:
            if render:
                env.render()
            action, acion_prob = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            trans = Transition(state, action, acion_prob, reward, next_state)
            agent.store_transition(trans)
            if done:
                episode_reward = sum([t.reward for t in agent.buffer])
                print('episode %d  reward  %d' % (episode, episode_reward))
                episode_reward_list.append(episode_reward)
                mean_reward = sum(episode_reward_list[-100:]) / 100
                if mean_reward > 195:
                    render = False
                if len(agent.buffer) >= agent.batch_size:
                    agent.update()
                break
            state = next_state
    agent.save_parameters()
    env.close()
    
    plt.plot(episode_reward_list)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('PPO CartPole-v0')
    plt.show()
    plt.savefig('./PPO CartPole-v0')


if __name__ == '__main__':
    main()



































