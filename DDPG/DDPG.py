import random
from collections import deque
import time

import gym
import numpy as np
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import loss as gloss
import gluonbook as gb

def f():
    pass
# noise
class OrnsteinUhlenbeck:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.05):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = nd.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = nd.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * nd.random.randn(len(self.X))
        self.X += dx
        return self.X

def f():
    pass
# parameters soft update
def soft_update(target_network, main_network, tau):
    value1 = target_network.collect_params().keys()
    value2 = main_network.collect_params().keys()
    d = zip(value1, value2)
    for x, y in d:
        target_network.collect_params()[x].data()[:] = target_network.collect_params()[x].data() * (1 - tau) + \
                                                       main_network.collect_params()[y].data() * tau


def smooth_function(l, n):
    m = []
    split_list = np.array_split(l, n)
    for i in split_list:
        m.append(np.mean(i))
    return m


class ActorNetwork(gluon.nn.Block):
    def __init__(self, action_dim, action_bound):
        super(ActorNetwork, self).__init__()
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.dense0 = gluon.nn.Dense(64, activation='relu')
        self.dense1 = gluon.nn.Dense(32, activation='relu')
        # use tanh to get action [-1, 1]
        self.dense2 = gluon.nn.Dense(action_dim, activation='tanh')

    def forward(self, state):
        action = self.dense2(self.dense1(self.dense0(state)))
        # scale
        action = action * self.action_bound
        return action


# Input (s, a), output q
class CriticNetwork(gluon.nn.Block):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.dense0 = gluon.nn.Dense(64, activation='relu')
        self.dense1 = gluon.nn.Dense(32, activation='relu')
        self.dense2 = gluon.nn.Dense(1)

    def forward(self, state, action):
        feature = nd.concat(state, action, dim=1)
        value = self.dense2(self.dense1(self.dense0(feature)))
        return value


class MemoryBuffer:
    def __init__(self, buffer_size, ctx):
        self.buffer = deque(maxlen=buffer_size)
        self.maxsize = buffer_size
        self.ctx = ctx

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        assert len(self.buffer) > batch_size
        minibatch = random.sample(self.buffer, batch_size)
        state_batch = nd.array([data[0] for data in minibatch], ctx=self.ctx)
        action_batch = nd.array([data[1] for data in minibatch], ctx=self.ctx)
        reward_batch = nd.array([data[2] for data in minibatch], ctx=self.ctx)
        next_state_batch = nd.array([data[3] for data in minibatch], ctx=self.ctx)
        return state_batch, action_batch, reward_batch, next_state_batch

    def store_transition(self, state, action, reward, next_state):
        transition = (state, action, reward, next_state)
        self.buffer.append(transition)


class DDPG_Agent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound,
                 actor_learning_rate,
                 critic_learning_rate,
                 batch_size,
                 memory_size,
                 gamma,
                 ctx,
                 replace_iter=None,   # for hard update
                 tau=0.01,
                 ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.replace_iter = replace_iter   # for hard update
        self.steps = 0
        self.tau = tau   # for soft update
        self.ctx = ctx

        self.noise = OrnsteinUhlenbeck(self.action_dim)

        self.memory_buffer = MemoryBuffer(self.memory_size, ctx=ctx)

        self.target_actor_network = ActorNetwork(self.action_dim, self.action_bound)
        self.main_actor_network = ActorNetwork(self.action_dim, self.action_bound)
        self.target_critic_network = CriticNetwork()
        self.main_critic_network = CriticNetwork()

        self.target_actor_network.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.target_critic_network.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.main_actor_network.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.main_critic_network.collect_params().initialize(init=init.Xavier(), ctx=ctx)

        self.actor_optimizer = gluon.Trainer(self.main_actor_network.collect_params(),
                                             'adam',
                                             {'learning_rate': self.actor_learning_rate})
        self.critic_optimizer = gluon.Trainer(self.main_critic_network.collect_params(),
                                              'adam',
                                              {'learning_rate': self.critic_learning_rate})

    def choose_action_explore(self, state):
        state = nd.array([state], ctx=self.ctx)
        explore_action = self.main_actor_network(state).squeeze()
        # add noise
        explore_action += nd.array(self.noise.sample() * self.action_bound, ctx=self.ctx)
        return explore_action

    def choose_action_greedily(self, state):
        state = nd.array([state], ctx=self.ctx)
        # no noise
        greedy_action = self.target_actor_network(state).squeeze().asnumpy()
        return greedy_action

    def optimize(self):
        state_batch, action_batch, reward_batch, next_state_batch = self.memory_buffer.sample(self.batch_size)

        # ---------------optimize critic------------------
        with autograd.record():
            next_action_batch = self.target_actor_network(next_state_batch).detach()
            next_q = self.target_critic_network(next_state_batch, next_action_batch).detach().squeeze()
            target_q = reward_batch + self.gamma * next_q

            eval_q = self.main_critic_network(state_batch, action_batch)
            loss = gloss.L2Loss()
            value_loss = loss(target_q, eval_q)
        self.main_critic_network.collect_params().zero_grad()
        value_loss.backward()
        self.critic_optimizer.step(self.batch_size)

        # ---------------optimize actor-------------------
        with autograd.record():
            pred_action_batch = self.main_actor_network(state_batch)
            actor_loss = -1 * nd.mean(self.main_critic_network(state_batch, pred_action_batch))
        self.main_actor_network.collect_params().zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step(self.batch_size)

    def network_hard_update(self):
        self.main_actor_network.save_parameters('main_actor_network')
        self.target_actor_network.load_parameters('main_actor_network')
        self.main_critic_network.save_parameters('main_critic_network')
        self.target_critic_network.load_parameters('main_critic_network')
        print('params hard replaced')

    def network_soft_update(self):
        soft_update(self.target_actor_network, self.main_actor_network, self.tau)
        soft_update(self.target_critic_network, self.main_critic_network, self.tau)

    def save(self):
        self.main_actor_network.save_parameters('DDPG Main Actor.params')
        self.target_actor_network.save_parameters('DDPG Target Actor.params')
        self.main_critic_network.save_parameters('DDPG Main Critic.params')
        self.target_critic_network.save_parameters('DDPG Target Critic.params')

    def load(self):
        self.main_actor_network.load_parameters('DDPG Main Actor.params')
        self.target_actor_network.load_parameters('DDPG Target Actor.params')
        self.main_critic_network.load_parameters('DDPG Main Critic.params')
        self.target_critic_network.load_parameters('DDPG Target Critic.params')


env = gym.make('Pendulum-v0').unwrapped
seed = 1
env.seed(1)
np.random.seed(1)
mx.random.seed(1)
ctx= gb.try_gpu()

action_dim = env.action_space.shape[0]         # 1
action_bound = env.action_space.high[0]        # 2
state_dim = env.observation_space.shape[0]        # 3

MAX_episodes = 100
MAX_episode_steps = 1000


agent = DDPG_Agent(state_dim,
                   action_dim,
                   action_bound,
                   actor_learning_rate=0.0001,
                   critic_learning_rate=0.001,
                   batch_size=32,
                   memory_size=100000,
                   gamma=0.99,
                   ctx=ctx,
                   replace_iter=3000,   # for hard update if needed
                   tau=0.01,
                   )


# you can get SOTA performance if you load the parameters
# agent.load()
episode_reward_list = []
for episode in range(MAX_episodes):
    episode_reward = 0
    state = env.reset()
    for step in range(MAX_episode_steps):
        # get SOTA results at about 40th episode using this parameters
        if episode > 40:
            env.render()
        action = agent.choose_action_explore(state)

        next_state, reward, done, info = env.step(action.asnumpy())
        agent.memory_buffer.store_transition(state, action.asnumpy(), reward, next_state)
        agent.steps += 1
        episode_reward += reward
        if len(agent.memory_buffer) > 1000:
            agent.optimize()
            agent.network_soft_update()
        if done:
            break

        state = next_state
    print('episode %d ends with rewards %f ' % (episode, episode_reward))
    episode_reward_list.append(episode_reward)
env.close()
agent.save()

plt.plot(episode_reward_list)
plt.xlabel('episode')
plt.ylabel('reward')
plt.title('DDPG Pendulum-v0')
plt.savefig('./DDPG Pendulum-v0')
plt.show()
