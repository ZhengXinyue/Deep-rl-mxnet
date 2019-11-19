import random
from collections import deque
import time

import gym
import numpy as np
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import loss as gloss, nn
import gluonbook as gb


def smooth_function(l, n):
    m = []
    split_list = np.array_split(l, n)
    for i in split_list:
        m.append(np.mean(i))
    return m


class ActorNetwork(nn.Block):
    def __init__(self, action_dim, action_bound):
        super(ActorNetwork, self).__init__()
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.dense0 = nn.Dense(400, activation='relu')
        self.dense1 = nn.Dense(300, activation='relu')
        self.dense2 = nn.Dense(self.action_dim, activation='tanh')

    def forward(self, state):
        action = self.dense2(self.dense1(self.dense0(state)))
        upper_bound = self.action_bound[:, 1]    # scale
        action = action * upper_bound
        return action


class CriticNetwork(nn.Block):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.dense0 = nn.Dense(400, activation='relu')
        self.dense1 = nn.Dense(300, activation='relu')
        self.dense2 = nn.Dense(1)

    def forward(self, state, action):
        input = nd.concat(state, action, dim=1)
        q_value = self.dense2(self.dense1(self.dense0(input)))
        return q_value


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
        done = nd.array([data[4] for data in minibatch], ctx=self.ctx)
        return state_batch, action_batch, reward_batch, next_state_batch, done

    def store_transition(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)


class DDPG_Agent:
    def __init__(self,
                 action_dim,
                 action_bound,
                 actor_learning_rate,
                 critic_learning_rate,
                 batch_size,
                 memory_size,
                 gamma,
                 tau,
                 explore_steps,
                 explore_noise,
                 noise_clip,
                 ctx
                 ):
        self.action_dim = action_dim
        self.action_bound = nd.array(action_bound, ctx=ctx)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.tau = tau
        self.explore_steps = explore_steps
        self.explore_noise = explore_noise
        self.noise_clip = noise_clip
        self.ctx = ctx
        self.total_steps = 0

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

    def choose_action_train(self, state):
        state = nd.array([state], ctx=self.ctx)
        action = self.main_actor_network(state)
        # no noise clip
        noise = nd.normal(loc=0, scale=self.explore_noise, shape=action.shape, ctx=self.ctx)
        action += noise
        clipped_action = self.action_clip(action).squeeze()
        return clipped_action

    def choose_action_evaluate(self, state):
        state = nd.array([state], ctx=self.ctx)
        action = self.main_actor_network(state)
        return action

    def action_clip(self, action):
        if len(action[0]) == 2:
            action0 = nd.clip(action[:, 0], float(self.action_bound[0][0].asnumpy()),
                              float(self.action_bound[0][1].asnumpy()))
            action1 = nd.clip(action[:, 1], float(self.action_bound[1][0].asnumpy()),
                              float(self.action_bound[1][1].asnumpy()))
            clipped_action = nd.concat(action0.reshape(-1, 1), action1.reshape(-1, 1))
        else:
            clipped_action = nd.clip(action, float(self.action_bound[0][0].asnumpy()),
                                     float(self.action_bound[0][1].asnumpy()))
        return clipped_action

    def soft_update(self, target_network, main_network):
        target_parameters = target_network.collect_params().keys()
        main_parameters = main_network.collect_params().keys()
        d = zip(target_parameters, main_parameters)
        for x, y in d:
            target_network.collect_params()[x].data()[:] = \
                target_network.collect_params()[x].data() * \
                (1 - self.tau) + main_network.collect_params()[y].data() * self.tau

    def update(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch= self.memory_buffer.sample(self.batch_size)

        # ---------------optimize critic------------------
        with autograd.record():
            next_action_batch = self.target_actor_network(next_state_batch)
            next_q = self.target_critic_network(next_state_batch, next_action_batch).squeeze()
            target_q = reward_batch + (1 - done_batch) * self.gamma * next_q

            current_q = self.main_critic_network(state_batch, action_batch)
            loss = gloss.L2Loss()
            value_loss = loss(target_q.detach(), current_q)
        self.main_critic_network.collect_params().zero_grad()
        value_loss.backward()
        self.critic_optimizer.step(self.batch_size)

        # ---------------optimize actor-------------------
        with autograd.record():
            pred_action_batch = self.main_actor_network(state_batch)
            actor_loss = -nd.mean(self.main_critic_network(state_batch, pred_action_batch))
        self.main_actor_network.collect_params().zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step(1)

        self.soft_update(self.target_actor_network, self.main_actor_network)
        self.soft_update(self.target_critic_network, self.main_critic_network)

    def save(self):
        self.main_actor_network.save_parameters('DDPG LunarLander Main Actor.params')
        self.target_actor_network.save_parameters('DDPG LunarLander Target Actor.params')
        self.main_critic_network.save_parameters('DDPG LunarLander Main Critic.params')
        self.target_critic_network.save_parameters('DDPG LunarLander Target Critic.params')

    def load(self):
        self.main_actor_network.load_parameters('DDPG LunarLander Main Actor.params')
        self.target_actor_network.load_parameters('DDPG LunarLander Target Actor.params')
        self.main_critic_network.load_parameters('DDPG LunarLander  Main Critic.params')
        self.target_critic_network.load_parameters('DDPG LunarLander Target Critic.params')


def main():
    env = gym.make('LunarLanderContinuous-v2').unwrapped
    seed = 1
    env.seed(1)
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    ctx = gb.try_gpu()
    ctx = mx.cpu()
    max_episodes = 2000
    max_episode_steps = 2000   # this doesn't matter, because this env itself has max episode steps(1000) constraint.
    env_action_bound = [[-1, 1], [-1, 1]]

    agent = DDPG_Agent(action_dim=int(env.action_space.shape[0]),
                       action_bound=env_action_bound,
                       actor_learning_rate=0.0001,
                       critic_learning_rate=0.001,
                       batch_size=64,
                       memory_size=100000,
                       gamma=0.99,
                       tau=0.001,
                       explore_steps=1000,
                       explore_noise=0.1,
                       noise_clip=0.5,
                       ctx=ctx)

    episode_reward_list = []
    mode = input("train or test: ")

    if mode == 'train':
        render = False
        for episode in range(max_episodes):
            episode_reward = 0
            state = env.reset()
            for step in range(max_episode_steps):
                if render:
                    env.render()
                if agent.total_steps < agent.explore_steps:
                    action = env.action_space.sample()
                    agent.total_steps += 1
                else:
                    action = agent.choose_action_train(state)
                    action = action.asnumpy()
                    print(action)
                    agent.total_steps += 1
                next_state, reward, done, info = env.step(action)
                agent.memory_buffer.store_transition(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                if agent.total_steps >= agent.explore_steps:
                    agent.update()
                if done:
                    break
            print('episode %d ends with reward %f ' % (episode, episode_reward))
            episode_reward_list.append(episode_reward)
        agent.save()

    elif mode == 'test':
        render = True
        agent.load()
        for episode in range(max_episodes):
            episode_reward = 0
            state = env.reset()
            for step in range(max_episode_steps):
                if render:
                    env.render()
                action = agent.choose_action_train(state)
                action = action.asnumpy()
                next_state, reward, done, info = env.step(action)
                agent.memory_buffer.store_transition(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                if done:
                    break
            print('episode  %d  ends with reward  %f  total steps:  %d' % (episode, episode_reward, agent.total_steps))
            episode_reward_list.append(episode_reward)
    else:
        raise NameError('Wrong input')

    env.close()
    plt.plot(episode_reward_list)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('DDPG LunarLanderContinuous-v2')
    if mode == 'train':
        plt.savefig('./LunarLanderContinuous_v2')
    plt.show()


if __name__ == '__main__':
    main()

