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


class Actor(nn.Block):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.dense0 = nn.Dense(400, activation='relu')
        self.dense1 = nn.Dense(300, activation='relu')
        self.dense2 = nn.Dense(self.action_dim, activation='tanh')

    def forward(self, state):
        action = self.dense2(self.dense1(self.dense0(state)))
        upper_bound = self.action_bound[:, 1]
        action = action * upper_bound
        return action


class Critic(nn.Block):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.dense0 = nn.Dense(400, activation='relu')
        self.dense1 = nn.Dense(300, activation='relu')
        self.dense2 = nn.Dense(1)

    def forward(self, state, action):
        input = nd.concat(state, action, dim=1)
        q_value = self.dense2(self.dense1(self.dense0(input)))
        return q_value


class TD3:
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound,
                 actor_learning_rate,
                 critic_learning_rate,
                 batch_size,
                 memory_size,
                 gamma,
                 tau,
                 explore_steps,
                 policy_update,
                 policy_noise,
                 explore_noise,
                 noise_clip,
                 ctx):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = nd.array(action_bound, ctx=ctx)

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.tau = tau
        self.explore_steps = explore_steps
        self.policy_update = policy_update
        self.policy_noise = policy_noise
        self.explore_noise = explore_noise
        self.noise_clip = noise_clip
        self.ctx = ctx

        self.main_actor_network = Actor(state_dim, action_dim, self.action_bound)
        self.target_actor_network = Actor(state_dim, action_dim, self.action_bound)
        self.main_critic_network1 = Critic(state_dim, action_dim)
        self.target_critic_network1 = Critic(state_dim, action_dim)
        self.main_critic_network2 = Critic(state_dim, action_dim)
        self.target_critic_network2 = Critic(state_dim, action_dim)

        self.main_actor_network.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.target_actor_network.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.main_critic_network1.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.target_critic_network1.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.main_critic_network2.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.target_critic_network2.collect_params().initialize(init=init.Xavier(), ctx=ctx)

        self.actor_optimizer = gluon.Trainer(self.main_actor_network.collect_params(),
                                             'adam',
                                             {'learning_rate': self.actor_learning_rate})
        self.critic1_optimizer = gluon.Trainer(self.main_critic_network1.collect_params(),
                                               'adam',
                                               {'learning_rate': self.critic_learning_rate})
        self.critic2_optimizer = gluon.Trainer(self.main_critic_network2.collect_params(),
                                               'adam',
                                               {'learning_rate': self.critic_learning_rate})

        self.total_steps = 0
        self.total_train_steps = 0

        self.memory_buffer = MemoryBuffer(buffer_size=self.memory_size, ctx=ctx)

    def choose_action_train(self, state):
        state = nd.array([state], ctx=self.ctx)
        action = self.main_actor_network(state)
        # no noise clip
        noise = nd.normal(loc=0, scale=self.explore_noise, shape=action.shape, ctx=self.ctx)
        action += noise
        clipped_action = self.action_clip(action).squeeze()
        self.total_steps += 1
        return clipped_action

    def choose_action_evaluate(self, state):
        state = nd.array([state], ctx=self.ctx)
        action = self.main_actor_network(state).squeeze()
        return action

    def action_clip(self, action):
        n = len(self.action_bound)
        action_list = []
        for i in range(n):
            action = nd.clip(action[:, i],
                             a_min=float(self.action_bound[i][0].asnumpy()),
                             a_max=float(self.action_bound[i][1].asnumpy()))
            action_list.append(action.reshape(-1, 1))
        if len(action_list) == 1:
            return action_list[0]
        else:
            clipped_action = 1  # -------------
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
        self.total_train_steps += 1
        state_batch, action_batch, reward_batch, next_state_batch, done = self.memory_buffer.sample(self.batch_size)

        # --------------optimize the critic network--------------------
        with autograd.record():
            # choose next action according to target policy network
            next_action_batch = self.target_actor_network(next_state_batch)
            noise = nd.normal(loc=0, scale=self.policy_noise, shape=next_action_batch.shape, ctx=self.ctx)
            # with noise clip
            noise = nd.clip(noise, a_min=-self.noise_clip, a_max=self.noise_clip)
            next_action_batch = next_action_batch + noise
            clipped_action = self.action_clip(next_action_batch)

            # get target q value
            target_q_value1 = self.target_critic_network1(next_state_batch, clipped_action)
            target_q_value2 = self.target_critic_network2(next_state_batch, clipped_action)
            target_q_value = nd.minimum(target_q_value1, target_q_value2).squeeze()
            target_q_value = reward_batch + (1.0 - done) * (self.gamma * target_q_value)

            # get current q value
            current_q_value1 = self.main_critic_network1(state_batch, action_batch)
            current_q_value2 = self.main_critic_network2(state_batch, action_batch)
            loss = gloss.L2Loss()

            value_loss1 = loss(current_q_value1, target_q_value.detach())
            value_loss2 = loss(current_q_value2, target_q_value.detach())

        self.main_critic_network1.collect_params().zero_grad()
        value_loss1.backward()
        self.critic1_optimizer.step(self.batch_size)

        self.main_critic_network2.collect_params().zero_grad()
        value_loss2.backward()
        self.critic2_optimizer.step(self.batch_size)

        # ---------------optimize the actor network-------------------------
        if self.total_train_steps % self.policy_update == 0:
            with autograd.record():
                pred_action_batch = self.main_actor_network(state_batch)
                actor_loss = -nd.mean(self.main_critic_network1(state_batch, pred_action_batch))

            self.main_actor_network.collect_params().zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step(1)

            self.soft_update(self.target_actor_network, self.main_actor_network)
            self.soft_update(self.target_critic_network1, self.main_critic_network1)
            self.soft_update(self.target_critic_network2, self.main_critic_network2)

    def save_model(self):
        self.main_actor_network.save_parameters('TD3_main_actor_network.params')
        self.target_actor_network.save_parameters('TD3_target_actor_network_params')
        self.main_critic_network1.save_parameters('TD3_main_critic_network.params')
        self.main_critic_network2.save_parameters('TD3_main_critic_network.params')
        self.target_critic_network1.save_parameters('TD3_target_critic_network.params')
        self.target_critic_network2.save_parameters('TD3_target_critic_network.params')

    def load_model(self):
        self.main_actor_network.load_parameters('TD3_main_actor_network.params')
        self.target_actor_network.load_parameters('TD3_target_actor_network_params')
        self.main_critic_network1.load_parameters('TD3_main_critic_network.params')
        self.main_critic_network2.load_parameters('TD3_main_critic_network.params')
        self.target_critic_network1.load_parameters('TD3_target_critic_network.params')
        self.target_critic_network2.load_parameters('TD3_target_critic_network.params')


def main():
    env = gym.make('Pendulum-v0').unwrapped
    seed = 1
    env.seed(1)
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    ctx = gb.try_gpu()

    max_episodes = 300
    max_episode_steps = 500
    render = True
    env_action_bound = [[float(env.action_space.low), float(env.action_space.high)]]

    agent = TD3(state_dim=env.observation_space.shape[0],
                action_dim=int(env.action_space.shape[0]),
                action_bound=env_action_bound,
                actor_learning_rate=0.001,
                critic_learning_rate=0.001,
                batch_size=64,
                memory_size=100000,
                gamma=0.99,
                tau=0.005,
                explore_steps=1000,
                policy_update=2,
                policy_noise=0.2,
                explore_noise=0.1,
                noise_clip=0.5,
                ctx=ctx)

    episode_reward_list = []
    mode = input("train or test: ")

    if mode == 'train':
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
                next_state, reward, done, info = env.step(action)
                agent.memory_buffer.store_transition(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                if agent.total_steps >= agent.explore_steps:
                    agent.update()
                if done:
                    break
            print('episode %d ends with rewards %f ' % (episode, episode_reward))
            episode_reward_list.append(episode_reward)
        agent.save_model()

    elif mode == 'test':
        agent.load_model()
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
            print('episode %d ends with rewards %f ' % (episode, episode_reward))
            episode_reward_list.append(episode_reward)
    else:
        raise NameError('Wrong input')


if __name__ == '__main__':
    main()













