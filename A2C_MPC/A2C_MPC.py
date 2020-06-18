import random
import math
from collections import defaultdict

import gym
import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import loss as gloss, nn


class Actor(nn.Block):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.dense0 = nn.Dense(400, activation='relu')
        self.dense1 = nn.Dense(300, activation='relu')
        self.dense2 = nn.Dense(self.action_dim)

    def forward(self, state):
        _ = self.dense2(self.dense1(self.dense0(state)))
        probs = nd.softmax(_, axis=1)
        return probs


class Critic(nn.Block):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense0 = nn.Dense(400, activation='relu')
        self.dense1 = nn.Dense(300, activation='relu')
        self.dense2 = nn.Dense(1)

    def forward(self, state):
        v_values = self.dense2(self.dense1(self.dense0(state)))
        return v_values


class SystemModel(nn.Block):
    def __init__(self, observation_dim):
        super(SystemModel, self).__init__()
        self.observation_dim = observation_dim
        self.dense0 = nn.Dense(500, activation='relu')
        self.dense1 = nn.Dense(500, activation='relu')
        self.dense2 = nn.Dense(500, activation='relu')
        self.dense3_state = nn.Dense(self.observation_dim)
        self.dense3_reward = nn.Dense(1)

    def forward(self, state, action):
        _ = nd.concat(state, action, dim=1)
        _ = self.dense2(self.dense1(self.dense0(_)))
        predict_state = self.dense3_state(_)
        predict_reward = self.dense3_reward(_)
        return predict_state, predict_reward


class A2C(object):
    def __init__(self,
                 gamma,
                 action_dim,
                 observation_dim,
                 with_mpc,
                 rollout,
                 planning_horizon,
                 ctx):
        self.gamma = gamma
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.with_mpc = with_mpc
        self.rollout = rollout
        self.horizon = planning_horizon
        self.ctx = ctx

        self.actor_network = Actor(self.action_dim)
        self.critic_network = Critic()
        self.system_model = SystemModel(self.observation_dim)
        self.actor_network.initialize(init=init.Xavier(), ctx=self.ctx)
        self.critic_network.initialize(init=init.Xavier(), ctx=self.ctx)
        self.system_model.initialize(init=init.Xavier(), ctx=self.ctx)
        self.actor_optimizer = gluon.Trainer(self.actor_network.collect_params(), 'adam')
        self.critic_optimizer = gluon.Trainer(self.critic_network.collect_params(), 'adam')
        self.system_model_optimizer = gluon.Trainer(self.system_model.collect_params(), 'adam')

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.total_rewards = []
        # self.model_loss = 1

    def compute_returns(self, next_return):
        r = next_return
        self.total_rewards = [0] * len(self.rewards)
        for step in reversed(range(len(self.rewards))):
            r = self.rewards[step] + self.gamma * r * (1 - self.dones[step])
            self.total_rewards[step] = r

    def store_transition(self, state, action, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def pi_choose_action(self, state):
        state = nd.array([state], ctx=self.ctx)
        all_action_prob = self.actor_network(state)
        action = int(nd.sample_multinomial(all_action_prob).asnumpy())
        # action = int(nd.argmax(all_action_prob.squeeze()).asnumpy())
        return action

    # choose the best action using pure Learning_MPC(random sampling)
    def pure_mpc_choose_action(self, state):
        initial_state = nd.array([state], ctx=self.ctx)
        mpc_action = None
        max_value = -float('inf')
        value_dict = defaultdict(list)
        for _ in range(self.rollout):
            trajectory_value = 0
            state = initial_state
            actions = []
            for j in range(self.horizon):
                action = random.choice(list(range(self.action_dim)))
                if j == 0:
                    first_action = action
                actions.append(action)
                action = nd.array([[action]], ctx=self.ctx)
                predict_state, predict_reward = self.system_model(state, action)
                x, x_dot, theta, theta_dot = predict_state.squeeze()
                trajectory_value += self.gamma ** j * predict_reward.squeeze().asscalar()
                state = predict_state
            final_value = self.critic_network(state).squeeze().asscalar()
            trajectory_value += self.gamma ** self.horizon * final_value
            if trajectory_value > max_value:
                mpc_action = first_action
                max_value = trajectory_value
            value_dict[tuple(actions)].append(trajectory_value)
        pi_action = int(nd.sample_multinomial(self.actor_network(initial_state)).asnumpy())
        print('mpc: ', mpc_action)
        print('pi: ', pi_action)
        # print(dict(value_dict))
        print('-------------------------------')
        return mpc_action

    # choose the best action using Learning_MPC and pi
    def mpc_pi_choose_action(self, state):
        initial_state = nd.array([state], ctx=self.ctx)
        best_action = None
        max_value = -float('inf')
        for _ in range(self.rollout):
            trajectory_value = 0
            state = initial_state
            for j in range(self.horizon):
                probs = self.actor_network(state)
                action = int(nd.sample_multinomial(probs).asnumpy())
                if j == 0:
                    first_action = action
                action = nd.array([[action]], ctx=self.ctx)
                predict_state, predict_reward = self.system_model(state, action)
                reward = 0
                trajectory_value += self.gamma ** j * reward
                state = predict_state
            final_value = self.critic_network(state).squeeze().asscalar()
            trajectory_value += self.gamma ** self.horizon * final_value
            if trajectory_value > max_value:
                best_action = first_action
                max_value = trajectory_value
        return best_action

    def update(self):
        states = nd.array(self.states, ctx=self.ctx)
        actions = nd.array(self.actions, ctx=self.ctx)
        rewards = nd.array(self.rewards, ctx=self.ctx)
        total_rewards = nd.array(self.total_rewards, ctx=self.ctx)
        next_states = nd.array(self.next_states, ctx=self.ctx)

        # ------------optimize actor-----------
        with autograd.record():
            values = self.critic_network(states)
            probs = self.actor_network(states)
            advantages = (total_rewards - values).detach()
            loss = -nd.pick(probs, actions).log() * advantages
        self.actor_network.collect_params().zero_grad()
        loss.backward()
        self.actor_optimizer.step(batch_size=len(states))

        # -----------optimize critic------------
        with autograd.record():
            values = self.critic_network(states)
            l2_loss = gloss.L2Loss()
            loss = l2_loss(values, total_rewards)
        self.critic_network.collect_params().zero_grad()
        loss.backward()
        self.critic_optimizer.step(batch_size=len(states))

        # -----------optimize model--------------
        if self.with_mpc:
            actions = actions.reshape(-1, 1)
            with autograd.record():
                predict_states, predict_rewards = self.system_model(states, actions)
                l2_loss = gloss.L2Loss()
                loss1 = l2_loss(predict_states, next_states)
                loss2 = l2_loss(predict_rewards, rewards)
                loss = loss1 + loss2
            self.system_model.collect_params().zero_grad()
            loss.backward()
            self.system_model_optimizer.step(batch_size=len(states))

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.total_rewards = []

    def save(self):
        self.actor_network.save_parameters('A2C_CartPole_actor_network.params')
        self.critic_network.save_parameters('A2C_CartPole_critic_network.params')
        self.system_model.save_parameters('A2C_CartPole_system_model.params')

    def load(self):
        self.actor_network.load_parameters('A2C_CartPole_actor_network.params')
        self.critic_network.load_parameters('A2C_CartPole_critic_network.params')
        self.system_model.load_parameters('A2C_CartPole_system_model.params')


if __name__ == '__main__':
    seed = 77777777
    np.random.seed(seed)
    mx.random.seed(seed)
    env = gym.make('CartPole-v0').unwrapped
    env.seed(seed)
    ctx = mx.cpu()
    render = False
    MPC_signal = True
    horizon = 30
    rollout = 32

    agent = A2C(gamma=0.99,
                action_dim=env.action_space.n,
                observation_dim=env.observation_space.shape[0],
                with_mpc=MPC_signal,
                rollout=rollout,
                planning_horizon=horizon,
                ctx=ctx)
    agent.load()
    episode_reward_list = []
    max_episodes = 400
    max_episode_steps = 500
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        for episode_step in range(max_episode_steps):
            if render:
                env.render()
            action = agent.pure_mpc_choose_action(state)
            next_state, reward, done, info = env.step(action)
            # print(next_state)
            # print(agent.system_model(nd.array([state], ctx=ctx), nd.array([action], ctx=ctx).reshape(-1, 1)))
            # print('-----------------------------')
            x, x_dot, theta, theta_dot = next_state
            if done:
                reward = -1
            agent.store_transition(state, action, reward, done, next_state)
            episode_reward += reward
            if done:
                break
            state = next_state

        agent.compute_returns(reward)
        print('episode %d ends with reward %d' % (episode, episode_reward))
        episode_reward_list.append(episode_reward)
        agent.update()

    # agent.save()
    env.close()

    plt.plot(episode_reward_list)
    plt.xlabel('episode')
    plt.ylabel('episode reward')
    plt.title('A2C_CartPole_v0')
    # plt.savefig('./A2C_CartPole_v0.png')
    plt.show()
