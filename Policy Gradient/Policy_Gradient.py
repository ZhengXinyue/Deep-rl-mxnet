import time

import gym
import numpy as np
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, nd, autograd, init
import gluonbook as gb


class PGNetwork(gluon.nn.Block):
    def __init__(self, n_actions):
        super(PGNetwork, self).__init__()
        self.n_actions = n_actions
        self.dense0 = gluon.nn.Dense(32, activation='relu')
        self.dense1 = gluon.nn.Dense(16, activation='relu')
        self.dense2 = gluon.nn.Dense(self.n_actions)

    def forward(self, state):
        values = self.dense2(self.dense1(self.dense0(state)))
        probs = nd.softmax(values, axis=1)
        return probs


class PG:
    def __init__(self,
                 learning_rate,
                 gamma,
                 n_action,
                 n_feature,
                 ctx):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_actions = n_action
        self.n_feature = n_feature
        self.ctx = ctx

        self.network = PGNetwork(self.n_actions)
        self.network.initialize(init=init.Xavier(), ctx=ctx)
        self.optimizer = gluon.Trainer(self.network.collect_params(),
                                       'adam',
                                       {'learning_rate': self.learning_rate})

        self.states = []
        self.actions = []
        self.rewards = []

    def store_transition(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def choose_action(self, state):
        state = nd.array([state], ctx=self.ctx)
        all_action_prob = self.network(state)
        action = int(nd.sample_multinomial(all_action_prob).asnumpy())
        return action

    def discount_and_normalized_rewards(self):
        # discounted episode rewards
        discounted_rewards = np.zeros_like(self.rewards)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_rewards[t] = running_add
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards

    def learn(self):
        rewards = nd.array(self.discount_and_normalized_rewards(), ctx=self.ctx)
        states = nd.array(self.states, ctx=self.ctx)
        with autograd.record():
            probs = self.network(states)
            actions = nd.array(self.actions, ctx=self.ctx)

            loss = -nd.pick(probs, actions).log() * rewards
        loss.backward()
        self.optimizer.step(batch_size=len(self.states))

        # reset
        self.states = []
        self.actions = []
        self.rewards = []

    def save(self):
        self.network.save_parameters('Policy Gradient parameters')
        print('Parameters saved')

    def load(self):
        self.network.load_parameters('Policy Gradient parameters')
        print('Parameters loaded')


# split a list into n sublists and sum them
def smooth_function(l, n):
    m = []
    split_list = np.array_split(l, n)
    for i in split_list:
        m.append(np.mean(i))
    return m


t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
seed = 3
np.random.seed(seed)
mx.random.seed(seed)
env = gym.make('CartPole-v0').unwrapped
env.seed(seed)
RENDER = False
ctx = gb.try_gpu()

agent = PG(learning_rate=0.0005,
           gamma=0.99,
           n_action=env.action_space.n,
           n_feature=env.observation_space.shape[0],
           ctx=ctx)

# you can get good performance if load the parameters
# agent.load()
episode_reward_list = []
for i_episode in range(1000):
    state = env.reset()
    while True:
        if RENDER:
            env.render()
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward)
        if done:
            episode_reward = sum(agent.rewards)
            print('episode %d ends with reward %d' % (i_episode, episode_reward))
            episode_reward_list.append(episode_reward)
            mean_reward = sum(episode_reward_list[-100:]) / 100
            # render if get good performance
            if mean_reward > 195:
                RENDER = True
            agent.learn()
            break
        state = next_state
agent.save()
env.close()


l = smooth_function(episode_reward_list, 250)
plt.plot(l)
plt.xlabel('episode')
plt.ylabel('episode reward')
plt.title('Policy Gradient CartPole-v0')
plt.savefig('./Policy Gradient.png')
plt.show()
