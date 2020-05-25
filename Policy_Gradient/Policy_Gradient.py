import gym
import numpy as np
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, nd, autograd, init


class PGNetwork(gluon.nn.Block):
    def __init__(self, n_actions):
        super(PGNetwork, self).__init__()
        self.n_actions = n_actions
        self.dense0 = gluon.nn.Dense(400, activation='relu')
        self.dense1 = gluon.nn.Dense(300, activation='relu')
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
                 ctx):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_actions = n_action
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
        # baseline
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
        self.network.save_parameters('Policy_Gradient_parameters')
        print('Parameters saved')

    def load(self):
        self.network.load_parameters('Policy_Gradient_parameters')
        print('Parameters loaded')


if __name__ == '__main__':
    seed = 7777777
    np.random.seed(seed)
    mx.random.seed(seed)
    env = gym.make('CartPole-v0').unwrapped
    env.seed(seed)
    render = False
    ctx = mx.cpu()

    agent = PG(learning_rate=0.0005,
               gamma=0.99,
               n_action=env.action_space.n,
               ctx=ctx)

    episode_reward_list = []
    for i_episode in range(200):
        state = env.reset()
        episode_reward = 0
        while True:
            if render:
                env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward)
            episode_reward += reward
            if done:
                print('episode %d ends with reward %d' % (i_episode, episode_reward))
                episode_reward_list.append(episode_reward)
                agent.learn()
                break
            state = next_state
    agent.save()
    env.close()

    plt.plot(episode_reward_list)
    plt.xlabel('episode')
    plt.ylabel('episode reward')
    plt.title('Policy_Gradient CartPole-v0')
    plt.savefig('./Policy-Gradient-CartPole-v0.png')
    plt.show()
