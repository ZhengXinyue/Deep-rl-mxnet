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


class Dueling_network(gluon.nn.Block):
    def __init__(self, n_actions):
        super(Dueling_network, self).__init__()
        self.n_actions = n_actions
        self.dense0 = gluon.nn.Dense(32, activation='relu')
        self.dense1 = gluon.nn.Dense(16, activation='relu')
        self.advantage_dense = gluon.nn.Dense(self.n_actions)
        self.state_value_dense = gluon.nn.Dense(1)

    # different from DQN and Double DQN
    def forward(self, state):
        common_value = self.dense1(self.dense0(state))
        advantate = self.advantage_dense(common_value)
        state_value = self.state_value_dense(common_value).squeeze().reshape((state.shape[0], 1))  # use broadcast
        mean_advantage = (nd.sum(advantate, axis=1) / self.n_actions).detach().reshape((state.shape[0], 1))
        Q_value = state_value + advantate - mean_advantage
        return Q_value


class MemoryBuffer:
    def __init__(self, buffer_size, ctx):
        self.buffer = deque(maxlen=buffer_size)
        self.maxsize = buffer_size
        self.ctx = ctx

    def __len(self):
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


class Dueling_DQN:
    def __init__(self,
                 n_action,
                 n_feature,
                 init_epsilon,
                 final_epsilon,
                 gamma,
                 buffer_size,
                 batch_size,
                 replace_iter,
                 annealing_end,
                 learning_rate,
                 ctx
                 ):
        self.n_action = n_action
        self.n_feature = n_feature
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        # discount factor
        self.gamma = gamma
        # memory buffer size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        # replace the parameters of the target network every T time steps
        self.replace_iter = replace_iter
        # The number of step it will take to linearly anneal the epsilon to its min value
        self.annealing_end = annealing_end
        self.learning_rate = learning_rate
        self.ctx = ctx

        self.total_steps = 0
        self.replay_buffer = MemoryBuffer(self.buffer_size, ctx)       # use deque

        # build the network
        self.target_network = Dueling_network(n_action)
        self.main_network = Dueling_network(n_action)
        self.target_network.collect_params().initialize(init.Xavier(), ctx=ctx)  # initialize the params
        self.main_network.collect_params().initialize(init.Xavier(), ctx=ctx)

        # optimize the main network
        self.optimizer = gluon.Trainer(self.main_network.collect_params(), 'adam', {'learning_rate': self.learning_rate})

    def choose_action(self, state):
        state = nd.array([state], ctx=self.ctx)
        if nd.random.uniform(0, 1) > self.epsilon:
            # choose the best action
            q_value = self.main_network(state)
            action = int(nd.argmax(q_value, axis=1).asnumpy())
        else:
            # random choice
            action = random.choice(range(self.n_action))
        # anneal
        self.epsilon = max(self.final_epsilon,
                           self.epsilon - (self.init_epsilon - self.final_epsilon) / self.annealing_end)
        self.total_steps += 1
        return action

    def update(self):
        state_batch, action_batch, reward_batch, next_state_batch = self.replay_buffer.sample(
            self.batch_size)
        with autograd.record():
            # get the Q(s,a)
            all_current_q_value = self.main_network(state_batch)
            main_q_value = nd.pick(all_current_q_value, action_batch)

            # different from DQN
            # get next action from main network, then get its Q value from target network
            all_next_q_value = self.target_network(next_state_batch).detach()  # only get gradient of main network
            max_action = nd.argmax(all_current_q_value, axis=1)
            target_q_value = nd.pick(all_next_q_value, max_action).detach()

            target_q_value = reward_batch + self.gamma * target_q_value

            # record loss
            loss = gloss.L2Loss()
            value_loss = loss(target_q_value, main_q_value)
        self.main_network.collect_params().zero_grad()
        value_loss.backward()
        self.optimizer.step(batch_size=self.batch_size)

    def replace_parameters(self):
        self.main_network.save_parameters('Dueling DQN temp params')
        self.target_network.load_parameters('Dueling DQN temp params')
        print('Dueling DQN parameters replaced')

    def save_parameters(self):
        self.target_network.save_parameters('Dueling DQN target network parameters')
        self.main_network.save_parameters('Dueling DQN main network parameters')

    #
    def load_parameters(self):
        self.target_network.load_parameters('Dueling DQN target network parameters')   # model path
        self.main_network.load_parameters('Dueling DQN main network parameters')


# split a list into n sublists and sum them
def smooth_function(l, n):
    m = []
    split_list = np.array_split(l, n)
    for i in split_list:
        m.append(np.mean(i))
    return m


seed = 3
mx.random.seed(seed)
random.seed(seed)
np.random.seed(seed)
ctx = gb.try_gpu()
t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
env = gym.make('MountainCar-v0').unwrapped
env.seed(seed)
render = False


agent = Dueling_DQN(n_action=env.action_space.n,
                    n_feature=env.observation_space.shape[0],
                    init_epsilon=1,
                    final_epsilon=0.1,
                    gamma=0.99,
                    buffer_size=50000,
                    batch_size=32,
                    replace_iter=5000,
                    annealing_end=300000,
                    learning_rate=0.00005,
                    ctx=ctx
                    )

# if you want to load model
# agent.load_parameters()
episode_steps_list = []
for i_episode in range(100):
    state = env.reset()
    episode_steps = 0
    while True:
        if render:
            env.render()
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        position, velocity = next_state

        my_reward = abs(position - (-0.5))

        agent.replay_buffer.store_transition(state, action, my_reward, next_state)
        if agent.total_steps > 1000:
            agent.update()
        if agent.total_steps > 1000 and agent.total_steps % agent.replace_iter == 0:
            agent.replace_parameters()

        episode_steps += 1

        if done:
            print('Dueling DQN episode {} ends with success at time step {}'.format(i_episode, episode_steps))
            episode_steps_list.append(episode_steps)
            break
        state = next_state
agent.save_parameters()
env.close()

plt.plot(episode_steps_list)
plt.ylim(0, 20000)
plt.xlabel('episode')
plt.ylabel('episode steps')
plt.title('Dueling DQN MountainCar-v0')
plt.savefig('./Dueling DQN MountainCar-v0.png')
plt.show()





