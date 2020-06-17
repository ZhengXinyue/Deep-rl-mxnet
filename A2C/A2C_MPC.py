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
        self.dense0 = nn.Dense(400, activation='relu')
        self.dense1 = nn.Dense(300, activation='relu')
        self.dense2 = nn.Dense(self.observation_dim)

    def forward(self, state, action):
        _ = nd.concat(state, action, dim=1)
        predict_state = self.dense2(self.dense1(self.dense0(_)))
        return predict_state


class A2C(object):
    def __init__(self,
                 gamma,
                 action_dim,
                 observation_dim,
                 with_mpc,
                 ctx):
        self.gamma = gamma
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.with_mpc = with_mpc
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
        self.total_reward = []

    def compute_returns(self, next_return):
        r = next_return
        self.total_reward = [0] * len(self.rewards)
        for step in reversed(range(len(self.rewards))):
            r = self.rewards[step] + self.gamma * r * (1 - self.dones[step])
            self.total_reward[step] = r

    def store_transition(self, state, action, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def choose_action(self, state):
        state = nd.array([state], ctx=self.ctx)
        all_action_prob = self.actor_network(state)
        action = int(nd.sample_multinomial(all_action_prob).asnumpy())
        return action

    def update(self):
        states = nd.array(self.states, ctx=self.ctx)
        actions = nd.array(self.actions, ctx=self.ctx)
        total_reward = nd.array(self.total_reward, ctx=self.ctx)
        next_states = nd.array(self.next_states, ctx=self.ctx)

        # ------------optimize actor-----------
        with autograd.record():
            values = self.critic_network(states)
            probs = self.actor_network(states)
            advantages = (total_reward - values).detach()
            loss = -nd.pick(probs, actions).log() * advantages
        self.actor_network.collect_params().zero_grad()
        loss.backward()
        self.actor_optimizer.step(batch_size=len(states))

        # -----------optimize critic------------
        with autograd.record():
            values = self.critic_network(states)
            l2_loss = gloss.L2Loss()
            loss = l2_loss(values, total_reward)
        self.critic_network.collect_params().zero_grad()
        loss.backward()
        self.critic_optimizer.step(batch_size=len(states))

        # -----------optimize model--------------
        if self.with_mpc:
            actions = actions.reshape(-1, 1)
            with autograd.record():
                predict_states = self.system_model(states, actions)
                l2_loss = gloss.L2Loss()
                loss = l2_loss(predict_states, next_states)
                # print(loss.mean())
            self.system_model.collect_params().zero_grad()
            loss.backward()
            self.system_model_optimizer.step(batch_size=len(states))

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.total_reward = []

    def save(self):
        self.actor_network.save_parameters('A2C_CartPole_actor_network.params')
        self.critic_network.save_parameters('A2C_CartPole_critic_network.params')
        self.system_model.save_parameters('A2C_CartPole_system_model.params')


if __name__ == '__main__':
    seed = 777777
    np.random.seed(seed)
    mx.random.seed(seed)
    env = gym.make('CartPole-v0').unwrapped
    env.seed(seed)
    ctx = mx.cpu()
    render = False
    MPC = False

    agent = A2C(gamma=0.99,
                action_dim=env.action_space.n,
                observation_dim=env.observation_space.shape[0],
                with_mpc=MPC,
                ctx=ctx)

    episode_reward_list = []
    max_episodes = 200
    max_episode_steps = 1000
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        reward = 0
        for episode_step in range(max_episode_steps):
            if render:
                env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, done, next_state)
            episode_reward += reward
            if done:
                break
            state = next_state

        agent.compute_returns(reward)
        print('episode %d ends with reward %d' % (episode, episode_reward))
        episode_reward_list.append(episode_reward)
        agent.update()

    agent.save()
    env.close()

    plt.plot(episode_reward_list)
    plt.xlabel('episode')
    plt.ylabel('episode reward')
    plt.title('A2C_CartPole_v0')
    plt.savefig('./A2C_CartPole_v0.png')
    plt.show()
