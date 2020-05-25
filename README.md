# Deep-rl-mxnet
 Mxnet implementation of Deep Reinforcement Learning papers.
 
## Now this repository contains:
  1. DQN [[code]](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Nature_DQN/Nature_DQN.py)
 - [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602v1)
 - [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)  
  2. Double DQN [[code]](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Double_DQN/Double_DQN.py)
 - [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461v3)
  3. Dueling DQN [[code]](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Dueling_DQN/Dueling_DQN.py)
 - [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581v3)
 4. Policy Gradient [[code]](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Policy_Gradient/Policy_Gradient.py)
 - [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
 5. Deep Deterministic Policy Gradient [[code]](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/DDPG/DDPG_Pendulum_v0.py) (Detailed implementation) :star: 
 - [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
  6. Proximal Policy Optimization [[code]](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/PPO/PPO_discrete.py)
 - [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
 - [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286)
  7. TD3 [[code]](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/TD3/TD3_LunarLander_v2.py) (Very detailed implementation) :star: :star:
 - [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
 
 
## Installation
```
$ git clone https://github.com/ZhengXinyue/Deep-rl-mxnet
$ cd Deep-rl-mxnet
```
create & activate virtual env then install dependency:

with venv/virtualenv + pip:
```
$ python -m venv env  # use `virtualenv env` for Python2, use `python3 ...` for Python3 on Linux & macOS
$ source env/bin/activate  # use `env\Scripts\activate` on Windows
$ pip install -r requirements.txt
```
 
If you get something like this: 
```
unable to execute 'swig': No such file or directory
```
do:
```
sudo apt-get install swig
```

## Mujoco Installation(Optional):
Please refer to [this repository](https://github.com/openai/mujoco-py)


## Examples:
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Nature_DQN/DQN-CartPole-v0.png)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Double_DQN/Double-DQN-CartPole-v0.png)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Dueling_DQN/Dueling-DQN-CartPole-v0.png)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Policy_Gradient/Policy-Gradient-CartPole-v0.png)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/DDPG/DDPG-Pendulum-v0.png)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/PPO/PPO-CartPole-v0.png)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/TD3/TD3-LunarLanderContinuous-v2.png)
  
  
## Maybe In the future:
 8. A3C 
 - [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783v2)
 9. SAC
 - [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290v2)
