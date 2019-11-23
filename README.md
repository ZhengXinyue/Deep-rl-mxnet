# Deep-rl-mxnet
 Mxnet implementation of Deep Reinforcement Learning papers.
 
## Now this repository contains:
 - DQN (Simple implementation) -->[MountainCar-v0](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Nature%20DQN/Nature_DQN.py)
 - Double DQN (Simple implementation) -->[MountainCar-v0](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Double%20DQN/Double_DQN.py)
 - Dueling DQN (Simple implementation) -->[MountainCar-v0](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Dueling%20DQN/Dueling_DQN.py)
 - Policy Gradient (Simple implementation) -->[CartPole-v0](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Policy%20Gradient/Policy_Gradient.py)
 - DDPG (Detailed implementation) :star: -->[Pendulum-v0](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/DDPG/DDPG_Pendulum.py),[LunarLander-v2](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/DDPG/DDPG_LunarLander_v2.py)
 - PPO (Simple implementation) -->[CartPole-v0](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/PPO/PPO_discrete.py)
 - TD3 (Very detailed implementation) :star: :star: -->[Pendulum-v0](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/TD3/TD3_Pendulum.py),[LunarLander-v2](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/TD3/TD3_LunarLander_v2.py),[HalfCheetah-v2](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/TD3/TD3_HalfCheetahv2.py)
 
## In the future this will contain:    
 1. A3C   
 2. SAC   


## Requirements:
 
 - Python 3   
 - OpenAI gym
 - Mxnet(gpu or cpu) and gluonbook 
 - Box2d(optional)
 - Mujoco(optional)
 
## Basic Installation:
```
pip install gym
pip install gluonbook
pip install mxnet  (cpu version)
pip install mxnet-cu90  (gpu version, corresponding to your cuda version)
```
## Box2d Installation(Optional):
```
pip install box2d.py
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

## Contents:
 
 1. DQN 
 - [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602v1)
 - [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)    
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Nature%20DQN/DQN%20MountainCar-v0.png)
 2. Double DQN
 - [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461v3)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Double%20DQN/Double%20DQN%20MountainCar-v0.png)
 3. Dueling DQN 
 - [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581v3)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Dueling%20DQN/Dueling%20DQN%20MountainCar-v0.png)
 4. Policy Gradient 
 - [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/Policy%20Gradient/Policy%20Gradient.png)
 5. Deep Deterministic Policy Gradient 
 - [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/DDPG/DDPG_Pendulum-v0.png)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/DDPG/LunarLanderContinuous_v2.png)
 6. Proximal Policy Optimization
 - [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
 - [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/PPO/PPO_CartPole_v0.png)
 7. TD3 
 - [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/TD3/TD3_Pendulum-v0.png)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/TD3/LunarLanderContinuous_v2.png)
  ![image](https://github.com/ZhengXinyue/Deep-rl-mxnet/blob/master/TD3/HalfCheetah_v2.png)
 8. A3C 
 - [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783v2)
 9. SAC
 - [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290v2)
