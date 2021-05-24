# Deep Q-Learning Rocketlander

<p align="center">
  <img src="https://github.com/mrernst/dqn_rocketlander/blob/main/imgs/rocket1.png" width="175">
  <img src="https://github.com/mrernst/dqn_rocketlander/blob/main/imgs/rocket2.png" width="175">
  <img src="https://github.com/mrernst/dqn_rocketlander/blob/main/imgs/rocket3.png" width="175">
  
DQN Rocketlander was a 1-week mini-project work during the summer semester 2019 course reinforcement learning. It is an implementation of deep double Q Learning with experience replay to tackle autonomous rocket landing of a first stage on a bark.

## Deep Q-Learning

Deep Q Learning [2] is based on traditional [Q-learning](https://en.wikipedia.org/wiki/Q-learning) and uses neural networks as function approximators. As an a off-policy reinforcement learning technique [1], DQN uses discreete action spaces and has proven to be very successful robotics applications and atari games [3].


[1] Sutton, R. S. & Barto, A. G. 1998, Reinforcement learning: An introduction, MIT Press Cambridge

[2] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. 2015, 'Human-level control through deep reinforcement learning', Nature, 518, 7540, 529--533.

[3] Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. 2013, 'Playing Atari with Deep Reinforcement Learning'


## The task / the environment

The environment is borrowed from the fantastic work from [embersarc](https://github.com/EmbersArc/) (star [his work](https://github.com/EmbersArc/gym-rocketlander) on github if you like it). It is inpired by the reusable falcon-9 system that autonomously lands it's first stage on a bark in the ocean.



## Getting started with the repository

There are different python programs, depending on the use-case. 

*  `main.py`
*  `evaluate.py`
*  `testrun.py`


To train the RL agent, run `python3 main.py` from the main folder. This will result in a training run and data-logging in tensorboard.

To evaluate a run, choose the path to the last saved network parameters and run `python3 evaluate.py`, this loads the parameters and produces a little output video.

To take a look at the tuned PID controllers at work run `python3 testrun.py`


### Prerequisites

* [numpy](http://www.numpy.org/)
* [pytorch](https://pytorch.org)
* [tensorboard](https://www.tensorflow.org)
* [ffmpeg](https://ffmpeg.org/)
* [matplotlib](https://matplotlib.org/)
* [gym](https://gym.openai.com/)
* [box2d](https://box2d.org/)


### Directory structure

```bash
.
├── main.py   								      # main python file
├── evaluate.py    							    # evaluate trained agent
├── testrun.py      						    # use PID controllers
├── README.md
├── LICENSE
├── imgs                            # image folder for the readme
│	├── failure.gif
│	├── success.gif
│	├── still1.png
│	├── still2.png
│	└── still3.png
├── env                           	# modified env from embersarc
│	└── rocketlander.py
├── util                           	# utilities
│	├── agent.py
│	├── metrics.py
│	├── neuralnet.py
│	├── pid.py
│	└── visualization.py
└── checkpoints                     # checkpoints/logs
	└── YOUR-CHECKPOINTS


```

### Difficulties

To make this environment work with deep Q learning instead of more powerful approaches like PPO and limited training ressources (Macbook Pro, 2016) the original problem was simplified a bit. Thus I set a smaller start height and start speed and limited the initial angular velocity. An agent successfully trained can then be adapted to more difficult scenarios in terms of curriculum learning.

#### Reward Shaping and Heuristics

Another area where extensive changes were made to the environment is the reward shaping function, take a look at the code for details. To speed up learning a maximum angle was defined at which the episode is terminated.

Additionally it helped to force engine shutdown on contact of one of the landing legs. This is due to the fact that a "suicide-burn" is the most fuel-efficient way to slow down the Rocket. But having the engine on full-thrust on contact makes it hard to successfully land by rapidly throttling down.

### Solving the Environment with PID controllers

There's some additional experiments included with traditional PID controllers. It's interesting that these simple controllers perform reasonably well on the task, but keep in mind that it has full control over continuous actions and in contrast the behavior of the RL agent is learned from experience without having a-priori information about the environment.


### Training outcome

###### 750 episodes
<p align="center">
<img src="https://github.com/mrernst/dqn_rocketlander/blob/main/imgs/failure.gif" width="320">

###### 3700 episodes
<p align="center">
<img src="https://github.com/mrernst/dqn_rocketlander/blob/main/imgs/success.gif" width="320">
