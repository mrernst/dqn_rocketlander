from env.rocketlander import RocketLander
from util.agent import *
from util.metrics import MetricLogger, TBMetricLogger

import numpy as np
import os
import random, datetime
from pathlib import Path

import gym


PRINT_DEBUG_MSG = False

import imageio
import base64



def embed_mp4(filename):
	  """Embeds an mp4 file in the notebook."""
	  video = open(filename,'rb').read()
	  b64 = base64.b64encode(video)
	  tag = '''
	  <video width="640" height="480" controls>
		<source src="data:video/mp4;base64,{0}" type="video/mp4">
	  Your browser does not support the video tag.
	  </video>'''.format(b64.decode())
	
	  return IPython.display.HTML(tag)



def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
	filename = filename + ".mp4"
	with imageio.get_writer(filename, fps=fps) as video:
		for _ in range(num_episodes):
			observation = env.reset()
			done = False
			video.append_data(env.render(mode='rgb_array'))
			while not done:
				action = policy(observation)
				observation, reward, done, info = env.step(action)
				video.append_data(env.render(mode='rgb_array'))
	return embed_mp4(filename)

env = RocketLander()
env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
#checkpoint = Path('checkpoints/2021-03-24T19-18-17/landernet1_18.chkpt')

action_space_n = env.action_space.n
#action_space_n = 6
agent = Elon(state_dim=10, action_dim=action_space_n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)
tblogger = TBMetricLogger(save_dir)
episodes = 10000


for e in range(episodes):

	state = env.reset()

	while True:

		# 3. Render environment (the visual) [WIP]
		# if env.stepnumber > 5000:
		# 	print('[INFO] help, I am stuck')
		# 	print("Action Taken  ",action)
		# 	print("Observation   ",next_state)
		# 	print("Reward Gained ",reward)
		# 	print("Info          ",info,end='\n\n')
		# 	env.reset()
		
		# env.render()
		
		# 4. Run agent on the state
		action = agent.act(state)
		#action = env.action_space.sample()

		# 5. Agent performs action
		next_state, reward, done, info = env.step(action)
		
		if PRINT_DEBUG_MSG:
			print("Action Taken  ",action)
			print("Observation   ",next_state)
			print("Reward Gained ",reward)
			print("Info          ",info,end='\n\n')
		
		# 6. Remember
		agent.add_to_memory(state, next_state, action, reward, done)

		# 7. Learn
		q, loss = agent.learn()

		# 8. Logging
		logger.log_step(reward, loss, q)
		tblogger.log_step(reward, loss, q, success=env.landed_ticks)

		# 9. Update state
		state = next_state

		# 10. Check if end of game
		if done:
			break

	logger.log_episode()
	tblogger.log_episode()
	
	if e % 20 == 0:
		logger.record(
			episode=e,
			epsilon=agent.exploration_rate,
			step=agent.curr_step
		)

tblogger.writer.close()