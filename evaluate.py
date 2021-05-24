from env.rocketlander import RocketLander
from util.agent import *
from util.metrics import MetricLogger

import numpy as np
import os
import random, datetime
from pathlib import Path

import gym

PRINT_DEBUG_MSG = True

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



def create_policy_eval_video(policy, filename, num_episodes=5, fps=60):
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


def create_policy_eval_gif(policy, filename, num_episodes=1, fps=60, skip_frames=3):
	filename = filename + ".gif"
	with imageio.get_writer(filename, fps=fps) as video:
		f = 0
		for _ in range(num_episodes):
			observation = env.reset()
			done = False
			video.append_data(env.render(mode='rgb_array'))
			while not done:
				action = policy(observation)
				observation, reward, done, info = env.step(action)
				if (f%skip_frames == 0):
					video.append_data(env.render(mode='rgb_array'))
				f += 1
	return video


from PIL import Image, ImageSequence
def resize_gif(path, out_name='out.gif', size=(640,480)):
	
	im = Image.open(path)
	frames = ImageSequence.Iterator(im)
	
	# wrap thumbnail generator
	def thumbnails(frames):
		for frame in frames:
			thumbnail = frame.copy()
			thumbnail.thumbnail(size, Image.ANTIALIAS)
			yield thumbnail
	
	frames = thumbnails(frames)
	
	# Save output
	om = next(frames) # Handle first frame separately
	om.info = im.info # Copy sequence info
	om.save(out_name, save_all=True, append_images=list(frames))




env = RocketLander()
env.reset()

# save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
# save_dir.mkdir(parents=True)

checkpoint = Path('checkpoints/2021-03-25T21-14-53/landernet1_15.chkpt')
checkpoint = Path('checkpoints/2021-03-25T21-14-53/landernet1_4.chkpt')
checkpoint = Path('checkpoints/2021-03-25T21-14-53/landernet1_2.chkpt')
checkpoint = Path('checkpoints/2021-03-26T14-28-56/landernet1_21.chkpt')

#checkpoint = Path('checkpoints/2021-03-25T21-15-29/landernet1_14.chkpt')



action_space_n = env.action_space.n
#action_space_n = 6
agent = Elon(state_dim=10, action_dim=action_space_n, save_dir=None, checkpoint=checkpoint)

agent.exploration_rate = agent.exploration_rate_min
agent.exploration_rate = 0.

#logger = MetricLogger(save_dir)

create_policy_eval_video(agent.act, "evaluation", num_episodes=15)

sys.exit()

episodes = 15

for e in range(episodes):

	state = env.reset()

	while True:

		env.render()

		action = agent.act(state)

		next_state, reward, done, info = env.step(action)
		
		if PRINT_DEBUG_MSG:
			print("Action Taken  ",action)
			print("Observation   ",next_state)
			print("Reward Gained ",reward)
			print("Info          ",info,end='\n\n')
		
		#agent.cache(state, next_state, action, reward, done)

		#logger.log_step(reward, None, None)

		state = next_state

		if done:
			break

