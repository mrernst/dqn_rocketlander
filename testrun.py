import gym
import gym.spaces
from env.rocketlander import RocketLander

from util.pid import PID_Benchmark

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



# Initialize the PID algorithm
pid = PID_Benchmark()

env = RocketLander(continuous=True)
observation = env.reset()

PRINT_DEBUG_MSG = True

create_policy_eval_video(pid.pid_algorithm, 'PID', fps=60)

for e in range(10):
	while True:
		env.render()
		#action = env.action_space.sample()
		action = pid.pid_algorithm(observation)
		observation,reward,done,info = env.step(action)
	
		if PRINT_DEBUG_MSG:
			print("Action Taken  ",action)
			print("Observation   ",observation)
			print("Reward Gained ",reward)
			print("Info          ",info,end='\n\n')
	
		if done:
			print("Simulation done.")
			env.reset()
			break
env.close()