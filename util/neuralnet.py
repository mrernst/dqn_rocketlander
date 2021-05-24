from torch import nn
import copy


class LanderNet2(nn.Module):
	"""
	first try of an DQN approximation network for the
	rocket-lander problem
	"""
	def __init__(self, input_dim, output_dim):
		super().__init__()

		self.online = nn.Sequential(
			nn.Linear(input_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, output_dim)
			)

		self.target = copy.deepcopy(self.online)

		# Q_target parameters are frozen.
		for p in self.target.parameters():
			p.requires_grad = False

	def forward(self, input, model):
		if model == 'online':
			return self.online(input)
		elif model == 'target':
			return self.target(input)

class LanderNet3(nn.Module):
	"""
	first try of an DQN approximation network for the
	rocket-lander problem
	"""
	def __init__(self, input_dim, output_dim):
		super().__init__()

		self.online = nn.Sequential(
			nn.Linear(input_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, output_dim)
			)

		self.target = copy.deepcopy(self.online)

		# Q_target parameters are frozen.
		for p in self.target.parameters():
			p.requires_grad = False

	def forward(self, input, model):
		if model == 'online':
			return self.online(input)
		elif model == 'target':
			return self.target(input)

class LanderNet1(nn.Module):
	"""
	first try of an DQN approximation network for the
	rocket-lander problem
	"""
	def __init__(self, input_dim, output_dim):
		super().__init__()

		self.online = nn.Sequential(
			nn.Linear(input_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, output_dim)
			)

		self.target = copy.deepcopy(self.online)

		# Q_target parameters are frozen.
		for p in self.target.parameters():
			p.requires_grad = False

	def forward(self, input, model):
		if model == 'online':
			return self.online(input)
		elif model == 'target':
			return self.target(input)