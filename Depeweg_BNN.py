"""
Bayesian Neural Network Implementation for HiP-MDP.
Adapted from original code by Jose Miguel Hernandez Lobato,
following https://arxiv.org/abs/1605.07127 and code bt Depeweg in the repo https://github.com/siemens/policy_search_bb-alpha
"""
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import math
import torch
import torch.optim as optim
import copy
class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_weights = 0

    def add_shape(self, name, shape):
        start = self.num_weights
        self.num_weights += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.num_weights), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return torch.reshape(vect[idxs], shape)

    def get_indexes(self, vect, name):
        idxs, _ = self.idxs_and_shapes[name]
        return idxs


class BayesianNeuralNetwork(object):
	"""Meta-class to handle much of the neccessary computations for BNN inference."""
	def __init__(self, param_set, nonlinearity):
		"""Initialize BNN.

		Arguments:
		param_set -- a dictionary containing the following keys:
			bnn_layer_sizes -- number of units in each layer including input and output
			weight_count -- numbber of latent weights
			num_state_dims -- state dimension
			bnn_num_samples -- number of samples of network weights drawn to approximate transitions
			bnn_alpha -- alpha divergence parameter
			bnn_batch_size -- minibatch size
			num_strata_samples -- number of transitions to draw from each strata in the prioritized replay
			bnn_training_epochs -- number of epochs of SGD in updating BNN network weights
			bnn_v_prior -- prior on the variance of the BNN weights
			bnn_learning_rate -- learning rate for BNN network weights
			wb_learning_rate -- learning rate for latent weights
			wb_num_epochs -- number of epochs of SGD in updating latent weights
		nonlinearity -- activation function for hidden layers

		"""
		# Initialization is an adaptation of make_nn_funs() from 
		# https://github.com/HIPS/autograd/blob/master/examples/bayesian_neural_net.py
		layer_sizes = param_set['bnn_layer_sizes']
		self.shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
		self.num_weights = sum((m+1) * n for m,n in self.shapes)
		self.parser = WeightsParser()
		self.parser.add_shape('mean', (self.num_weights,1))
		self.parser.add_shape('log_variance', (self.num_weights,1))
		self.parser.add_shape('log_v_noise', (1,1))
		w = .1*np.random.randn(self.parser.num_weights)
		w[self.parser.get_indexes(w,'log_variance')] = -10
		w[self.parser.get_indexes(w, 'log_v_noise')] = np.log(param_set['theta'])
		self.weights = torch.tensor(w,requires_grad=True)
		self.nonlinearity = nonlinearity
		# Number of latent parameters per instance
		self.num_latent_params = param_set['weight_count']
		self.num_state_dims = param_set['num_state_dims']
		# Number of samples drawn from BNN parameter distributions
		self.num_drawn_samples = param_set['bnn_num_samples']
		# Parameters for BNN training
		self.alpha = param_set['bnn_alpha']
		self.N = param_set['bnn_batch_size'] 
		self.bnn_batch_size = param_set['bnn_batch_size']
		self.train_epochs = param_set['bnn_training_epochs']
		self.v_prior = param_set['bnn_v_prior']
		self.learning_rate = param_set['bnn_learning_rate']
		self.gamma = param_set['gamma']
		self.lamda = param_set['lambda']
		self.N_train = param_set['N_train']
		self.input_parser = WeightsParser()
		self.input_parser.add_shape('mean', (self.N_train,1))
		self.input_parser.add_shape('log_variance', (self.N_train,1))
		z = .001*np.random.randn(self.input_parser.num_weights)
		self.v__z = torch.tensor(param_set['lambda'])
		z[self.input_parser.get_indexes(z,'log_variance')] = self.v__z
		self.inputs = z
		self.v_prior_z_2 = self.__logistic__(self.v__z)
		
	def __make_batches__(self, shape=0):
		if shape > 0:
			return [slice(i, min(i+self.bnn_batch_size, shape)) for i in range(0, shape, self.bnn_batch_size)]
		else:
			return [slice(i, min(i+self.bnn_batch_size, self.N)) for i in range(0, self.N, self.bnn_batch_size)]

	def __unpack_layers__(self, weight_samples):
		for m, n in self.shapes:
			yield weight_samples[:(m*n)].reshape((m, n)), \
				  weight_samples[(m*n):((m*n) + n)].reshape(( 1, n))
			weight_samples = weight_samples[((m+1) * n):]
	def __predict__(self,weight_samples,inputs):
		k = 0
		for W,b in self.__unpack_layers__(weight_samples=weight_samples):
			outputs = torch.matmul(inputs,W) + b
			inputs = torch.max(outputs, torch.tensor(0))
		return outputs

	def __draw_samples__(self, q):
		return torch.randn(self.num_drawn_samples, len(q['m'])) * torch.sqrt(q['v']) + q['m']
	def __draw_samples_q__(self, q):
		return torch.randn(self.num_drawn_samples,self.bnn_batch_size, len(q['m'])) * torch.sqrt(q['v']) + q['m']
	def __logistic__(self, x): return 1.0 / (1.0+torch.exp(-x))

	def __get_parameters_q__(self, weights, scale=1.0):
		v = self.v_prior * self.__logistic__(self.parser.get(weights,'log_variance'))[:,0]
		m = self.parser.get(weights,'mean')[:,0]
		return {'m': m, 'v': v}
	def __get_parameters_z__(self, z, scale=1.0):
		v = self.v_prior_z_2  * self.__logistic__(z[self.bnn_batch_size:])
		m = z[:self.bnn_batch_size]
		return {'m': m, 'v': v}
	def __get_parameters_z_t__(self, z, scale=1.0):
		v = self.v_prior_z_2  * self.__logistic__(z[self.N_train:self.input_parser.num_weights])
		m = z[0:self.N_train]
		return {'m': m, 'v': v}
	def __get_parameters_f_hat__(self, q):
		v = 1.0 / (1.0/self.N*(1.0/q['v']-1.0/self.v_prior))
		m = 1.0 / self.N* q['m'] / q['v'] * v
		return {'m': m, 'v': v}
	def __get_parameters_f_hat_z__(self, q):
		v = 1.0 / (1.0*(1.0/q['v']-1.0/self.v_prior_z_2 ))
		m = 1.0 * q['m'] / q['v'] * v
		return {'m': m, 'v': v}
	
	def __log_normalizer__(self, q): 
		return torch.sum(0.5*torch.log(q['v']*2*math.pi) + 0.5*q['m']**2/q['v']) # First summation term of (11)

	def __log_likelihood_factor__(self, samples_q, v_noise, X, inputs, y):
		# Account for occasions where we're optimizing the latent weighting distributions
		arr = torch.zeros([self.num_drawn_samples,self.bnn_batch_size,self.num_state_dims])
		for i in range(self.num_drawn_samples):
			arr[i,:,:] = self.__predict__(samples_q[i,:], torch.hstack([X, inputs[i,:].unsqueeze(1)]))
		return (-0.5*torch.log(2*math.pi*v_noise)) - (0.5*(torch.tensor(np.tile(np.expand_dims(y,axis=0), (self.num_drawn_samples,1,1)))-arr)**2)/v_noise
	
	def __log_Z_prior__(self):
		return (self.num_weights) * 0.5 * np.log(self.v_prior*2*math.pi)

	def __log_Z_likelihood__(self, q,z, f_hat,f_hat_z, v_noise, X, y):
		samples = self.__draw_samples__(q)
		samples_z = self.__draw_samples__(z)
		log_f_W = torch.sum(-0.5/f_hat['v']*samples**2 + f_hat['m']/f_hat['v']*samples, 1) 
		log_f_z = -0.5/f_hat_z['v']*samples_z**2 + f_hat_z['m']/f_hat_z['v']*samples_z
		log_f_W_tiled = log_f_W.repeat(self.bnn_batch_size,1).T
		if self.num_state_dims> 1:
			log_factor_value = self.alpha * (torch.sum(self.__log_likelihood_factor__(samples, v_noise, X, samples_z, y).squeeze(),dim=2) - log_f_z - log_f_W_tiled )
		else:
			log_factor_value = self.alpha * (self.__log_likelihood_factor__(samples, v_noise, X, samples_z, y).squeeze() - log_f_z - log_f_W_tiled )
		return torch.sum(torch.logsumexp(log_factor_value +np.log(1.0/self.num_drawn_samples), dim=0)) 

	def energy(self, weights,  X, inputs, y):
		v_noise = torch.exp(self.parser.get(weights, 'log_v_noise')[0,0])
		q = self.__get_parameters_q__(weights)
		z = self.__get_parameters_z__(inputs)
		f_hat = self.__get_parameters_f_hat__(q)
		f_hat_z = self.__get_parameters_f_hat_z__(z)
		return  - 1.0 * self.N / (X.shape[ 0 ] * self.alpha)*self.__log_Z_likelihood__(q,z, f_hat,f_hat_z, v_noise, X,y) - self.N/X.shape[0]*self.__log_normalizer__(z) - self.__log_normalizer__(q) + self.__log_Z_prior__()
	
	def feed_forward(self, X):
		q = self.__get_parameters_q__(self.weights)
		samples_q = self.__draw_samples__(q)
		z = self.__get_parameters_z_t__(torch.tensor(self.inputs))	
		samples_z = self.__draw_samples__(z)
		outputs = self.__predict__(samples_q[0,:], torch.hstack([torch.tensor(X), samples_z[0,:].unsqueeze(1)])).squeeze()
		return outputs

	def get_td_error(self, X, y, location=0.0, scale=1.0, by_dim=False):
		# Compute the L2 norm of the error for each transition tuple in X
		outputs = self.feed_forward(X, 1.0).flatten()
		if by_dim:
			return (y-outputs)**2
		return np.sqrt(np.mean((outputs.detach().numpy()-y[:,0])**2))


	def fit_network(self, X, y):
		X = torch.tensor(X)
		y = torch.tensor(y)
		# Create a target tensor with the same shape as the source tensor
		weights = self.weights.clone()
		# Copy the values from the source tensor to the target tensor
		inputs = self.inputs.copy()
		weights = weights.clone().detach().requires_grad_(True)

		for epoch in range(self.train_epochs):
			# Gather a sample of data from the experience buffer, convert to input and target tensors
			self.N = X.shape[0]
			batch_idxs = self.__make_batches__()
			# Permute the indices of the training inputs for SGD purposes
			permutation = np.random.permutation(X.shape[0])
			for idxs in batch_idxs:
				dum = torch.tensor(np.hstack([inputs[permutation[idxs]], inputs[permutation[idxs] + self.N_train]]),requires_grad=True)
				params = [weights,dum]
				optimizer = optim.Adam(params, lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
				optimizer.zero_grad()
				loss = self.energy(weights, X[permutation[idxs]], dum, y[permutation[idxs]])
				loss.backward()
				# print(loss)
				optimizer.step()
				self.inputs[permutation[idxs]] = dum[0:self.bnn_batch_size].detach().numpy().copy()
				self.inputs[permutation[idxs]+self.N_train] = dum[self.bnn_batch_size:dum.shape[0]].detach().numpy().copy()
				self.weights = weights.clone()
		