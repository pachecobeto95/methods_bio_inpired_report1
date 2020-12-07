import random
import math
import numpy as np, os, sys
from optproblems import cec2005



class Particle():
	def __init__(self, dim, bounds, method_init, max_obj=False):
		self.dim = dim
		self.min_value, self.max_value = bounds
		self.xi = np.random.uniform(self.min_value, self.max_value, size=dim)
		self.max_obj = max_obj
		self.pbest = -np.inf if max_obj else np.inf 
		self.pbest_xi = None
		self.fit = None


		if (method_init == "zero"):
			self.vi = np.zeros(self.dim)

		elif (method_init == "lower"):
			self.vi = np.random.uniform(-.1, .1, size=dim)

		elif (method_init == "uniform"):
			self.vi = np.random.uniform(-40, 40, size=dim)

		else:
			print("This initial velocity method is not implemented")
			sys.exit()

	# evaluate current fitness
	def evaluate(self, function):
		self.fit = function(self.xi)
		# change the personal best if the current fitness is better

		check_replacement = self.fit > self.pbest if self.max_obj else self.fit < self.pbest  
		if (check_replacement):
			self.pbest_xi = self.xi
			self.pbest = self.fit

	def update_position(self):
		self.xi = self.xi + self.vi
		if ((self.xi < self.min_value).all() or  (self.xi > self.max_value).all()):
			self.vi = -self.vi
		self.xi = np.clip(self.xi, self.min_value, self.max_value)

	def update_velocity(self, gbest_xi, inertia, cognitive_const=2.05, social_const=2.05):
		r1, r2 = np.random.uniform(0, 1, size=self.dim), np.random.uniform(0, 1, size=self.dim)
		self.vi = (inertia*self.vi + cognitive_const*r1*(self.pbest_xi-self.xi) + social_const*r2*(gbest_xi-self.xi) )
		#self.vi = np.clip(self.vi, -40, 40)
		#print(self.vi)

	#def update_clerck-velocity():

class PSO():
	def __init__(self, function, pop_size, n_epochs, dim, bounds, parameters_dict, optimum, max_obj=False, epsilon=10**(-8), patience=100):
		self.function = function
		self.pop_size = pop_size
		self.n_epochs = n_epochs
		self.dim = dim
		self.bounds = bounds
		self.min_value, self.max_value = self.bounds
		self.max_obj = max_obj
		self.method_init = parameters_dict["method_init"]
		self.inertia = parameters_dict["inertia"]
		self.cognitive_const = parameters_dict["cognitive_const"]
		self.social_const = parameters_dict["social_const"]
		self.epsilon = epsilon
		self.optimum = optimum
		self.patience = patience
		self.count_fes = 0
		self.max_fes = 10000*dim
		#self.maxFes_list = self.max_fes*(np.array([0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))
		self.error_evolution_list = []
		self.fes_list = self.max_fes*np.array([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])



	def run(self, show_info=True):
		self.best_error = np.inf		
		swarm = [Particle(self.dim, self.bounds, self.method_init) for i in range(self.pop_size)]
		gbest = -np.inf if self.max_obj else np.inf
		gbest_xi = None
		epoch = 0
		best_error = np.inf
		stuck = 0
		error_evolution = []
		best_error_miss = True
		fes_final = None		
		epoch = 0
		best_error  = np.inf

		while (self.count_fes <= self.max_fes and best_error > self.epsilon):
			if (show_info):
				print("Epoch: %s, FES: %s, Best Error: %s"%(epoch, self.count_fes, best_error))
			
			for i in range(self.pop_size):
				swarm[i].evaluate(self.function)
				self.count_fes+=1
				if(best_error!=np.inf):
					self.error_evolution_list.append(self.best_error)

				check_replacement = swarm[i].fit > gbest if self.max_obj else swarm[i].fit < gbest
				if (check_replacement):
					gbest = swarm[i].fit
					gbest_xi = swarm[i].xi


				#for j in range(self.pop_size):
				swarm[i].update_velocity(gbest_xi, self.inertia, self.cognitive_const, self.social_const)
				swarm[i].update_position() 


			past_best_error = best_error
			best_error = abs(gbest-self.optimum)

			#if (self.count_fes in self.maxFes_list):
			error_evolution.append(best_error)			

			if past_best_error == best_error: stuck += 1
			if (stuck > self.patience):
				stuck=0
				swarm = [Particle(self.dim, self.bounds, self.method_init) for i in range(self.pop_size)]


			self.best_error = best_error
			#print(self.count_fes, best_error < self.epsilon, fes_final)
			
			epoch+=1

		self.error_evolution_list.append(best_error)
		print("FINAL: Epoch: %s, FES: %s, Best Error: %s"%(epoch, self.count_fes, best_error))
		success = 1 if best_error < self.epsilon else 0
		return best_error, success, self.error_evolution_list, self.count_fes


