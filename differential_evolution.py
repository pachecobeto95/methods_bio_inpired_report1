import numpy as np
import matplotlib.pyplot as plt
import random
import sys, math
from utils import plot_contourn, plot_fitness



class DE():
	def __init__(self, function, pop_size, n_epoch, dim, bounds, parameters_dict, optimum, max_obj=False, epsilon=10**(-8),
		show_info=True):
		self.function = function
		self.pop_size = pop_size
		self.n_epoch = n_epoch
		self.dim= dim
		self.min_value, self.max_value = bounds
		self.parameters_dict = parameters_dict
		self.max_obj = max_obj
		self.epsilon = epsilon
		self.optimum = optimum
		self.show_info = show_info
		self.count_fes = 0
		self.maxFes = 10000*dim
		#self.maxFes_list = self.maxFes*(np.array([0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))
		self.fes_list = self.maxFes*np.array([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
		self.error_evolution_list = []

		if (parameters_dict["cross_method"] == "binomial"):
			self.crossover = self.binomial_crossover

		else:
			print("This crossover method has been not implemented")
			sys.exit()


		if (parameters_dict["mutation_method"] == "classic"):
			self.mutation = self.classic_mutation

		elif(parameters_dict["mutation_method"] == "variant"):
			self.mutation = self.variant_mutation

		else:
			print("This crossover method has been not implemented")
			sys.exit()



	def generate_initial_population(self):
		population = []
		for i in range(self.pop_size):
			population.append(np.random.uniform(low=self.min_value, high=self.max_value, size=self.dim))

		return np.array(population)

	
	def classic_mutation(self, pop_idx, population, best_ind):
		impact_factor = self.parameters_dict["impact_factor"]
		idxs = [idx for idx in range(self.pop_size) if idx!=pop_idx]

		a, b, c = population[np.random.choice(idxs, 3, replace=False)]
		if(self.parameters_dict["base"]=="best"):
			a = best_ind

		mutated_vector = np.clip(a + np.random.uniform(0, 1)*(b-c), self.min_value, self.max_value)

		return np.array(mutated_vector)

	def variant_mutation(self, pop_idx, population, best_ind):
		impact_factor = self.parameters_dict["impact_factor"]
		idxs = [idx for idx in range(self.pop_size) if idx!=pop_idx]

		a, b, c, d, e = population[np.random.choice(idxs, 5, replace=False)]

		if(self.parameters_dict["base"]=="best"):
			a = best_ind

		mutated_vector = np.clip(a + np.random.uniform(0, 1)*(b-c) + np.random.uniform(0, 1)*(d-e), self.min_value, self.max_value)

		return np.array(mutated_vector)


	def binomial_crossover(self, mutant_vector, pop, pop_idx):
		cross_rate = self.parameters_dict["cross_rate"]
		cross_points = np.random.rand(self.dim) < cross_rate
		if not np.any(cross_points):
			cross_points[np.random.randint(0, self.dim)] = True

		trial = np.where(cross_points, mutant_vector, pop[pop_idx])
		trial_fitness = self.evaluate(np.array([trial]))[0]
		return np.array(trial), trial_fitness

	def evaluate(self, X):
		z = []
		for cromo in X:
			z.append(self.function(cromo))
			self.count_fes+=1
			if(self.best_error!=np.inf):
				self.error_evolution_list.append(self.best_error)

		return z


	def run(self, show_info=False):
		self.best_error = -np.inf if self.max_obj else np.inf		
		self.count_fes = 0
		population = self.generate_initial_population()
		fitness = self.evaluate(population)                        # evaluate the initial population
		#self.count_fes+=self.pop_size
		best_idx = np.argmax(fitness) if self.max_obj else np.argmin(fitness)
		best_fit = fitness[best_idx]
		best_ind = population[best_idx]
		#if (self.count_fes in self.maxFes_list):
		#error_evolution.append(abs(self.optimum - fitness[best_idx]))
		best_error_miss = True
		fes_final = None		
		epoch = 0
		best_error  = np.inf

		while self.count_fes<=self.maxFes and best_error >  self.epsilon:
			if (show_info):
				print("Epoch: %s, FES: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.count_fes, fitness[best_idx], best_error))

			for i in range(self.pop_size):
				mutated_vector = self.mutation(i, population, best_ind)
				trial_vector, trial_fitness =  self.crossover(mutated_vector,population, i)

				pop_replace, best_replace =self.check_replacement(fitness[i], trial_fitness, fitness[best_idx])

				if (pop_replace):
					fitness[i] = trial_fitness
					population[i] = trial_vector

					if (best_replace):
						best_idx = i
						best_ind = trial_vector

			epoch+=1
			best_error = abs(self.optimum - fitness[best_idx])
			self.best_error = best_error



		self.error_evolution_list.append(self.best_error)
		print("Epoch: %s, FES: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.count_fes, fitness[best_idx], best_error))
		success = 1 if best_error < self.epsilon else 0
		return best_error, success, self.error_evolution_list, self.count_fes

	def check_replacement(self, fit_ind, fit_trial, fit_best):
		pop_replace = fit_trial>fit_ind if self.max_obj else fit_trial  < fit_ind
		best_replace = fit_trial > fit_best if self.max_obj else fit_trial < fit_best
		return pop_replace, best_replace


