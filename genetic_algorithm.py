import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from utils import write_parameters_genetic_alg, write_performance
from optproblems import cec2005
from pathlib import Path


class Genetic_Algorithm():
	def __init__(self, function, n_genes, bounds, pop_size, hiper_parameters_dict, max_fes=10000, optimum=0, epsilon=10**(-8), 
		max_obj=False):
		self.function = function
		self.n_genes = n_genes
		self.min_value, self.max_value = bounds
		self.pop_size = pop_size
		self.hiper_parameters_dict = hiper_parameters_dict
		self.optimum = optimum
		self.epsilon = epsilon
		self.max_obj = max_obj
		self.count_fes = 0
		self.maxFes = 10000*n_genes
		self.patience = 10
		self.alpha = hiper_parameters_dict["alpha"]
		self.cross_rate = hiper_parameters_dict["cross_ratio"]
		self.elitism_bool = hiper_parameters_dict["elitism_bool"]
		self.elitism_number = hiper_parameters_dict["elitism_number"]
		self.mutation_rate = hiper_parameters_dict["mutation_ratio"]
		self.fes_list = 100000*np.array([0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
		self.error_evolution_list = []


		"""
		if (hiper_parameters_dict["selection_method"] == "tournment"):
			self.selection = self.tournment_selection
		else:
			print("This selection method has been not implemented")
			sys.exit()

		if (hiper_parameters_dict["cross_method"] == "blx"):
			self.crossover = self.crossover_blx

		elif (hiper_parameters_dict["cross_method"] == "aritmetic"):
			self.crossover = self.crossover_aritmetic
		else:
			print("This crossover method has been not implemented")
			sys.exit()


		if (hiper_parameters_dict["mutation_method"] == "uniform"):
			self.mutation = self.mutation_uniform

		elif(hiper_parameters_dict["mutation_method"] == "gauss"):
			self.mutation = self.mutation_gauss

		else:
			print("This crossover method has been not implemented")
			sys.exit()
		"""


	def evaluate(self, X):
		z = []
		for cromo in X:
			z.append(self.function(cromo))
			self.count_fes+=1
			if(self.best_error!=np.inf):
				self.error_evolution_list.append(self.best_error)

		return z

	def tournment_selection(self, pop, apt, mating_pool_size=None):
		selected_pop = []
		selected_apt = []

		if mating_pool_size is None:
			mating_pool_size = self.pop_size

		while len(selected_pop) < mating_pool_size:
			idx1, idx2 = np.random.randint(0, len(pop), 2)
			ind1, ind2 = pop[idx1], pop[idx2]
			apt1, apt2 = apt[idx1], apt[idx2]
			if (self.max_obj):
				if (apt1 >= apt2):
					selected, winner = ind1, apt1
				else:
					selected, winner = ind2, apt2
			else:
				if (apt1 <= apt2):
					selected, winner = ind1, apt1
				else:
					selected, winner = ind2, apt2

			selected_pop.append(selected)
			selected_apt.append(winner)

		return selected_pop, selected_apt		


	def generate_initial_population(self, pop_size):
		initial_population = []
		for n in range(pop_size):
			cromossomo = [np.random.uniform(self.min_value, self.max_value) for i in range(self.n_genes)]
			initial_population.append(cromossomo)
		return np.array(initial_population)

	def ordering(self, pop, apt):
		sorted_apt = sorted(apt)
		sorted_pop = np.array(pop)[np.argsort(apt)]

		return list(sorted_pop), list(sorted_apt)

	def blx_crossover(self, pop, apt):
		children = []
		children_apt = []
		for i in range(len(pop)):
			idx1, idx2 = np.random.randint(0, len(pop), 2)
			parent1, parent2 = pop[idx1], pop[idx2]
			apt1, apt2 = apt[idx1], apt[idx2]
			if (np.random.uniform(0, 1) < self.cross_rate):
				while 1:
					beta = np.random.uniform(-self.alpha, 1+self.alpha, size=self.n_genes)
					child = parent1 + beta*(parent1-parent2)

					if (np.all(child <= self.max_value) and np.all(child >= self.min_value)):
						children.append(child)
						children_apt.append(self.evaluate([child])[0])
						break
			
			else:
				child = parent1 if apt1 <= apt2 else parent2
				child_apt = apt1 if apt1 <= apt2 else apt2
				
				children.append(child)
				children_apt.append(child_apt)

		return children, children_apt 


	def mutation_uniform(self, pop, apt):
		new_pop = []
		new_apt = []
		for i in range(len(pop)):
			new_cromo = []
			for j in range(self.n_genes):
				if (np.random.uniform(0, 1) < self.mutation_rate):
					new_cromo.append(np.random.uniform(-100, 100))
				else:
					new_cromo.append(pop[i][j])
			
			new_pop.append(new_cromo)
			if new_cromo == list(pop[i]):
				new_apt.append(apt[i])
			else:
				new_apt.append(self.evaluate([new_cromo])[0])
		
		return new_pop, new_apt


	def creep_mutation(self, pop, apt):
		new_pop = []
		new_apt = []
		for i in range(len(pop)):
			while 1:
				new_cromo = [gene+np.random.normal(0, 1) if (np.random.uniform(0, 1) < 1) else gene for gene in pop[i]]
				if (np.all(np.array(new_cromo)>=self.min_value) and np.all(np.array(new_cromo)<=self.max_value)):
					if (np.all(np.array(new_cromo) != pop[i])):
						new_pop.append(new_cromo)
						new_apt.append(self.evaluate([new_cromo])[0])
						break

		return pop, apt


	def select_survivor(self, pop, apt, mutated_pop, mutated_apt):
		new_pop = []
		new_apt = []
		new_pop.extend(list(pop))
		new_apt.extend(list(apt))
		new_pop.extend(mutated_pop)
		new_apt.extend(mutated_apt)

		sorted_pop, sorted_apt = np.array(new_pop)[np.argsort(new_apt)], sorted(new_apt)
		sorted_pop, sorted_apt = sorted_pop[:self.pop_size], sorted_apt[:self.pop_size]
		return sorted_pop, sorted_apt


	def run(self, show_info=True):
		self.best_error = -np.inf if self.max_obj else np.inf
		pop = self.generate_initial_population(self.pop_size)
		apt = self.evaluate(pop)
		epoch = 0
		best_idx = np.argmin(apt)
		self.best_error = best_error = abs(apt[best_idx] - self.optimum)
		stuck = 0
		best_error_miss = True
		fes_final = None

		while (self.count_fes <= self.maxFes and best_error > self.epsilon):
			if (show_info):
				print("Epoch: %s, Fes: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.count_fes, apt[best_idx], best_error))

			sorted_pop, sorted_apt = self.ordering(pop, apt) 
			elite_pop, elite_apt = sorted_pop[:self.elitism_number], sorted_apt[:self.elitism_number]			
			mating_pool, mating_apt = self.tournment_selection(pop, apt)
			cross_pop, cross_apt = self.blx_crossover(mating_pool, mating_apt)
			mutated_pop, mutated_apt = self.mutation_uniform(cross_pop, cross_apt)
	
			pop, apt = self.select_survivor(pop, apt, mutated_pop, mutated_apt)
			if (self.elitism_bool):
				list(pop).extend(elite_pop)
				list(apt).extend(elite_apt)

			best_idx = np.argmax(apt) if self.max_obj else np.argmin(apt)
			best_error = abs(apt[best_idx] - self.optimum)
			self.best_error = best_error
			
			epoch+=1


		self.error_evolution_list.append(self.best_error)
		print("FINAL: Epoch: %s, Fes: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.count_fes, apt[best_idx], best_error))
		success = 1 if best_error < self.epsilon else 0
		return best_error, success, self.error_evolution_list, self.count_fes

