import numpy as np
import matplotlib.pyplot as plt
import random
import sys, math
import pandas as pd
from optproblems import cec2005
from pathlib import Path
from copy import deepcopy
from utils import write_parameters_de, write_performance, write_evolution


class Bee(object):
	#initialize a bee
	"""A Bee requires three main tasks"""
	def __init__(self, function, error):
		self.function = function
		self.min_value, self.max_value = function.min_value, function.max_value
		self.dim = function.dim
		self.max_obj = function.max_obj
		self.xi = function.generate_random_position()
		self.fitness = function.evaluate(self.xi, error)
		self.trial = 0
		self.prob = 0

	#evaluate if a position belongs to the boundary space
	def evaluate_decision_boundary(self, current_pos):
		return np.clip(current_pos, self.min_value, self.max_value)

	# updates the current position, if current fitness is better than the old fitness
	def update_bee(self, pos, fitness):
		check_update = fitness>self.fitness if self.max_obj else fitness < self.fitness
		if (check_update):
			self.fitness = fitness
			self.xi = pos
			self.trial = 0
		else:
			self.trial+=1

	# when food source is abandoned (e.g.; self.trial > MAX), this generates a random food source e send be to there.  
	def reset_bee(self, max_trial, error):
		if (self.trial > max_trial):
			self.xi = self.function.generate_random_position()
			self.fitness = self.function.evaluate(self.xi, error)
			self.trial = 0

 
class EmployeeBee(Bee):
	def explore(self, max_trial, bee_idx, swarm, error):
		idxs = [idx for idx in range(len(swarm)) if idx!=bee_idx]
		if (self.trial <= max_trial):
			phi = np.random.uniform(low=-1, high=1, size=self.dim)
			other_bee = swarm[random.choice(idxs)]
			new_xi = self.xi + phi*(self.xi - other_bee.xi)
			new_xi = self.evaluate_decision_boundary(new_xi)
			new_fitness = self.function.evaluate(new_xi, error)
			self.update_bee(new_xi, new_fitness)
		else:
			self.reset_bee(max_trial, error)


	def get_fitness(self):
		return 1/(1+self.fitness) if self.fitness >= 0 else 1+abs(self.fitness)

	def compute_probability(self, max_fitness):
		self.prob = self.get_fitness()/max_fitness

class OnlookBee(Bee):
	def onlook(self, best_food_sources, max_trials, error):
		candidate = np.random.choice(best_food_sources)
		self.exploit(candidate.xi, candidate.fitness, max_trials, error)

	def exploit(self, candidate, fitness, max_trials, error):
		if (self.trial <= max_trials):
			component = np.random.choice(candidate)
			phi = np.random.uniform(low=-1, high=1, size=len(candidate))
			n_pos = candidate + phi*(candidate - component)
			n_pos = self.evaluate_decision_boundary(n_pos)
			n_fitness = self.function.evaluate(n_pos, error)
			check_update = n_fitness > self.fitness if self.max_obj else n_fitness < self.fitness
			if (check_update):
				self.fitness = n_fitness
				self.xi = n_pos
				self.trial = 0
			else:
				self.trial+=1

class ABC(object):
	def __init__(self, function, colony_size, dim, optimum, maxFes=10000, max_trials=100, max_obj=False, epsilon=10**(-8)):
		self.function = function
		self.colony_size = colony_size
		self.max_trials = max_trials
		self.epsilon = epsilon
		self.dim = dim
		self.maxFes = dim*maxFes
		self.optimal_solution = None
		self.optimality_tracking = []
		self.optimum = optimum
		self.max_obj = max_obj

		self.maxFes = 10000*dim
		self.fes_list = self.maxFes*np.array([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


	def reset_algorithm(self):
		self.optimal_solution = None
		self.optimality_tracking = []

	
	def update_optimality_tracking(self):
		self.optimality_tracking.append(self.optimal_solution)

	def initialize_employees(self):
		self.employee_bees = [EmployeeBee(self.function, self.best_error) for idx in range(self.colony_size // 2)]

	def update_optimal_solution(self):
		#print(min(self.onlookers_bees + self.employee_bees, key=lambda bee: bee.fitness))
		swarm_fitness_list = []
		for bee in (self.onlookers_bees + self.employee_bees):
			swarm_fitness_list.append(bee.fitness)

		n_optimal_solution = max(swarm_fitness_list) if self.max_obj else min(swarm_fitness_list)
		#n_optimal_solution = min(self.onlookers_bees + self.employee_bees, key=lambda bee: bee.fitness)
		if not self.optimal_solution:
			self.optimal_solution = deepcopy(n_optimal_solution)
		else:
			if n_optimal_solution < self.optimal_solution:
				self.optimal_solution = deepcopy(n_optimal_solution)


	def initialize_onlookers(self):
		self.onlookers_bees = [OnlookBee(self.function, self.best_error) for idx in range(self.colony_size // 2)]


	def employee_bee_phase(self):
		for i, bee in enumerate(self.employee_bees):
			bee.explore(self.max_trials, i, self.employee_bees, self.best_error)
		#map(lambda idx, bee: bee.explore(self.max_trials, idx, self.employee_bees), self.employee_bees)

	def calculate_probabilities(self):
		sum_fitness = sum(map(lambda bee: bee.get_fitness(), self.employee_bees))
		#map(lambda bee: bee.compute_probability(sum_fitness), self.employee_bees)
		for bee in self.employee_bees:
			bee.compute_probability(sum_fitness)

	def select_best_food_sources(self):

		self.best_food_sources = []
		while (len(self.best_food_sources))==0:
			self.best_food_sources = [bee for bee in self.employee_bees if bee.prob > np.random.uniform(0,1)]
		
		#self.best_food_sources =\
		# filter(lambda bee: bee.prob > np.random.uniform(0,1), self.employee_bees)

		#print(list(self.best_food_sources), len(list(self.best_food_sources)))
		#while len(list(self.best_food_sources))==0:
		#	print("oi")
		#	self.best_food_sources =\
		#	 filter(lambda bee: bee.prob > np.random.uniform(0,1), self.employee_bees)
		#print(list(self.best_food_sources), len(list(self.best_food_sources)))

		#sys.exit()

	def onlookers_bee_phase(self):
		for bee in self.onlookers_bees:
			bee.onlook(self.best_food_sources, self.max_trials, self.best_error)
		
		#map(lambda idx, bee: bee.onlook(self.best_food_sources, self.max_trials), self.onlookers_bees)

	def scout_bee_phase(self):
		map(lambda bee: bee.reset_bee(self.max_trials), self.onlookers_bees + self.employee_bees)


	def optimize(self, show_info=True):
		self.best_error = -np.inf if self.max_obj else np.inf		
		self.count_fes = 0


		self.reset_algorithm()
		self.initialize_employees()
		self.initialize_onlookers()

		best_error_miss = True
		fes_final = None		
		epoch = 0
		best_error  = np.inf


		while (self.function.count_fes < self.maxFes and best_error>self.epsilon):
			self.employee_bee_phase()
			self.update_optimal_solution()

			self.calculate_probabilities()
			self.select_best_food_sources()

			self.onlookers_bee_phase()
			self.scout_bee_phase()

			self.update_optimal_solution()
			self.update_optimality_tracking()

			best_error = abs(self.optimum - self.optimal_solution)
			self.best_error = best_error



			if(show_info):
				print("Epoch: %s, Fes: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.function.count_fes, self.optimal_solution, best_error))
			epoch+=1
		
		print("FINAL: Epoch: %s, Fes: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.function.count_fes, self.optimal_solution, best_error))
		self.function.error_evolution_list.append(best_error)
		success = 1 if best_error < self.epsilon else 0
		return best_error, success, self.function.error_evolution_list, fes_final

class Function_Objective(object):
	def __init__(self, dim, bounds, max_obj=False):
		self.dim = dim
		self.min_value, self.max_value = bounds
		self.max_obj = max_obj
		self.count_fes = 0

	#generate random position
	def generate_random_position(self):
		return np.random.uniform(low=self.min_value, high=self.max_value, size=self.dim)

	def evaluate(self, x):
		pass

class F(Function_Objective):
	def __init__(self, dim, f):
		super(F, self).__init__(dim, [-100.0, 100.0], max_obj=False)
		self.dim = dim
		self.f = f
		self.error_evolution_list = []
		self.maxFes = 100000
		self.fes_list = self.maxFes*np.array([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])




	def evaluate(self, x, error):
		self.count_fes+=1
		if (error!=np.inf):
			self.error_evolution_list.append(error)

		return self.f(x)


n_epoch = 100000
dim = 10
colony_size = 30
optimum = -450
max_trials = 5000
n_runs = 25
best_error_list = []
fes_list = []
error_evolution_list = []
success_count = 0
f = cec2005.F4(dim)
function_name = "F5"
savePathParameters = Path("results/abc_parameters.csv")
savePathReporter = Path("results/abc_statistics_10.csv")
savePathEvolution = Path("results/%s_abc_evolution_error.csv"%(function_name))

for n in range(n_runs):
	print("Runs: %s"%(n))
	function = F(dim, f)
	bee_colony = ABC(function, colony_size, dim, optimum, max_trials=max_trials)
	error, success, error_evolution, fes = bee_colony.optimize(show_info=True)
	best_error_list.append(error)
	error_evolution_list.append(error_evolution)
	fes_list.append(fes)
	success_count+=success

success_rate = success_count/n_runs
print(success_rate)
sys.exit()
write_performance("%s"%(function_name), best_error_list, fes_list, fes_list, colony_size, success_rate, savePathReporter, max_obj=False)
#write_parameters_("F4", n_epochs, colony_size, parameters_dict, savePathParameters, max_obj=False)
write_evolution(error_evolution_list, n_runs, savePathEvolution)

