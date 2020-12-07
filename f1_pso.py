import random
import math
import numpy as np, os, sys
from optproblems import cec2005
from particle_swarm import PSO
from pathlib import Path
from utils import write_parameters_pso, write_performance, write_evolution

dim = 10
bounds = [-100, 100]
method_init = "zero"
f = cec2005.F5(dim)
n_runs = 25
pop_size = 100
n_epochs = 100000
global_min = -310
inertia_max = 0.9
inertia_min = 0.2
cognitive_const = 2.05
social_const = 2.05
parameters_dict = {"method_init": "zero", "inertia":.2, "cognitive_const":cognitive_const, "social_const":social_const}

best_error_list = []
fes_list = []
error_evolution_list = []
success_count = 0
savePathParameters = Path("results/pso_parameters.csv")
savePathReporter = Path("pso_statistics_10.csv")
savePathEvolution = Path("results/f5_pso_evolution_error.csv")

for n in range(n_runs):
	print("Runs: %s"%(n))
	ps = PSO(f, pop_size, n_epochs, dim, bounds, parameters_dict, optimum=global_min)
	best_error, success, error_evolution, fes = ps.run(show_info=True)
	best_error_list.append(best_error)
	error_evolution_list.append(error_evolution)
	fes_list.append(fes)
	success_count+=success

success_rate = success_count/n_runs
print("Success Rate: %s"%(success_rate))
write_performance("F5", best_error_list, fes_list, fes_list, pop_size, success_rate, savePathReporter, max_obj=False)
#write_parameters_pso("F5", n_epochs, pop_size, parameters_dict, savePathParameters, max_obj=False)
write_evolution(error_evolution_list, n_runs, savePathEvolution)
