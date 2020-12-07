import numpy as np
import random
import sys, math
from differential_evolution import DE
import pandas as pd
from utils import write_parameters_de, write_performance, write_evolution
from optproblems import cec2005
from pathlib import Path


def Func_obj(X):
	z = []
	for ind in X:
		z.append(function1(ind))
	return z


bounds = [-100, 100]
n_epochs = 10000
pop_size = 30
dim = 10
cross_rate = 0.7
impact_factor = 0.9
mutation_method = "variant"
crossover_method = "binomial"
max_obj = False
optimum = -310.0
n_runs = 25
parameters_dict = {"cross_method": crossover_method, "cross_rate": cross_rate,
"mutation_method": mutation_method, "base": "best", "impact_factor":impact_factor}
savePathParameters = Path("results/de_parameters.csv")
savePathReporter = Path("results/de_statistics_10.csv")
savePathEvolution = Path("results/f5_de_evolution_error.csv")

function1 = cec2005.F5(dim)

error_list = []
epoch_list = []
fes_list = []
error_evolution_list = []
success_count = 0
for n in range(n_runs):
	print("Run: %s"%(n))
	de = DE(function1, pop_size, n_epochs, dim, bounds, parameters_dict, optimum=optimum, max_obj=False)
	error, success, error_evolution, fes = de.run(show_info=False)
	error_list.append(error)
	fes_list.append(fes)
	error_evolution_list.append(error_evolution)
	success_count+=success

success_rate = success_count/n_runs

print(success_rate)
sys.exit()
#write_performance("F5", error_list, fes_list, fes_list, pop_size, success_rate, savePathReporter, max_obj=False)
#write_parameters_de("F5", n_epochs, pop_size, parameters_dict, savePathParameters, max_obj=False)
write_evolution(error_evolution_list, n_runs, savePathEvolution)
