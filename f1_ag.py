import numpy as np
import matplotlib.pyplot as plt
import random
import sys, math
import pandas as pd
from genetic_algorithm import Genetic_Algorithm
from utils import write_parameters_genetic_alg, write_performance, write_evolution
from optproblems import cec2005
from pathlib import Path




savePathParameters = Path("results/ag_parameters_final.csv")
savePathReporter = Path("results/ag_statistics_10_final.csv")
savePathEvolution = Path("results/f4_ag_error_evolution_final.csv")

bounds = [-100, 100]
n_epochs = 10000
n_epochs = 10000
pop_size = 100
n_genes = 10
crossover_ratio = 1
alpha = 0.5 
mutation_ratio = 0.01
elitism_bool = True
elitism_number = 1
f1 = cec2005.F4(n_genes)
global_min = -450.0
hiper_parameters_dict = {"cross_method": "blx", "cross_ratio": crossover_ratio, "alpha": alpha,
"mutation_method": "uniform", "mutation_ratio": mutation_ratio, "elitism_bool": elitism_bool, "elitism_number": elitism_number, 
"selection_method": "tournment"}

#function1 = cec2005.F1(n_genes)

n_runs = 10
error_list = []
success_count = 0
fes_list = []
error_evolution_list = []
for n in range(n_runs):
	ag = Genetic_Algorithm(f1, n_genes, bounds, pop_size, hiper_parameters_dict, optimum=global_min)
	print("Run: %s"%(n))
	error, success, error_evolution, fes = ag.run(show_info=True)
	error_list.append(error)
	fes_list.append(fes)
	success_count+=success
	error_evolution_list.append(error_evolution)
	print("Success" if success else "Fail")

success_rate = success_count/n_runs
print("success_rate")
print(success_rate)
#write_performance("F3", error_list, fes_list, error_evolution_list, pop_size, success_rate, savePathReporter, max_obj=False)
#write_parameters_genetic_alg("F3", pop_size, hiper_parameters_dict, savePathParameters, max_obj=False)
write_evolution(error_evolution_list, n_runs, savePathEvolution)
