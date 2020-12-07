import numpy as np
import matplotlib.pyplot as plt
import random
import sys, math
import pandas as pd, os

def plot_contourn(function, ind=None, apt=0):
	x = np.linspace(-5, 10, 1000) 
	y = np.linspace(-5, 10, 1000)
	X, Y = np.meshgrid(x, y)
	print(x)
	
	Z = function(X, Y)
	plt.contour(X, Y, Z, colors='black')
	
	if (ind is not None):
		plt.scatter([ind[0]], [ind[1]], color="red")
	plt.show()

def plot_fitness(best_apt, worst_apt, mean_apt, n_epoch, savefig):
	fig, ax = plt.subplots()
	plt.plot(range(n_epoch), best_apt, label="Best")
	plt.plot(range(n_epoch), worst_apt, label="Worst")
	plt.plot(range(n_epoch), mean_apt, label="Mean")
	plt.legend(fontsize=18)
	plt.xlabel("Epochs", fontsize=16)
	plt.ylabel("Fitness", fontsize=16)
	plt.tight_layout()
	plt.savefig(savefig)



def write_performance(function, best_error_list, n_epoch, fes_list, pop_size, success_rate, savePath, max_obj=False):
	columns = ["best_epoch", "worst_epoch", "pop_size", "best", "worst", "mean", "median", "std", "success_rate"]
	

	df1 = pd.DataFrame(columns=columns)
	df1.loc[0, "function"] = function
	df1.loc[0, "best_epoch"] = n_epoch[np.argmax(best_error_list) if max_obj else np.argmin(best_error_list)]
	df1.loc[0, "worst_epoch"] = n_epoch[np.argmin(best_error_list) if max_obj else np.argmax(best_error_list)]
	#.loc[0, "best_fes"] = fes_list[np.argmax(best_error_list) if max_obj else np.argmin(best_error_list)]
	#df1.loc[0, "worst_fes"] = fes_list[np.argmin(best_error_list) if max_obj else np.argmax(best_error_list)]
	df1.loc[0, "pop_size"] = pop_size
	df1.loc[0, "best"] = max(best_error_list) if max_obj else min(best_error_list)
	df1.loc[0, "worst"] = min(best_error_list) if max_obj else max(best_error_list)
	df1.loc[0, "mean"] = np.mean(best_error_list)
	df1.loc[0, "median"] = np.median(best_error_list)
	df1.loc[0, "std"] = np.std(best_error_list)
	df1.loc[0, "success_rate"] = success_rate
	
	if (os.path.exists(savePath)):
		df = pd.read_csv(savePath)
		df = df.append(df1, ignore_index=True)

	else:
		df = df1

	df.to_csv(savePath)

def write_parameters_acor(function, pop_size, parameters_dict, savePath, max_obj=False):
	columns = ["function", "pop_size"]
	columns = columns + list(parameters_dict.keys())
	df1 = pd.DataFrame(columns=columns)
	df1.loc[0, "pop_size"] = pop_size
	df1.loc[0, "function"] = function
	for key in parameters_dict.keys():
		df1.loc[0, key] = parameters_dict[key]

	if (os.path.exists(savePath)):
		df = pd.read_csv(savePath)
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
		df = df.append(df1, ignore_index=True)

	else:
		df = df1

	df.to_csv(savePath)



def write_parameters_genetic_alg(function, pop_size, hiper_parameters_dict, savePath, max_obj=False):
	columns = ["function", "pop_size", "cross_method", "cross_ratio", "alpha", "mutation_method", "mutation_ratio",
	"elitism_bool", "selection_method"]
	df1 = pd.DataFrame(columns=columns)
	df1.loc[0, "pop_size"] = pop_size
	df1.loc[0, "function"] = function
	for key in hiper_parameters_dict.keys():
		df1.loc[0, key] = hiper_parameters_dict[key]

	if (os.path.exists(savePath)):
		df = pd.read_csv(savePath)
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
		df = df.append(df1, ignore_index=True)

	else:
		df = df1

	df.to_csv(savePath)

def write_parameters_de(function, epoch, pop_size, parameters_dict, savePath, max_obj=False):
	columns = ["function", "epochs", "pop_size", "cross_method", "cross_rate", "mutation_method", "base", "impact_factor"]

	df1 = pd.DataFrame(columns=columns)
	df1.loc[0, "epochs"] = epoch
	df1.loc[0, "pop_size"] = pop_size
	df1.loc[0, "function"] = function
	for key in parameters_dict.keys():
		df1.loc[0, key] = parameters_dict[key]

	if (os.path.exists(savePath)):
		df = pd.read_csv(savePath)
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
		df = df.append(df1, ignore_index=True)

	else:
		df = df1

	df.to_csv(savePath)

def write_parameters_pso(function, epoch, pop_size, parameters_dict, savePath, max_obj=False):
	columns = ["function", "method_init", "inertia", "cognitive_const", "social_const"]

	df1 = pd.DataFrame(columns=columns)
	df1.loc[0, "epochs"] = epoch
	df1.loc[0, "pop_size"] = pop_size
	df1.loc[0, "function"] = function
	for key in parameters_dict.keys():
		df1.loc[0, key] = parameters_dict[key]

	if (os.path.exists(savePath)):
		df = pd.read_csv(savePath)
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
		df = df.append(df1, ignore_index=True)

	else:
		df = df1

	df.to_csv(savePath)

def write_evolution(error_evolution_list, n_runs, savePath):
	cont = 0
	fes = 100000
	error_dict = {}
	for i in range(n_runs):
		#print(len(error_evolution_list[i]))
		if (len(error_evolution_list[i]) > fes):
			error_run = error_evolution_list[i][:fes]
		else:
			error_run = error_evolution_list[i] + [error_evolution_list[i][-1]]*(fes - len(error_evolution_list[i]))
		
		#print(len(error_run))
		error_dict["Run_%s"%(i)] = error_run

	df = pd.DataFrame(error_dict)
	df.to_csv(savePath)