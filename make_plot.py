import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, sys
from pathlib import Path


fes = [100, 1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 99999]
fontsize = 16
ag_color = "steelblue"
de_color = "salmon"
pso_color = "goldenrod"
abc_color = "darkviolet"
acor_color = "black"
sso_color="green"



for f in [5]:

	agPath = Path("results/f%s_ag_error_evolution_final.csv"%(f))
	dePath = Path("results/f%s_de_evolution_error.csv"%(f))
	psoPath = Path("results/f%s_pso_evolution_error.csv"%(f))
	abcPath = Path("results/F%s_abc_evolution_error.csv"%(f))
	acorPath = Path("results/F%a_acor_evolution_error.csv"%(f))
	ssoPath = Path("results/f%a_sso_evolution_error.csv"%(f))

	saveag = Path("graphic/f%s.png"%(f))
	saveagpdf = Path("graphic/f%s.pdf"%(f))

	df_ag = pd.read_csv(agPath)
	df_de = pd.read_csv(dePath)
	df_pso = pd.read_csv(psoPath)
	df_abc = pd.read_csv(abcPath)
	df_acor = pd.read_csv(acorPath)
	df_sso = pd.read_csv(ssoPath)


	ag = np.mean(np.array([df_ag.iloc[:, i].values[fes] for i in range(1, df_ag.shape[1])]), axis=0)
	de = np.mean(np.array([df_de.iloc[:, i].values[fes] for i in range(1, df_de.shape[1])]), axis=0)
	pso = np.mean(np.array([df_pso.iloc[:, i].values[fes] for i in range(1, df_pso.shape[1])]), axis=0)
	abc = np.mean(np.array([df_abc.iloc[:, i].values[fes] for i in range(1, df_abc.shape[1])]), axis=0)
	acor = np.mean(np.array([df_acor.iloc[:, i].values[fes] for i in range(1, df_acor.shape[1])]), axis=0)
	sso = np.mean(np.array([df_sso.iloc[:, i].values[fes] for i in range(1, df_sso.shape[1])]), axis=0)
	
	fig, ax = plt.subplots()
	plt.plot(np.array(fes), ag, marker="o", color=ag_color, label="GA")
	plt.plot(np.array(fes), de, marker="x", color=de_color, label="DE")
	plt.plot(np.array(fes), pso, marker="d", color=pso_color, label="PSO")
	plt.plot(np.array(fes), abc, marker="*", color=abc_color, label="ABC")
	plt.plot(np.array(fes), acor, marker="p", color=acor_color, label="ACOR")
	plt.plot(np.array(fes), sso, marker="p", color=sso_color, label="SSO")
	plt.plot(np.array(fes), (10**(-8))*np.ones(len(fes)), marker="s", color="red", label="Limiar de Sucesso")


	plt.legend(frameon=False, fontsize=fontsize-4, ncol=3)
	plt.xlabel("Número de Avaliações", fontsize=fontsize+2)
	plt.ylabel("Erro Médio", fontsize=fontsize+2)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.xlim(0, 100000)
	plt.ylim(10**(-9), 10**9)
	plt.yscale("log")
	plt.tight_layout()
	plt.savefig(saveag)
	plt.savefig(saveagpdf)